import argparse
import dataclasses
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT as HyperbolicGPT
from model import GPTConfig as HyperbolicGPTConfig
from model_baseline import GPT as BaselineGPT
from model_baseline import GPTConfig as BaselineGPTConfig
from utils.representation_metrics import isotropy_metrics, stack_activation_batches


MODEL_CONFIG_FIELDS = {
    "baseline": {field.name for field in dataclasses.fields(BaselineGPTConfig)},
    "hyperbolic": {field.name for field in dataclasses.fields(HyperbolicGPTConfig)},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze hidden-state isotropy from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument("--dataset", default=None, help="Dataset name under data/<dataset> (e.g. shakespeare_char, fineweb)")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to sample from")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--model_kind", default="auto", choices=["auto", "baseline", "hyperbolic"], help="Model class to use")
    parser.add_argument("--probe_layers", default="last", help="Comma-separated block indices, or 'last', or 'all'")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Approximate number of tokens to collect per probe layer")
    parser.add_argument("--num_batches", type=int, default=16, help="Max batches to sample")
    parser.add_argument("--batch_size", type=int, default=8, help="Sampling batch size for activation collection")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--output_dir", default="analysis_out", help="Directory to write JSON results")
    parser.add_argument("--output_name", default=None, help="Optional output JSON filename")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def infer_model_kind(checkpoint: dict) -> str:
    state_dict = checkpoint.get("model", {})
    config = checkpoint.get("config", {})
    if any(
        key.startswith("embedding_curvature")
        or ".curvature_predictor." in key
        or key.endswith(".c")
        or key.startswith("shared_curvature")
        for key in state_dict.keys()
    ):
        return "hyperbolic"
    if any(
        key in config
        for key in ["curvature_mode", "dynamic_curvature", "per_head_curvature", "use_embedding_curvature"]
    ):
        return "hyperbolic"
    return "baseline"


def filter_config(config_dict: dict, model_kind: str) -> dict:
    allowed = MODEL_CONFIG_FIELDS[model_kind]
    return {k: v for k, v in config_dict.items() if k in allowed}


def load_checkpoint(checkpoint_path: str, model_kind: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved_kind = infer_model_kind(checkpoint) if model_kind == "auto" else model_kind
    config_dict = dict(checkpoint.get("config", {}))
    config_dict.update(checkpoint.get("model_args", {}))
    config_dict = filter_config(config_dict, resolved_kind)

    if resolved_kind == "baseline":
        model = BaselineGPT(BaselineGPTConfig(**config_dict))
    else:
        model = HyperbolicGPT(HyperbolicGPTConfig(**config_dict))

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}' as {resolved_kind}: {exc}") from exc

    model.eval()
    return model, checkpoint, resolved_kind, missing, unexpected


def load_dataset_tokens(dataset: str, split: str) -> np.memmap:
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    data_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset split: {data_path}")
    return np.memmap(data_path, dtype=np.uint16, mode="r")


def sample_batch(tokens: np.memmap, block_size: int, batch_size: int, device: str, vocab_size: int) -> torch.Tensor:
    max_start = len(tokens) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the requested block size")
    ix = torch.randint(0, max_start, (batch_size,))
    x_np = [np.asarray(tokens[i : i + block_size], dtype=np.int64) for i in ix.tolist()]
    x_np = [np.where(batch >= vocab_size, 0, batch) for batch in x_np]
    x = torch.stack([torch.from_numpy(arr) for arr in x_np])
    return x.to(device, non_blocking=(device.startswith("cuda")))


def parse_probe_layers(probe_layers: str, n_layer: int):
    if probe_layers == "last":
        return [n_layer - 1]
    if probe_layers == "all":
        return list(range(n_layer))
    layers = []
    for item in probe_layers.split(","):
        item = item.strip()
        if item:
            idx = int(item)
            if idx < 0 or idx >= n_layer:
                raise ValueError(f"Probe layer {idx} is out of range for n_layer={n_layer}")
            layers.append(idx)
    if not layers:
        raise ValueError("No probe layers selected")
    return sorted(set(layers))


def collect_activations(model, x, probe_layers):
    activations = OrderedDict()
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            activations[layer_idx] = output.detach().cpu()
        return hook

    transformer = model.transformer
    for layer_idx in probe_layers:
        handles.append(transformer.h[layer_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(x)

    for handle in handles:
        handle.remove()

    return activations


def summarize_metrics(metrics: dict) -> dict:
    summary = {k: v for k, v in metrics.items() if k != "eigenvalues"}
    summary["eigenvalues"] = metrics["eigenvalues"].tolist()
    return summary


def default_output_name(checkpoint_path: str, model_kind: str, split: str) -> str:
    checkpoint_dir = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path))) or "checkpoint"
    safe_dir = checkpoint_dir.replace(os.sep, "_").replace(" ", "_")
    return f"representation_isotropy_{safe_dir}_{model_kind}_{split}.json"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    model, checkpoint, resolved_kind, missing, unexpected = load_checkpoint(args.checkpoint, args.model_kind)
    model = model.to(device)

    config = checkpoint.get("config", {})
    dataset = args.dataset or config.get("dataset")
    if dataset is None:
        raise ValueError("Dataset must be supplied via --dataset or be present in the checkpoint config")

    tokens = load_dataset_tokens(dataset, args.split)
    probe_layers = parse_probe_layers(args.probe_layers, model.config.n_layer)
    block_size = int(model.config.block_size)
    vocab_size = int(model.config.vocab_size)

    collected = {layer_idx: [] for layer_idx in probe_layers}
    tokens_seen = 0

    for _ in range(args.num_batches):
        if tokens_seen >= args.max_tokens:
            break
        x = sample_batch(tokens, block_size=block_size, batch_size=args.batch_size, device=device, vocab_size=vocab_size)
        batch_acts = collect_activations(model, x, probe_layers)
        for layer_idx, activation in batch_acts.items():
            collected[layer_idx].append(activation)
        tokens_seen += x.shape[0] * x.shape[1]

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_kind": resolved_kind,
        "dataset": dataset,
        "split": args.split,
        "probe_layers": probe_layers,
        "tokens_seen": tokens_seen,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "layers": {},
    }

    for layer_idx, batches in collected.items():
        features = stack_activation_batches(batches)
        metrics = isotropy_metrics(features)
        results["layers"][str(layer_idx)] = summarize_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or default_output_name(args.checkpoint, resolved_kind, args.split)
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()

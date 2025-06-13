# Transformer Block Looping Experiment

This project explores iterating transformer blocks multiple times within a single forward pass, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Motivation

Standard transformer models process input through a fixed sequence of layers. This project investigates an alternative approach where specific groups of blocks within the transformer architecture can be applied iteratively. The core idea is to explore whether repeatedly applying the transformations of certain layer groups can lead to deeper processing, potentially improving model performance or efficiency under certain conditions.

This concept is related to recurrent mechanisms and adaptive computation, where the model might dynamically adjust the amount of processing applied. The work in this repo is mainly inspired by https://arxiv.org/pdf/2502.05171

## Project Goal

This project aims to implement and evaluate transformers capable of looping specific groups of blocks. Key aspects include:
*   **Flexible Group Looping:** Allowing configuration of which groups of layer indices to loop together (e.g., `loop_groups: [[0,1],[3]]` would loop layers 0-1 as one group, and layer 3 as another).
*   **Loop Count Control:**
    *   Specifying exact loop counts per group (`loop_counts`).
    *   Setting a maximum number of loops (`max_loops`) which can also be used for sampling loop counts during training.
*   **Input Handling for Loops:**
    *   Optionally concatenating the initial pre-loop group representation with the previous iteration's output (or noise for the first iteration) and adapting it back to the required dimension (`concatenate_initial_representation`).
    *   Alternatively, adding the initial pre-loop group representation to the previous iteration's output.
*   **Noise Injection:** Optionally adding scaled Gaussian noise during the first loop iteration of a group if it's looped multiple times (`loop_noise_scale`). The variance of this noise is based on `2 / (5 * n_embd)`.
*   **Spectral Clipping:**
    *   Using spectral clipping to control the singular values of weight matrices, ensuring they don't exceed a specified threshold (`spectral_clip_beta`).
    *   This is implemented using a Newton-Schulz iteration for efficient orthogonalization.
*   **Analysis & Evaluation:**
    *   Providing tools to track representations across loops for analysis (`loops_representation`).
    *   Evaluating the model not only with sampled loop counts but also with fixed loop counts (e.g., 1, 5, 15, 30 loops) during validation to understand performance sensitivity to loop depth.
    *   Supporting automatic loop exit based on representation convergence (`automatic_loop_exit` and `automatic_loop_exit_threshold`).

The goal is to understand the impact of these looping mechanisms on training dynamics and model capabilities compared to standard transformer architectures.

## Implementation

We use the minimal and efficient nanoGPT implementation as our foundation, modifying the `GPT` model in `model.py` to incorporate the looping functionality described above. The core changes are within the `forward` method of the `GPT` class and the corresponding additions to the `GPTConfig` dataclass. The `train.py` script handles the calculation of `effective_n_layer`, the sampling of loop counts during training, and the multi-faceted evaluation including fixed loop counts.

## Dataset Preparation

For experimenting with transformer block looping, we use the same dataset preparation approach as the original nanoGPT:

### Shakespeare Dataset (Small Scale Testing)

For quick experimentation, the Shakespeare dataset provides a lightweight option:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin` files with character-level tokenization.

### Fineweb Dataset (Full Scale Training)

For more extensive training, prepare the OpenWebText dataset:

```sh
python data/fineweb/prepare.py
```

This downloads and tokenizes the OpenWebText dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

Both datasets are prepared to be used with the training scripts. For transformer block looping experiments, we can use these datasets to compare performance against baseline transformer architectures.

To train, simply run:

```sh
python train.py config/train_fineweb.py --compile=False 
```

You can also run the baseline by simply doing:

```sh
python train.py config/train_fineweb.py --use_model_baseline=True
```

(this can also be compiled)

## Multi-Token Prediction Experiment

This project also includes an experiment for pre-training with multi-token prediction. Instead of always predicting the very next token, the model is trained to predict a token that is `s` steps ahead, where `s` is a shift that is dynamically sampled during training.

### How it Works

*   **Label Shifting**: For each training sequence, a `shift` value `s` is sampled from a geometric distribution. This makes smaller shifts (like `s=1`) more common, while still allowing for a "fat tail" of larger shifts. The maximum shift is configurable.
*   **Conditioning on the Shift**: The model is informed of the `shift` `s` for a given sequence by prepending a special token, `<s>`, to the input. There is a unique token for each possible shift value up to `max_shift`.
*   **Training**: The model learns to use the `<s>` token to adjust its predictions. For an input sequence `[<s>, t_1, t_2, ...]`, the model is trained to predict the target sequence `[t_{1+s}, t_{2+s}, ...]`.

This approach encourages the model to learn more about longer-term dependencies in the data.

### Usage

To enable multi-token prediction during training, set the `predict_ahead` flag:

```sh
python train.py config/train_fineweb.py --predict_ahead=True
```

You can adjust the `max_shift` and `label_shift_prob` parameters in `train.py` to control the behavior of the shift sampling.

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy




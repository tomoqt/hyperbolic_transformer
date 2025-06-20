# saves the fineweb dataset to binary files for training, with caching from huggingface

import os
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pickle

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

def download_cached_file(fname, local_dir):
    """Download a single cached file from huggingface if it doesn't exist locally"""
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir
        )

def process_binary_file(file_path):
    """Read a binary file and return the token ids"""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
    return data.tolist()

if __name__ == '__main__':
    import sys
    # Create local directory if it doesn't exist
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    os.makedirs(local_dir, exist_ok=True)

    # Download validation file
    val_file_name = "fineweb_val_000000.bin"
    print(f"Downloading validation file: {val_file_name}")
    download_cached_file(val_file_name, local_dir)

    # Allow specifying number of chunks via command line argument
    num_chunks = 18  # default: full fineweb10B dataset chunks
    if len(sys.argv) > 1:
        num_chunks = int(sys.argv[1])
        print(f"Will download {num_chunks} chunks instead of full dataset")

    # Download training files
    train_files_names = [f"fineweb_train_{i:06d}.bin" for i in range(1, num_chunks + 1)]
    print(f"Downloading {num_chunks} training chunks...")
    for fname in tqdm(train_files_names, desc="Downloading training chunks"):
        download_cached_file(fname, local_dir)

    # Process and combine data files into single binary files
    data_splits = {
        'train': train_files_names,
        'val': [val_file_name]
    }

    for split, fnames in data_splits.items():
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        
        # Calculate total size of the data
        total_tokens = 0
        for fname in fnames:
            file_path = os.path.join(local_dir, fname)
            total_tokens += os.path.getsize(file_path) // 2 # 2 bytes per token (uint16)

        # Create memory-mapped file
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))

        # Write each chunk to the memmap file
        print(f"Writing {filename}...")
        idx = 0
        for fname in tqdm(fnames, desc=f"Processing {split} chunks"):
            file_path = os.path.join(local_dir, fname)
            chunk_tokens = np.fromfile(file_path, dtype=dtype)
            arr[idx : idx + len(chunk_tokens)] = chunk_tokens
            idx += len(chunk_tokens)
        arr.flush()

    # Create and save meta.pkl
    print("Creating meta.pkl...")
    meta = {
        'vocab_size': enc.n_vocab,
        'itos': {i: enc.decode([i]) for i in range(enc.n_vocab)},
        'stoi': {enc.decode([i]): i for i in range(enc.n_vocab)},
        'encode': enc.encode,
        'decode': enc.decode
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset preparation complete!")
    train_file_out = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_file_out = os.path.join(os.path.dirname(__file__), 'val.bin')
    print(f"Training data saved to: {train_file_out}")
    print(f"Validation data saved to: {val_file_out}")
    print(f"Meta data saved to: {os.path.join(os.path.dirname(__file__), 'meta.pkl')}")
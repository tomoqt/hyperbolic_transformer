# saves the fineweb dataset to binary files for training, with caching from huggingface

import os
import sys
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
    # Create local directory if it doesn't exist
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    os.makedirs(local_dir, exist_ok=True)

    # Download validation file
    val_file = "fineweb_val_000000.bin"
    print(f"Downloading validation file: {val_file}")
    download_cached_file(val_file, local_dir)

    # Allow specifying number of chunks via command line argument
    num_chunks = 18  # default: full fineweb10B dataset chunks
    if len(sys.argv) >= 2:
        num_chunks = int(sys.argv[1])
        print(f"Will download {num_chunks} chunks instead of full dataset")

    # Download training files
    print(f"Downloading {num_chunks} training chunks...")
    for i in tqdm(range(1, num_chunks + 1), desc="Downloading training chunks"):
        train_file = f"fineweb_train_{i:06d}.bin"
        download_cached_file(train_file, local_dir)

    # Process and combine training chunks into a single binary file
    print("Combining training chunks into single binary file...")
    train_tokens = []
    for i in tqdm(range(1, num_chunks + 1), desc="Processing training chunks"):
        train_file = os.path.join(local_dir, f"fineweb_train_{i:06d}.bin")
        train_tokens.extend(process_binary_file(train_file))

    # Process validation file
    val_tokens = process_binary_file(os.path.join(local_dir, val_file))

    # Save combined training data
    train_file = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_file_out = os.path.join(os.path.dirname(__file__), 'val.bin')

    print("Writing combined training data...")
    train_array = np.array(train_tokens, dtype=np.uint16)
    train_memmap = np.memmap(train_file, dtype=np.uint16, mode='w+', shape=train_array.shape)
    train_memmap[:] = train_array[:]
    train_memmap.flush()

    print("Writing validation data...")
    val_array = np.array(val_tokens, dtype=np.uint16)
    val_memmap = np.memmap(val_file_out, dtype=np.uint16, mode='w+', shape=val_array.shape)
    val_memmap[:] = val_array[:]
    val_memmap.flush()

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
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file_out}")
    print(f"Meta data saved to: {os.path.join(os.path.dirname(__file__), 'meta.pkl')}")

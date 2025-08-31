import os
import sys
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_file(fname, local_dir):
    """Downloads a single file if it doesn't already exist."""
    fpath = os.path.join(local_dir, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    else:
        print(f"{fname} already exists. Skipping.")

def main():
    # The full dataset has 103 chunks. Set to a smaller number for a quick test.
    NUM_TRAIN_CHUNKS = 103 
    # The directory to save the data
    LOCAL_DATA_DIR = 'fineweb10B'

    # Create the directory if it doesn't exist
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    print(f"Data will be saved in: {os.path.abspath(LOCAL_DATA_DIR)}")

    # Download the validation set
    download_file("fineweb_val_000000.bin", LOCAL_DATA_DIR)

    # Download the training chunks
    print(f"Downloading {NUM_TRAIN_CHUNKS} training chunks...")
    for i in tqdm(range(1, NUM_TRAIN_CHUNKS + 1), desc="Downloading train chunks"):
        fname = f"fineweb_train_{i:06d}.bin"
        download_file(fname, LOCAL_DATA_DIR)
        
    print("\nDownload complete!")

if __name__ == 'main':
  main()

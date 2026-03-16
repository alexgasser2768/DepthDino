import os
import glob
from PIL import Image
from tqdm import tqdm
import argparse

def clean_dataset(data_dir, min_width=224):
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Using recursive search just in case, though dataloader uses flat
    # But to be safe and thorough, let's stick to what dataloader expects generally 
    # (dataloader.py uses glob.glob(os.path.join(data_dir, "*.jpg")), implying flat structure)
    jpg_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    print(f"Found {len(jpg_files)} images in {data_dir}.")
    
    removed_count = 0
    for jpg_path in tqdm(jpg_files, desc="Checking images"):
        try:
            with Image.open(jpg_path) as img:
                width, height = img.size
                
            if width < min_width or height < min_width:
                # Remove jpg
                os.remove(jpg_path)
                
                # Remove cache
                cache_path = jpg_path.replace(".jpg", ".da3_cache.npz")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                removed_count += 1
        except Exception as e:
            print(f"Error processing {jpg_path}: {e}")

    print(f"Removed {removed_count} images with width < {min_width}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove images below a certain width from the dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the dataset directory")
    parser.add_argument("--min_width", type=int, default=224, help="Minimum width threshold")
    args = parser.parse_args()
    
    clean_dataset(args.data_dir, args.min_width)

import os
import urllib.request
import zipfile
import json
import time

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping download.")
        return

    print(f"Starting download: {url} -> {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    start_time = time.time()
    
    # Custom download with simple progress printing to avoid excessive logging
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB chunks
        
        last_logged_pct = -10
        
        with open(filepath, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                
                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    if pct - last_logged_pct >= 10:
                        elapsed = time.time() - start_time
                        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        print(f"Downloaded: {pct}% ({downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB) - Speed: {speed:.2f} MB/s")
                        last_logged_pct = pct
                        
    print(f"Finished downloading {filepath} in {time.time() - start_time:.1f}s")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} -> {extract_to}...")
    start_time = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete in {time.time() - start_time:.1f}s")

def main():
    base_dir = r"D:\novamind_data\MSVD"
    os.makedirs(base_dir, exist_ok=True)
    
    urls = {
        "msvd_train.json": "https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/msvd_train.json",
        "msvd_val.json": "https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/msvd_val.json",
        "msvd_test.json": "https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/msvd_test.json",
        "MSVD_Videos.zip": "https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/MSVD_Videos.zip"
    }
    
    # 1. Download json files
    for filename, url in urls.items():
        if filename.endswith(".json"):
            filepath = os.path.join(base_dir, filename)
            download_file(url, filepath)
            
    # 2. Download videos zip
    zip_path = os.path.join(base_dir, "MSVD_Videos.zip")
    download_file(urls["MSVD_Videos.zip"], zip_path)
    
    # 3. Extract videos
    videos_dir = os.path.join(base_dir, "youtube_videos")
    # Check if directory already has videos to skip extraction
    if os.path.exists(videos_dir) and len(os.listdir(videos_dir)) > 1000:
        print(f"Videos directory {videos_dir} already contains videos. Skipping extraction.")
    else:
        extract_zip(zip_path, base_dir)
        
    print("MSVD Dataset successfully prepared at:", base_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import subprocess, os, glob, zipfile

def clone_repo(repo_url: str, clone_dir: str):
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

def unzip_all_in(dir_path: str):
    # Recursively find all .zip files
    for zip_path in glob.glob(os.path.join(dir_path, "**", "*.zip"), recursive=True):
        extract_to = zip_path[:-4]  # drop the .zip extension
        os.makedirs(extract_to, exist_ok=True)
        print(f"Unzipping {zip_path} → {extract_to}/")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)

if __name__ == "__main__":
    REPO_URL = "https://github.com/JohnMBrandt/DQ-DETR-DiNOv2.git"
    CLONE_DIR = "DQ-DETR"

    # 1. Clone the repository
    clone_repo(REPO_URL, CLONE_DIR)

    # 2. Unzip every .zip inside it
    unzip_all_in(CLONE_DIR)

    print("All done!")
#!/usr/bin/env python3
"""
Upload benchmark files to HuggingFace Hub.

Usage:
    # First, login to HuggingFace:
    huggingface-cli login
    
    # Then upload:
    python upload_to_hf.py --file pushworld_level0_transformed_all.pkl --repo feynmaniac/pushworld
    
    # Or upload all benchmarks:
    python upload_to_hf.py --all --repo feynmaniac/pushworld
"""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def upload_file(file_path: str, repo_id: str, repo_type: str = "dataset"):
    """Upload a file to HuggingFace Hub."""
    api = HfApi()
    
    filename = os.path.basename(file_path)
    print(f"Uploading {filename} to {repo_id}...")
    
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    
    print(f"✓ Uploaded: https://huggingface.co/datasets/{repo_id}/blob/main/{filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload benchmark files to HuggingFace Hub"
    )
    parser.add_argument(
        "--file", "-f",
        help="Single file to upload"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Upload all benchmark files in current directory"
    )
    parser.add_argument(
        "--repo", "-r", required=True,
        help="HuggingFace repo ID (e.g., feynmaniac/pushworld)"
    )
    parser.add_argument(
        "--dir", "-d", default=".",
        help="Directory containing benchmark files (for --all)"
    )
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        upload_file(args.file, args.repo)
    elif args.all:
        # Find all benchmark pkl files
        import glob
        files = glob.glob(os.path.join(args.dir, "pushworld_*.pkl"))
        if not files:
            print(f"No pushworld_*.pkl files found in {args.dir}")
            sys.exit(1)
        
        print(f"Found {len(files)} files to upload:")
        for f in files:
            print(f"  - {f}")
        
        confirm = input("\nProceed? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
        
        for f in files:
            upload_file(f, args.repo)
    else:
        print("Error: Specify --file or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
import random


def sample_random_files(dir_path, n=5):
    # Get all files (not subdirectories)
    all_files = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]

    # Check that n ≤ total files
    if n > len(all_files):
        raise ValueError(f"Requested {n} files, but only {len(all_files)} available.")

    # Sample without replacement
    sampled_files = random.sample(all_files, n)

    # Return full paths
    return [os.path.join(dir_path, f) for f in sampled_files]


def sample_random_dirs(parent_dir, n=5):
    # Get all entries that are directories
    all_dirs = [
        d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))
    ]

    # Sample without replacement (will raise if n > len(all_dirs))
    sampled_dirs = random.sample(all_dirs, n)

    # Return full paths (optional)
    return [os.path.join(parent_dir, d) for d in sampled_dirs]


def extract_app_from_dirname(full_path):
    return os.path.splitext(os.path.basename(full_path))[0]

"""
Automated pipeline: wait for data download → extract → train.

Usage: python run_pipeline.py
"""
import os
import sys
import time
import tarfile
import subprocess

DATA_DIR = "D:/emg/emg_nature/emg_data"
TAR_FILE = os.path.join(DATA_DIR, "discrete_gestures_full_data.tar")
PROJECT_DIR = "D:/emg/emg_transfer"
CONDA_ENV = "neuromotor"

# Use the emg_nature download script to handle tar extraction
# (it also downloads the CSV if needed)
EXTRACT_SCRIPT = (
    "D:/emg/emg_nature/generic-neuromotor-interface-main/"
    "generic-neuromotor-interface-main/"
    "generic_neuromotor_interface/scripts/download_data.py"
)


def wait_for_download(tar_path, expected_size_gb=31.0, check_interval=120):
    last_size = os.path.getsize(tar_path) if os.path.exists(tar_path) else 0
    print(f"[{time.strftime('%H:%M:%S')}] Waiting for download: {tar_path}")
    print(f"  Initial size: {last_size / 1e9:.1f} GB / {expected_size_gb:.0f} GB")

    while True:
        if os.path.exists(tar_path):
            current_size = os.path.getsize(tar_path)
            progress = current_size / (expected_size_gb * 1e9) * 100
            speed = (current_size - last_size) / check_interval / 1e6
            eta_min = (expected_size_gb * 1e9 - current_size) / max(speed * 1e6, 1e-9) / 60

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"{current_size / 1e9:.1f}/{expected_size_gb:.0f} GB "
                  f"({progress:.0f}%)  "
                  f"speed: {speed:.1f} MB/s  "
                  f"ETA: {eta_min:.0f} min")

            last_size = current_size

            # Check if download might be complete (size stable for two checks)
            time.sleep(check_interval)
            if os.path.exists(tar_path):
                new_size = os.path.getsize(tar_path)
                if new_size == current_size and new_size > expected_size_gb * 0.95 * 1e9:
                    print(f"[{time.strftime('%H:%M:%S')}] Download appears complete!")
                    return True
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Tar file not found yet, waiting...")
            time.sleep(check_interval)


def extract_tar(tar_path, dest_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Extracting {tar_path} to {dest_dir}...")
    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        print(f"  {len(members)} files to extract")
        tar.extractall(path=dest_dir)
    print(f"[{time.strftime('%H:%M:%S')}] Extraction complete!")

    # Count extracted HDF5 files
    hdf5_files = []
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if f.startswith("discrete_gestures_") and f.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, f))
    print(f"  Found {len(hdf5_files)} gesture HDF5 files")
    return len(hdf5_files)


def run_training():
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting Phase 2 training...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "emg_transfer.train",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_file = os.path.join(PROJECT_DIR, "training_output.log")
    with open(log_file, "w") as f:
        for line in process.stdout:
            f.write(line)
            f.flush()
            # Also print key lines
            if any(kw in line.lower() for kw in
                   ["epoch", "loss", "accuracy", "cler", "error", "completed",
                    "best", "saved", "freeze", "unfrozen"]):
                print(f"  [{time.strftime('%H:%M:%S')}] {line.rstrip()}")

    process.wait()
    print(f"\n[{time.strftime('%H:%M:%S')}] Training completed with exit code {process.returncode}")
    print(f"Full output: {log_file}")

    # Show final results
    if os.path.exists(log_file):
        print("\n--- Last 30 lines of training log ---")
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-30:]:
                print(line.rstrip())


def main():
    print(f"[{time.strftime('%H:%M:%S')}] === EMG Transfer Pipeline ===")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Conda env: {CONDA_ENV}")

    # Step 1: Wait for download
    wait_for_download(TAR_FILE, expected_size_gb=31.0, check_interval=120)

    # Step 2: Extract
    n_files = extract_tar(TAR_FILE, DATA_DIR)
    if n_files == 0:
        print("ERROR: No gesture files extracted!")
        sys.exit(1)

    # Step 3: Verify corpus CSV exists
    corpus_csv = os.path.join(DATA_DIR, "discrete_gestures_corpus.csv")
    if not os.path.exists(corpus_csv):
        print(f"WARNING: corpus CSV not found at {corpus_csv}")
        print("Attempting to download it...")
        import urllib.request
        url = ("https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data/"
               "data/discrete_gestures/discrete_gestures_corpus.csv")
        urllib.request.urlretrieve(url, corpus_csv)
        print(f"  Downloaded to {corpus_csv}")

    # Step 4: Run training
    run_training()

    print(f"\n[{time.strftime('%H:%M:%S')}] Pipeline finished!")


if __name__ == "__main__":
    main()

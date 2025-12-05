import argparse
import os
import shutil
import tonic
from utils import create_symlink


def main():
    default_datasets_path = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(default_datasets_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Setup and organize datasets")
    parser.add_argument("--force", action="store_true",
                        help="Force reinstallation of datasets "
                        "(delete existing installation)"
                        )
    parser.add_argument("--dir", default=default_datasets_path,
                        help="Base directory to store dataset files"
                        )
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    if not args.force and check(args.dir):
        print("NMNIST dataset already installed.")
        return

    print("Setting up NMNIST dataset...")
    organize(args.dir, default_datasets_path)

    print("Verifying NMNIST dataset installation...")
    if check(default_datasets_path):
        print("NMNIST dataset successfully installed.")
    else:
        print("Error installing NMNIST dataset.")


def organize(dataset_path: str, default_path: str) -> None:
    nmnist_path = os.path.join(dataset_path, 'NMNIST')

    if os.path.exists(nmnist_path):
        print(f"Removing existing files at {nmnist_path}...")
        shutil.rmtree(nmnist_path)

    print("Downloading NMNIST dataset...")
    tonic.datasets.NMNIST(save_to=dataset_path, train=True)
    tonic.datasets.NMNIST(save_to=dataset_path, train=False)

    print("Removing temporary files...")
    os.remove(os.path.join(nmnist_path, 'train.zip'))
    os.remove(os.path.join(nmnist_path, 'test.zip'))

    create_symlink(nmnist_path, os.path.join(default_path, 'NMNIST'))


def check(datasets_path: str) -> bool:
    expected_counts = {
        "Train": 60000,
        "Test": 10000
    }

    for split, expected_count in expected_counts.items():
        split_path = os.path.join(datasets_path, 'NMNIST', split)

        if not os.path.isdir(split_path):
            return False

        for class_idx in range(10):
            class_folder = os.path.join(split_path, str(class_idx))
            if not os.path.isdir(class_folder):
                return False

        total_files = sum(
            len([f for f in files if f.endswith(".bin")])
            for _, _, files in os.walk(split_path)
        )

        if total_files != expected_count:
            return False

    return True


if __name__ == "__main__":
    main()

import os
import shutil
from demo.utils.data import download_file, extract_file, create_symlink


def organize(datasets_path: str, default_path: str, zip_path: str = None) -> None:
    url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/468j46mzdv-1.zip"
    nmnist_path = os.path.join(datasets_path, 'N-MNIST')

    if os.path.exists(nmnist_path):
        print(f"Removing existing files at {nmnist_path}...")
        shutil.rmtree(nmnist_path)

    download = not zip_path
    if download:
        os.makedirs(datasets_path, exist_ok=True)
        zip_path = os.path.join(datasets_path, 'N-MNIST.zip')
        print("Downloading N-MNIST dataset files...")
        download_file(url, zip_path)

    print("Unzipping archive files...")
    extract_file(zip_path, datasets_path, 'N-MNIST')

    if download:
        os.remove(zip_path)

    train_zip_path = os.path.join(nmnist_path, 'Train.zip')
    print("Unzipping Train.zip...")
    extract_file(train_zip_path, nmnist_path)
    os.remove(train_zip_path)

    test_zip_path = os.path.join(nmnist_path, 'Test.zip')
    print("Unzipping Test.zip...")
    extract_file(test_zip_path, nmnist_path)
    os.remove(test_zip_path)

    create_symlink(nmnist_path, os.path.join(default_path, 'N-MNIST'))


def check(datasets_path: str) -> bool:
    expected_counts = {
        "Train": 60000,
        "Test": 10000
    }

    for split, expected_count in expected_counts.items():
        split_path = os.path.join(datasets_path, 'N-MNIST', split)

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

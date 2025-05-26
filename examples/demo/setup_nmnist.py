import argparse
import os
from demo.utils import nmnist


def main():
    default_datasets_path = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(default_datasets_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Setup and organize datasets")
    parser.add_argument("--force", action="store_true",
                        help="Force reinstallation of datasets (delete existing installation)")
    parser.add_argument("--dir", default=default_datasets_path,
                        help="Base directory to store dataset files")
    parser.add_argument("--archive", required=False,
                        help="The path to the zip file containing the dataset files.")
    args = parser.parse_args()

    if not args.force and nmnist.check(default_datasets_path):
        print("N-MNIST dataset already installed.")
        return

    print("Setting up N-MNIST dataset...")
    nmnist.organize(args.dir, default_datasets_path, args.archive)

    print("Verifying N-MNIST dataset installation...")
    if nmnist.check(default_datasets_path):
        print("N-MNIST dataset successfully installed.")
    else:
        print("Error installing N-MNIST dataset.")


if __name__ == "__main__":
    main()

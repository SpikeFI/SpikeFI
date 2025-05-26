import argparse
import os
from torchvision.datasets.utils import check_integrity
from demo.utils import gesture


def main():
    default_datasets_path = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(default_datasets_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Setup and organize datasets")
    parser.add_argument("--force", action="store_true",
                        help="Force reinstallation of datasets (delete existing installation)")
    parser.add_argument("--dir", default=default_datasets_path,
                        help="Base directory to store dataset files")
    parser.add_argument("--archive", required=True,
                        help="The path to the zip file containing the dataset files.\n"
                             "Please download DVS Gesture dataset from "
                             "https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942")
    args = parser.parse_args()

    if not args.force and gesture.check(default_datasets_path):
        print("DVS Gesture dataset is already installed.")
        return

    print("Checking integrity of archive file...")
    if not check_integrity(fpath=args.archive, md5="8a5c71fb11e24e5ca5b11866ca6c00a1"):
        print(f'The file [{args.archive}] does not exist or is corrupted.')

    print("Setting up DVS Gesture dataset...")
    gesture.organize(args.dir, default_datasets_path, args.archive)

    print("Verifying DVS Gesture dataset installation...")
    if gesture.check(default_datasets_path):
        print("DVS Gesture dataset successfully installed.")
    else:
        print("Error installing DVS Gesture dataset.")


if __name__ == "__main__":
    main()

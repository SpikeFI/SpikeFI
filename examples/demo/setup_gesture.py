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

    if not args.force and check(args.dir):
        print("DVS Gesture dataset is already installed.")
        return

    print("Setting up DVS Gesture dataset...")
    organize(args.dir, default_datasets_path)

    print("Verifying DVS Gesture dataset installation...")
    if check(default_datasets_path):
        print("DVS Gesture dataset successfully installed.")
    else:
        print("Error installing DVS Gesture dataset.")


def check(datasets_path: str) -> bool:
    expected_counts = {
        "Train": 1077,
        "Test": 264
    }

    for split, expected_count in expected_counts.items():
        split_path = os.path.join(datasets_path, 'DVSGesture', split)

        if not os.path.isdir(split_path):
            return False

        for class_idx in range(11):
            class_folder = os.path.join(split_path, str(class_idx))
            if not os.path.isdir(class_folder):
                return False

        total_files = sum(
            len([f for f in files if f.endswith(".npy")])
            for _, _, files in os.walk(split_path)
        )

        if total_files != expected_count:
            return False

    return True


def organize(dataset_path: str, default_path: str) -> None:
    gesture_path = os.path.join(dataset_path, 'DVSGesture')

    if os.path.exists(gesture_path):
        print(f"Removing existing files at {gesture_path}...")
        shutil.rmtree(gesture_path)

    print("Downloading DVS Gesture dataset...")
    tonic.datasets.DVSGesture(save_to=dataset_path, train=True)
    tonic.datasets.DVSGesture(save_to=dataset_path, train=False)

    print("Organizing dataset structure...")

    # Make output directories
    for a in range(11):
        os.makedirs(os.path.join(gesture_path, 'Train', str(a)), exist_ok=True)
        os.makedirs(os.path.join(gesture_path, 'Test', str(a)), exist_ok=True)

    # Rearrange dateset files
    for id in range(1, 30):
        for light in [
            'fluorescent', 'fluorescent_led', 'lab', 'led', 'natural'
        ]:
            par_dir1 = 'ibmGestureTrain' if id < 24 else 'ibmGestureTest'
            par_dir2 = 'Train' if id < 24 else 'Test'
            sample = f'user{id:02d}_{light}'

            old_sample_dir = os.path.join(gesture_path, par_dir1, sample)
            new_sample_dir = os.path.join(gesture_path, par_dir2)
            if os.path.exists(old_sample_dir):
                for a in range(11):
                    old_fpath = os.path.join(old_sample_dir, f'{a}.npy')
                    new_fpath = os.path.join(
                        new_sample_dir, str(a), f'{sample}_{a}.npy'
                    )

                    if os.path.isfile(old_fpath):
                        os.rename(old_fpath, new_fpath)

    print("Removing temporary files...")
    os.remove(os.path.join(gesture_path, 'ibmGestureTrain.tar.gz'))
    os.remove(os.path.join(gesture_path, 'ibmGestureTest.tar.gz'))
    shutil.rmtree(os.path.join(gesture_path, 'ibmGestureTrain'))
    shutil.rmtree(os.path.join(gesture_path, 'ibmGestureTest'))

    create_symlink(gesture_path, os.path.join(default_path, 'DVSGesture'))


if __name__ == "__main__":
    main()

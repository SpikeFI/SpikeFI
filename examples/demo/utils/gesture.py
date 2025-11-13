import csv
from dv import LegacyAedatFile
import numpy as np
import os
import shutil
import slayerSNN as snn
from demo.utils.data import extract_file, create_symlink


def get_action_names(gesture_path: str) -> list[str]:
    action_names = []
    with open(
        os.path.join(gesture_path, 'gesture_mapping.csv'), newline=''
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            action = row[0].strip()
            action_names.append(action)

    return action_names


def read_aedat_event(fname: str) -> tuple[list[float]]:
    with LegacyAedatFile(fname) as f:
        events = list(f)

    xEvent = [e.x for e in events]
    yEvent = [e.y for e in events]
    pEvent = [e.polarity for e in events]
    tEvent = [(e.timestamp / 1000) for e in events]

    return xEvent, yEvent, pEvent, tEvent


def split_data(
        id: int,
        sample: str,
        action_names: list[str],
        gesture_path: str,
        aedat_path: str
) -> int:
    # Read raw DVS events
    x, y, p, t = read_aedat_event(os.path.join(aedat_path, sample + '.aedat'))
    x, y, p, t = map(np.array, (x, y, p, t))

    # Load label file: columns = [action, start_time, end_time]
    labels = np.loadtxt(
        os.path.join(aedat_path, sample + '_labels.csv'),
        delimiter=',',
        skiprows=1
    )
    labels[:, 0] -= 1  # Adjust action index to start at 0

    split = 'Train' if id <= 23 else 'Test'

    count = 0
    last_action = -1
    for action, t_start, t_end in labels:
        if action == last_action:
            continue  # Skip duplicate arm_roll samples

        count += 1
        last_action = action

        # Select indices for events in the label time window
        mask = (t >= t_start / 1000) & (t < t_end / 1000)
        # Normalize timestamps to start from 0
        t_segment = t[mask] - t_start / 1000
        TD = snn.io.event(x[mask], y[mask], p[mask], t_segment)

        # Save encoded spike tensor
        a = int(action)
        out_fpath = os.path.join(
            gesture_path, split, str(a), f'{sample}_{action_names[a]}_{a}.npy'
        )
        snn.io.encodeNpSpikes(out_fpath, TD)

    return count


def convert(gesture_path: str, aedat_dir: str = 'Aedat') -> None:
    aedat_path = os.path.join(gesture_path, aedat_dir)
    user_ids = np.arange(29) + 1
    lighting = ['fluorescent', 'fluorescent_led', 'lab', 'led', 'natural']
    action_names = get_action_names(gesture_path)

    # Make output directories if not exist
    for a in range(11):
        os.makedirs(os.path.join(gesture_path, 'Train', str(a)), exist_ok=True)
        os.makedirs(os.path.join(gesture_path, 'Test', str(a)), exist_ok=True)

    total_samples = 0
    for id in user_ids:
        for light in lighting:
            sample = f'user{id:02d}_{light}'

            if os.path.isfile(os.path.join(aedat_path, sample + '.aedat')):
                print(f"Converting samples for user {id:02d} "
                      f"under '{light.replace('_', ' ')}' light")
                total_samples += split_data(
                    id, sample, action_names, gesture_path, aedat_path
                )

    print(f"{total_samples} samples were created in total.")


def check(datasets_path: str) -> bool:
    expected_counts = {
        "Train": 1077,
        "Test": 264
    }

    for split, expected_count in expected_counts.items():
        split_path = os.path.join(datasets_path, 'DVS Gesture', split)

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


def organize(datasets_path: str, default_path: str, zip_path: str) -> None:
    gesture_path = os.path.join(datasets_path, 'DVS Gesture')

    if os.path.exists(gesture_path):
        print(f"Removing existing files at {gesture_path}...")
        shutil.rmtree(gesture_path)

    print("Unzipping archive files...")
    extract_file(zip_path, gesture_path, 'Aedat')

    aedat_path = os.path.join(gesture_path, 'Aedat')

    base_files = [
        'README.txt', 'LICENSE.txt', 'gesture_mapping.csv',
        'trials_to_test.txt', 'trials_to_train.txt', 'errata.txt'
    ]
    for fname in base_files:
        shutil.move(
            os.path.join(aedat_path, fname),
            os.path.join(gesture_path, fname)
        )

    print("Converting dataset to numpy format...")
    convert(gesture_path)

    create_symlink(gesture_path, os.path.join(default_path, 'DVS Gesture'))

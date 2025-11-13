import os
import requests
import shutil
import tarfile
from tqdm import tqdm
import zipfile


def download_file(url: str | bytes, dest_path: str) -> None:
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)


def _extract(archive_path: str, extract_to: str = None) -> None:
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif (
        archive_path.endswith('.tar.gz')
        or archive_path.endswith('.tgz')
        or archive_path.endswith('.tar')
    ):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def extract_file(
        archive_path: str,
        extract_to: str = None,
        top_folder_name: str = None
) -> None:
    if not top_folder_name:
        _extract(archive_path, extract_to)
        return

    temp_path = os.path.join(extract_to, 'temp')
    os.makedirs(temp_path, exist_ok=True)
    _extract(archive_path, temp_path)

    extracted_items = os.listdir(temp_path)
    top_folders = [
        item for item in extracted_items
        if os.path.isdir(os.path.join(temp_path, item))
    ]

    if len(top_folders) != 1:
        raise ValueError("Expected exactly one top folder after unzipping, "
                         f"but found: {top_folders}")
    top_folder = top_folders[0]

    if top_folder != top_folder_name:
        os.rename(
            os.path.join(temp_path, top_folder),
            os.path.join(temp_path, top_folder_name)
        )

    shutil.move(os.path.join(temp_path, top_folder_name), extract_to)
    shutil.rmtree(temp_path)


def create_symlink(target_path: str, link_path: str) -> None:
    # Resolve absolute paths
    target_abs = os.path.abspath(target_path)
    link_abs = os.path.abspath(link_path)

    # Check if they are the same
    if target_abs == link_abs:
        return

    # Check if link path already exists
    if os.path.exists(link_abs):
        if os.path.islink(link_abs):
            print(f"Removing existing symlink at {link_abs}...")
            os.unlink(link_abs)
        elif os.path.isdir(link_abs):
            print(f"Removing existing files at {link_abs}...")
            shutil.rmtree(link_abs)
        elif os.path.isfile(link_abs):
            print(f"Removing file {link_abs}...")
            os.remove(link_abs)
        else:
            raise FileExistsError(f"Default path '{link_abs}' already exists. "
                                  "Please delete it and try again.")

    # Create the symbolic link
    print(f"Creating symbolic link at {link_abs} → {target_abs}...")
    os.symlink(target_abs, link_abs)

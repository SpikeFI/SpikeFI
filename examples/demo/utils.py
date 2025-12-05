import os
import shutil


def create_symlink(src: str, dst: str) -> None:
    # Resolve absolute paths
    dst_abs = os.path.abspath(src)
    src_abs = os.path.abspath(dst)

    # Check if they are the same
    if dst_abs == src_abs:
        return

    if os.path.islink(src_abs) or os.path.exists(src_abs):
        print(f"Removing existing link or directory at {src_abs}...")
        if os.path.isdir(src_abs) and not os.path.islink(src_abs):
            shutil.rmtree(src_abs)
        else:
            os.unlink(src_abs)

    # Create the symbolic link
    print(f"Creating symlink {src_abs} → {dst_abs}")
    os.symlink(dst_abs, src_abs)

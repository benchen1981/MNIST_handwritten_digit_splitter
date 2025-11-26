import os
import shutil

from .file_utils import ensure_folder

def clear_output(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    ensure_folder(path)

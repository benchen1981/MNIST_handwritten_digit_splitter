import os

def ensure_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

import os
from collections import Counter

def compute_label_stats(output_dir: str):
    """
    掃描 output_dir，統計各子資料夾（視為標籤）的檔案數量。
    回傳 dict，例如 {"0": 120, "1": 98, "unknown": 3}
    """
    if not os.path.exists(output_dir):
        return {}

    counter = Counter()

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            rel_dir = os.path.relpath(root, output_dir)
            label = rel_dir.split(os.sep)[0] if rel_dir != "." else "unknown"
            counter[label] += 1

    return dict(counter)

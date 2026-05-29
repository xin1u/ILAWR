import os
import re
import glob


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_latest_checkpoint(save_dir, pattern='*epoch*.pth'):
    files = glob.glob(os.path.join(save_dir, pattern))
    if not files:
        return 0
    epochs = []
    for f in files:
        match = re.findall(r'epoch(\d+)', f)
        if match:
            epochs.append(int(match[0]))
    return max(epochs) if epochs else 0

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF


def preprocess_pair(degraded, clean, crop_size, augment=False):
    if not isinstance(crop_size, str):
        i, j, h, w = tfs.RandomCrop.get_params(
            degraded, output_size=(crop_size, crop_size))
        degraded = TF.crop(degraded, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)
    if augment:
        if random.random() > 0.5:
            degraded = TF.hflip(degraded)
            clean = TF.hflip(clean)
        if random.random() > 0.5:
            degraded = TF.vflip(degraded)
            clean = TF.vflip(clean)
    degraded = TF.to_tensor(degraded.convert("RGB"))
    clean = TF.to_tensor(clean.convert("RGB"))
    return degraded, clean


class HazeDataset(data.Dataset):
    """RESIDE outdoor haze dataset (SOTS format)."""
    def __init__(self, root, crop_size=240, ext='.png', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.ext = ext
        self.augment = augment
        hazy_dir = os.path.join(root, 'hazy')
        names = os.listdir(hazy_dir)
        self.hazy_paths = [os.path.join(hazy_dir, n) for n in names]
        self.gt_dir = os.path.join(root, 'gt')

    def __getitem__(self, idx):
        hazy = Image.open(self.hazy_paths[idx])
        if isinstance(self.crop_size, int):
            while hazy.size[0] < self.crop_size or hazy.size[1] < self.crop_size:
                idx = random.randint(0, len(self.hazy_paths) - 1)
                hazy = Image.open(self.hazy_paths[idx])
        fname = os.path.basename(self.hazy_paths[idx])
        gt_id = fname.split('.')[0].split('_')[0]
        gt_name = gt_id + self.ext
        clean = Image.open(os.path.join(self.gt_dir, gt_name))
        clean = tfs.CenterCrop(hazy.size[::-1])(clean)
        return preprocess_pair(hazy, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.hazy_paths)


class RainDataset(data.Dataset):
    """Rain100H dataset (input/target format)."""
    def __init__(self, root, crop_size=240, augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.augment = augment
        inp_dir = os.path.join(root, 'input')
        names = os.listdir(inp_dir)
        self.inp_paths = [os.path.join(inp_dir, n) for n in names]
        self.gt_dir = os.path.join(root, 'target')

    def __getitem__(self, idx):
        rainy = Image.open(self.inp_paths[idx])
        if isinstance(self.crop_size, int):
            while rainy.size[0] < self.crop_size or rainy.size[1] < self.crop_size:
                idx = random.randint(0, len(self.inp_paths) - 1)
                rainy = Image.open(self.inp_paths[idx])
        fname = os.path.basename(self.inp_paths[idx])
        clean = Image.open(os.path.join(self.gt_dir, fname))
        clean = tfs.CenterCrop(rainy.size[::-1])(clean)
        return preprocess_pair(rainy, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.inp_paths)


class SnowDataset(data.Dataset):
    """Snow100K dataset (synthetic/gt format)."""
    def __init__(self, root, crop_size=240, ext='.jpg', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.augment = augment
        syn_dir = os.path.join(root, 'synthetic')
        names = os.listdir(syn_dir)
        self.snow_paths = [os.path.join(syn_dir, n) for n in names]
        self.gt_dir = os.path.join(root, 'gt')

    def __getitem__(self, idx):
        snowy = Image.open(self.snow_paths[idx])
        if isinstance(self.crop_size, int):
            while snowy.size[0] < self.crop_size or snowy.size[1] < self.crop_size:
                idx = random.randint(0, len(self.snow_paths) - 1)
                snowy = Image.open(self.snow_paths[idx])
        fname = os.path.basename(self.snow_paths[idx])
        clean = Image.open(os.path.join(self.gt_dir, fname))
        clean = tfs.CenterCrop(snowy.size[::-1])(clean)
        return preprocess_pair(snowy, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.snow_paths)


class RaindropDataset(data.Dataset):
    """Raindrop dataset (data/gt format)."""
    def __init__(self, root, crop_size=240, ext='.png', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.ext = ext
        self.augment = augment
        data_dir = os.path.join(root, 'data')
        names = os.listdir(data_dir)
        self.img_paths = [os.path.join(data_dir, n) for n in names]
        self.gt_dir = os.path.join(root, 'gt')

    def __getitem__(self, idx):
        drop = Image.open(self.img_paths[idx])
        if isinstance(self.crop_size, int):
            while drop.size[0] < self.crop_size or drop.size[1] < self.crop_size:
                idx = random.randint(0, len(self.img_paths) - 1)
                drop = Image.open(self.img_paths[idx])
        fname = os.path.basename(self.img_paths[idx])
        gt_id = fname.split('_')[0]
        gt_name = gt_id + '_clean' + self.ext
        clean = Image.open(os.path.join(self.gt_dir, gt_name))
        clean = tfs.CenterCrop(drop.size[::-1])(clean)
        return preprocess_pair(drop, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.img_paths)


class LowLightDataset(data.Dataset):
    """LOL-v2 Real_captured dataset (Low/Normal format)."""
    def __init__(self, root, crop_size=240, ext='.png', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.ext = ext
        self.augment = augment
        low_dir = os.path.join(root, 'Low')
        names = os.listdir(low_dir)
        self.low_paths = [os.path.join(low_dir, n) for n in names]
        self.normal_dir = os.path.join(root, 'Normal')

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx])
        if isinstance(self.crop_size, int):
            while low.size[0] < self.crop_size or low.size[1] < self.crop_size:
                idx = random.randint(0, len(self.low_paths) - 1)
                low = Image.open(self.low_paths[idx])
        fname = os.path.basename(self.low_paths[idx])
        base = fname.split('.')[0]
        gt_name = 'normal' + base[3:] + self.ext
        normal = Image.open(os.path.join(self.normal_dir, gt_name))
        normal = tfs.CenterCrop(low.size[::-1])(normal)
        return preprocess_pair(low, normal, self.crop_size, self.augment)

    def __len__(self):
        return len(self.low_paths)


class SPADataset(data.Dataset):
    """SPA+ real-world rain dataset (input/target format with gt suffix)."""
    def __init__(self, root, crop_size=240, ext='.png', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.ext = ext
        self.augment = augment
        inp_dir = os.path.join(root, 'input')
        names = os.listdir(inp_dir)
        self.inp_paths = [os.path.join(inp_dir, n) for n in names]
        self.gt_dir = os.path.join(root, 'target')

    def __getitem__(self, idx):
        rainy = Image.open(self.inp_paths[idx])
        if isinstance(self.crop_size, int):
            while rainy.size[0] < self.crop_size or rainy.size[1] < self.crop_size:
                idx = random.randint(0, len(self.inp_paths) - 1)
                rainy = Image.open(self.inp_paths[idx])
        fname = os.path.basename(self.inp_paths[idx])
        base = fname.split('.')[0]
        gt_name = base + 'gt' + self.ext
        clean = Image.open(os.path.join(self.gt_dir, gt_name))
        clean = tfs.CenterCrop(rainy.size[::-1])(clean)
        return preprocess_pair(rainy, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.inp_paths)


class REVIDEDataset(data.Dataset):
    """REVIDE indoor haze dataset (hazy/gt with sub-folders)."""
    def __init__(self, root, crop_size=240, ext='.JPG', augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.ext = ext
        self.augment = augment
        self.img_paths = []
        hazy_root = os.path.join(root, 'hazy')
        for folder in os.listdir(hazy_root):
            folder_path = os.path.join(hazy_root, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    self.img_paths.append(os.path.join(folder_path, f))
        self.gt_root = os.path.join(root, 'gt')

    def __getitem__(self, idx):
        hazy = Image.open(self.img_paths[idx])
        if isinstance(self.crop_size, int):
            while hazy.size[0] < self.crop_size or hazy.size[1] < self.crop_size:
                idx = random.randint(0, len(self.img_paths) - 1)
                hazy = Image.open(self.img_paths[idx])
        full_path = self.img_paths[idx]
        fname = os.path.basename(full_path)
        folder = os.path.basename(os.path.dirname(full_path))
        gt_path = os.path.join(self.gt_root, folder, fname)
        clean = Image.open(gt_path)
        clean = tfs.CenterCrop(hazy.size[::-1])(clean)
        return preprocess_pair(hazy, clean, self.crop_size, self.augment)

    def __len__(self):
        return len(self.img_paths)


class PairedDataset(data.Dataset):
    """Generic dataset for paired LQ/GT directories with matching filenames."""
    def __init__(self, root, crop_size=240, augment=False):
        super().__init__()
        self.crop_size = crop_size
        self.augment = augment
        lq_dir = os.path.join(root, 'LQ')
        self.gt_dir = os.path.join(root, 'GT')
        names = sorted(os.listdir(lq_dir))
        self.lq_paths = [os.path.join(lq_dir, n) for n in names]

    def __getitem__(self, idx):
        lq = Image.open(self.lq_paths[idx])
        if isinstance(self.crop_size, int):
            while lq.size[0] < self.crop_size or lq.size[1] < self.crop_size:
                idx = random.randint(0, len(self.lq_paths) - 1)
                lq = Image.open(self.lq_paths[idx])
        fname = os.path.basename(self.lq_paths[idx])
        gt = Image.open(os.path.join(self.gt_dir, fname))
        gt = tfs.CenterCrop(lq.size[::-1])(gt)
        return preprocess_pair(lq, gt, self.crop_size, self.augment)

    def __len__(self):
        return len(self.lq_paths)


SYNTHETIC_TASKS = {
    'haze': lambda p, s, a: HazeDataset(
        os.path.join(p, 'haze/RESIDE/SOTS/outdoor'), crop_size=s, ext='.png', augment=a),
    'rain': lambda p, s, a: RainDataset(
        os.path.join(p, 'rain/test/Rain100H'), crop_size=s, augment=a),
    'snow': lambda p, s, a: SnowDataset(
        os.path.join(p, 'snow/Snow100K/Snow100K-testset/Snow100K-M'), crop_size=s, augment=a),
    'raindrop': lambda p, s, a: RaindropDataset(
        os.path.join(p, 'rain/raindrop_data/test_a/test_a'), crop_size=s, augment=a),
}

REALWORLD_TASKS = {
    'Real_haze': lambda p, s, a: REVIDEDataset(
        os.path.join(p, 'haze/REVIDE_inside/Test'), crop_size=s, ext='.JPG', augment=a),
    'Real_rain': lambda p, s, a: SPADataset(
        os.path.join(p, 'rain/SPA+/Testing/real_test_1000'), crop_size=s, augment=a),
    'Real_snow': lambda p, s, a: SnowDataset(
        os.path.join(p, 'snow/RealSnow/testing'), crop_size=s, augment=a),
    'Real_lol': lambda p, s, a: LowLightDataset(
        os.path.join(p, 'Low-Light/LOL-v2/Real_captured/Test'), crop_size=s, augment=a),
}

PAIRED_TASKS = {
    'hazy': lambda p, s, a: PairedDataset(os.path.join(p, 'hazy'), crop_size=s, augment=a),
    'rainy': lambda p, s, a: PairedDataset(os.path.join(p, 'rainy'), crop_size=s, augment=a),
    'snowy': lambda p, s, a: PairedDataset(os.path.join(p, 'snowy'), crop_size=s, augment=a),
    'raindrop': lambda p, s, a: PairedDataset(os.path.join(p, 'raindrop'), crop_size=s, augment=a),
}


def build_dataloaders(args, split='train'):
    data_root = args.data_path
    is_train = (split == 'train')
    crop = args.crop_size if is_train else 'whole img'
    bs = args.bs if is_train else 1

    first = args.task_order[0]
    if first.startswith('Real'):
        task_map = REALWORLD_TASKS
    elif first in PAIRED_TASKS:
        task_map = PAIRED_TASKS
    else:
        task_map = SYNTHETIC_TASKS

    loaders = []
    for task_name in args.task_order:
        ds = task_map[task_name](data_root, crop, is_train)
        loaders.append(DataLoader(
            dataset=ds, batch_size=bs, shuffle=is_train,
            num_workers=4 if is_train else 0, pin_memory=True))
    return loaders

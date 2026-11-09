import random
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


class TID2013DatasetLoader(data.Dataset):
    """Return image tensors, MOS, individual scores and image names.

    The workbook has image names in row 1, observer ratings in rows 2-20,
    and MOS in row 22. Columns after the first correspond to images.
    Scores are returned in the workbook's units.
    """

    def __init__(self, root, transform_prob=0., img_size=(224, 224),
                 train_phase=True, is_sample=True, seed=42, global_permute=True,
                 gt_file='CUID_Ind_score.xlsx', train_ratio=0.9,
                 num_observers=19, image_subdir='stimuli', image_suffix='.bmp'):
        if not global_permute:
            raise ValueError('Only the original image-level split is supported for CUID')
        if not 0 < train_ratio < 1 or num_observers < 1:
            raise ValueError('Invalid train_ratio or num_observers')
        self.root = Path(root).resolve()
        self.img_size = tuple(img_size)
        self.is_sample = is_sample
        self.image_subdir = image_subdir
        self.image_suffix = image_suffix
        self.phase = 'train' if train_phase else 'test'
        self.transform_prob = transform_prob
        book = load_workbook(gt_file, read_only=True, data_only=True)
        sheet = book.active
        rows = list(sheet.iter_rows(values_only=True))
        book.close()
        mos_row = num_observers + 2  # Zero-based index of the MOS row.
        if len(rows) <= mos_row:
            raise ValueError('Workbook must contain image names, observer rows, a separator row and MOS')
        table = []
        for column in range(1, len(rows[0])):
            name = rows[0][column]
            if name is None:
                continue
            if not isinstance(name, str) or Path(name).name != name or '\\' in name:
                raise ValueError(f'Invalid image filename in column {column + 1}')
            try:
                ratings = [float(rows[row][column]) for row in range(1, num_observers + 1)]
                mos = float(rows[mos_row][column])
            except (TypeError, ValueError, IndexError) as error:
                raise ValueError(f'Missing or invalid ratings for {name}') from error
            if not np.isfinite(ratings + [mos]).all():
                raise ValueError(f'Nonfinite ratings for {name}')
            table.append([name, *ratings, mos])
        table = np.asarray(table, dtype=object)
        train_count = int(len(table) * train_ratio)
        if not 0 < train_count < len(table):
            raise ValueError('Both training and validation splits must contain images')
        table = np.random.RandomState(seed=seed).permutation(table)
        selected = table[:train_count] if train_phase else table[train_count:]
        selected = np.random.RandomState(seed=seed).permutation(selected)
        self.files = {self.phase: selected}
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files[self.phase])

    def __getitem__(self, index):
        row = self.files[self.phase][index]
        name = row[0]
        filename = Path(name).with_suffix(self.image_suffix) if self.image_suffix else Path(name)
        path = (self.root / self.image_subdir / filename).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path must stay inside the dataset root')
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        height, width = self.img_size
        if self.is_sample:
            if image.height < height or image.width < width:
                raise ValueError(f'{path}: image is smaller than the requested crop {self.img_size}')
            top = random.randint(0, image.height - height)
            left = random.randint(0, image.width - width)
            image = image.crop((left, top, left + width, top + height))
        else:
            image = image.resize((width, height), Image.Resampling.NEAREST)
        tensor = self.to_tensor(image)
        opinions = torch.tensor(np.asarray(row[1:-1], dtype=np.float32))
        return tensor, float(row[-1]), opinions, name

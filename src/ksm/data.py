import torch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
from torch.utils.data import Subset, DataLoader
import torchvision
import os
import json
from typing import List


def get_subset_dataloader(train_ds, test_ds, subset_ratio=0.5, subset_num=50, subset_indices_path='./subset_indices', batch_size=256, num_workers=8):
    # Sample subsets from the provided datasets
    subset_size = int(len(train_ds) * subset_ratio)
    indices_folder = f'{subset_indices_path}/subset_ratio_{subset_ratio}_with_{subset_num}/'
    if os.path.exists(indices_folder):
        indices_file = os.path.join(indices_folder, 'train_data_indices.json')
        with open(indices_file, 'r') as f:
            train_ids_list = json.load(f)
    else:
        train_ids_list = []
        os.makedirs(indices_folder, exist_ok=True)
        n = len(train_ds)
        for i in range(subset_num):
            # Randomly select a subset of indices
            train_indices = torch.randperm(n)[:subset_size].tolist()
            train_ids_list.append(train_indices)
        with open(os.path.join(indices_folder, 'train_data_indices.json'), 'w') as f:
            json.dump(train_ids_list, f)

    train_subsets = [Subset(train_ds, train_indices) for train_indices in train_ids_list]
    train_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for s in train_subsets
    ]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loaders, train_ids_list, test_loader

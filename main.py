import os
from args import get_args
import pandas as pd
from datasets import KneesDataset
from torch.utils.data import DataLoader
from model import UNetLext
import torch
from trainer import train_model

# 1: we need arguments
args = get_args()

# 2: Loading the CSV files
train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

# 3: Define our datasets
train_dataset = KneesDataset(train_set)
val_dataset = KneesDataset(val_set)

# 4: Create data dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

# 5: Initialize the model
model = UNetLext(
    input_channels=1,
    output_channels=1,
    pretrained=False,
    path_weights='',
    path_pretrained='',
    restore_weights=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 6: Train
train_model(model, train_loader, val_loader, device)

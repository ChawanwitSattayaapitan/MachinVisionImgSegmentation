import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from args import get_args
from datasets import KneesDataset, read_xray
from model import UNetLext
from utils import dice_score_from_logits

# 1: we need arguments
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2: Loading the CSV files
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))
test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))

# 3: Define our dataset and dataloader
val_dataset = KneesDataset(val_set)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

# 4: Initialize the model and load the best weights
model = UNetLext(input_channels=1, output_channels=1, restore_weights=True,
                 path_weights=os.path.join(args.out_dir, 'best_model.pth'))
model = model.to(device)

# 5: Dice score on the validation set
dice_scores = []
model.eval()
with torch.no_grad():
    for data_batch in val_loader:
        images = data_batch['xray'].to(device)
        masks = data_batch['mask'].to(device)
        outputs = model(images)
        dice_scores.append(dice_score_from_logits(outputs, masks))

print("Validation Dice score:", np.mean(dice_scores))

# 6: Predict on the test set and show the result
with torch.no_grad():
    for i in range(3):
        xray = read_xray(test_set['xray'].iloc[i])
        image = torch.from_numpy(xray).unsqueeze(0).to(device)

        output = model(image)
        pred_mask = (torch.sigmoid(output) > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()

        plt.subplot(1, 3, 1)
        plt.imshow(xray[0], cmap="gray")
        plt.title("original")
        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("predicted mask")
        plt.subplot(1, 3, 3)
        plt.imshow(xray[0], cmap="gray")
        plt.imshow(pred_mask, alpha=0.3)
        plt.title("overlay")
        plt.show()
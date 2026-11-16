from args import get_args
import torch
import torch.nn as nn
from utils import dice_loss_from_logits
import os


def train_model(model, train_loader, val_loader, device):
    args = get_args()

    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)



    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss = validate_model(model, val_loader, bce, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_epoch_{epoch + 1}.pth'))

def validate_model(model, val_loader, bce, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)

    return val_loss





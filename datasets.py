from torch.utils.data import Dataset
import cv2
import numpy as np

def read_xray(dir):
    xray = cv2.imread(dir, cv2.IMREAD_GRAYSCALE) # H W
    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape(1, *xray.shape)  # 1 H W
    return xray

def read_mask(dir):
    mask = cv2.imread(dir,cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape(1, *mask.shape)
    return mask

class KneesDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray = read_xray(self.df['xray'].iloc[index]) # x value
        mask = read_mask(self.df['masks'].iloc[index]) # y value

        res = {
            'xray': xray,
            'mask': mask
        }
        return res


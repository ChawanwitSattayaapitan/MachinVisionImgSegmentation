import os
import pandas as pd
import matplotlib.pyplot as plt

data_list = []
xray_dir = "data/xray"

for xray in os.listdir(xray_dir):
    xray_path = os.path.join(xray_dir, xray)
    mask_path = xray_path.replace("xray", "masks")
    if not os.path.exists(mask_path):
        mask_path = None
    data_list.append((xray_path, mask_path))

data_df = pd.DataFrame(data_list, columns=['xray', 'masks'])
data_df.to_csv("data/CSVs/dataset.csv", index=False)

test_df = data_df[data_df['masks'].isna()]
labeled_df = data_df[data_df['masks'].notna()]


train_df = labeled_df[:80]
val_df = labeled_df[80:]

train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/val.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)

# Plot bar
print(len(data_df), len(train_df), len(val_df), len(test_df))

plt.bar(['train', 'val', 'test'], [len(train_df), len(val_df), len(test_df)])
plt.ylabel("Number of samples")
plt.title("Number of samples in each CSV")
plt.savefig("bar_chart.png")
plt.show()
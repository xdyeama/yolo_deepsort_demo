import os
import shutil
import random

# paths
root = "train"  # your current folder with train/images and train/labels
images_dir = os.path.join(root, "images")
labels_dir = os.path.join(root, "labels")

# output dirs
output = {
    "train": {"images": "dataset/train/images", "labels": "dataset/train/labels"},
    "val":   {"images": "dataset/val/images",   "labels": "dataset/val/labels"},
    "test":  {"images": "dataset/test/images",  "labels": "dataset/test/labels"},
}

# create folders
for split in output.values():
    os.makedirs(split["images"], exist_ok=True)
    os.makedirs(split["labels"], exist_ok=True)

# get all image filenames
images = [f for f in os.listdir(images_dir)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

random.shuffle(images)

n = len(images)
train_end = int(0.7 * n)
val_end   = int(0.9 * n)  # 70% + 20%

train_files = images[:train_end]
val_files   = images[train_end:val_end]
test_files  = images[val_end:]

def move_files(file_list, split_name):
    for img in file_list:
        img_src = os.path.join(images_dir, img)
        label = os.path.splitext(img)[0] + ".txt"
        label_src = os.path.join(labels_dir, label)

        img_dst = os.path.join(output[split_name]["images"], img)
        label_dst = os.path.join(output[split_name]["labels"], label)

        shutil.copy2(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Split complete:")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")


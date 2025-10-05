import os
import shutil
import random

# Source directory with original dataset images
dataset_path = "C:/Users/varsh/Downloads/archive (5)/TB_Chest_Radiography_Database"

# Output directory where organized dataset will be saved
output_base = "C:/Users/varsh/Downloads/archive (5)/new_database"

# Define classes
classes = ['Normal', 'Tuberculosis']

# Split ratios: train 70%, val 15%, test 15%
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create train, val, and test directories inside output_base
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

# Function to split and copy images
def split_and_copy(class_name):
    class_path = os.path.join(dataset_path, class_name)
    file_names = os.listdir(class_path)
    random.shuffle(file_names)
    total_files = len(file_names)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    train_files = file_names[:train_count]
    val_files = file_names[train_count:train_count + val_count]
    test_files = file_names[train_count + val_count:]

    for f in train_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_base, 'train', class_name, f)
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_base, 'val', class_name, f)
        shutil.copy(src, dst)

    for f in test_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_base, 'test', class_name, f)
        shutil.copy(src, dst)

# Process each class
for c in classes:
    split_and_copy(c)

print(f"Dataset successfully organized at {output_base} with train, val, and test splits.")

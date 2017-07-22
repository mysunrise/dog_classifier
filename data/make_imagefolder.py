import os
import sys
import shutil

current_dir = sys.path[0]
print(current_dir)
original_data_path = '/home/ubuntu/data/dog'
data_train_txt = os.path.join(original_data_path, 'data_train_image.txt')
data_val_txt = os.path.join(original_data_path, 'val.txt')
train_file = open(data_train_txt)
val_file = open(data_val_txt)
if not os.path.exists(os.path.join(current_dir, 'train')):
    os.mkdir(os.path.join(current_dir, 'train'))
if not os.path.exists(os.path.join(current_dir, 'val')):
    os.mkdir(os.path.join(current_dir, 'val'))

for line in train_file:
    name, class_id, _ = line.split(' ')
    if not os.path.exists(os.path.join(current_dir, 'train', class_id)):
        os.mkdir(os.path.join(current_dir, 'train', class_id))
    shutil.copy(os.path.join(original_data_path, 'train', name + '.jpg'), os.path.join(current_dir, 'train', class_id))

for line in val_file:
    name, class_id, _ = line.split(' ')
    if not os.path.exists(os.path.join(current_dir, 'val', class_id)):
        os.mkdir(os.path.join(current_dir, 'val', class_id))
    shutil.copy(os.path.join(original_data_path, 'test1', name + '.jpg'), os.path.join(current_dir, 'val', class_id))
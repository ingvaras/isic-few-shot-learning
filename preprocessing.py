import os
import shutil

import pandas as pd

labels_set = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")
labels = labels_set.columns.values[1:]
image_to_labels = {}
for index, row in labels_set.iterrows():
    image_name = row['image']
    label = [column for column in labels if row[column] == 1.0][0]
    image_to_labels[image_name] = label

print(image_to_labels)

for image_name, labels in image_to_labels.items():
    source_path = os.path.join('data/ISIC_2019_Training_Input', image_name + '.jpg')
    target_path = os.path.join('data/all', labels, image_name + '.jpg')
    shutil.move(source_path, target_path)
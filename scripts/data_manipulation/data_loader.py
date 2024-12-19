import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class DeepSDFDataset2D(Dataset):
    def __init__(self, root_folder, split='train', transform=None):
        assert split in ['train', 'test'], "Split must be one of 'train', 'test'."
        self.root_folder = root_folder
        self.split = split
        self.transform = transform

        self.points = []
        self.sdf_values = []
        self.shape_indices = []
        self.labels = []
        self.shape_counts = []

        label_dirs = sorted([d for d in os.listdir(root_folder) if not d.startswith('.')])
        self.label_map = {label: idx for idx, label in enumerate(label_dirs)}
        self.num_labels = len(label_dirs)

        shape_counter = 0

        for label in label_dirs:
            label_csv_dir = os.path.join(root_folder, label, 'csv_files')

            if not os.path.exists(label_csv_dir):
                print(f"CSV directory missing in {label}. Skipping...")
                continue
            csv_files = sorted([f for f in os.listdir(label_csv_dir) if f.endswith('.csv')])
            train_files, temp_files = train_test_split(csv_files, test_size=0.3, shuffle=True, random_state=42)
            if self.split == 'train':
                selected_files = train_files
            elif self.split == 'test':
                selected_files = temp_files

            for csv_file in selected_files:
                file_path = os.path.join(label_csv_dir, csv_file)
                df = pd.read_csv(file_path)
                points = df[['x', 'y']].values
                sdf = df['sdf'].values
                shape_count = df['shape_count'].values
                self.points.append(points)
                self.sdf_values.append(sdf)
                self.shape_counts.append(shape_count)

                self.shape_indices.extend([shape_counter] * len(df))
                self.labels.extend([self.label_map[label]] * len(df))

                shape_counter += 1

        self.points = [point for point in self.points if len(point) > 0]
        self.sdf_values = [sdf_value for sdf_value in self.sdf_values if len(sdf_value) > 0]
        self.shape_counts = [shape_count for shape_count in self.shape_counts]
        self.points = torch.tensor(np.concatenate(self.points, axis=0), dtype=torch.float32)
        self.sdf_values = torch.tensor(np.concatenate(self.sdf_values, axis=0), dtype=torch.float32).unsqueeze(1)
        self.shape_counts = torch.tensor(np.concatenate(self.shape_counts, axis=0)).unsqueeze(1)
        self.shape_indices = torch.tensor(self.shape_indices, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        sdf = self.sdf_values[idx]
        shape_idx = self.shape_indices[idx]
        label_idx = self.labels[idx]
        shape_count = self.shape_counts[idx]

        sample = {
            'shape_idx': shape_idx,
            'point': point,
            'sdf': sdf,
            'label_idx': label_idx,
            'shape_count': shape_count
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def dump_train_to_csv(train_loader, output_csv_path):
    all_data = []
    for batch in train_loader:
        shape_idx = batch['shape_idx'].numpy()
        points = batch['point'].numpy()
        sdf_values = batch['sdf'].numpy()
        label_idx = batch['label_idx'].numpy()

        for idx in range(len(shape_idx)):
            all_data.append({
                'shape_idx': shape_idx[idx],
                'label_idx': label_idx[idx],
                'x': points[idx][0],
                'y': points[idx][1],
                'sdf': sdf_values[idx][0]
            })

    df = pd.DataFrame(all_data)
    df_sorted = df.sort_values(by=['shape_idx'])
    df_sorted.to_csv(output_csv_path, index=False)

    num_unique_objects = df_sorted['shape_idx'].nunique()
    print(f"Total unique objects in train set: {num_unique_objects}")
    print(f"Data dumped to {output_csv_path}")




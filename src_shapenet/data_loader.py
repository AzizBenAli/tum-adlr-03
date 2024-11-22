import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DeepSDFDataset2D(Dataset):
    def __init__(self, root_folder, split='train', transform=None):
        """
        Args:
            root_folder (str): Path to the parent folder containing label subdirectories.
            split (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert split in ['train', 'test'], "Split must be one of 'train', 'test'."
        self.root_folder = root_folder
        self.split = split
        self.transform = transform

        self.points = []
        self.sdf_values = []
        self.shape_indices = []
        self.labels = []

        label_dirs = sorted([d for d in os.listdir(root_folder) if not d.startswith('.')])
        self.label_map = {label: idx for idx, label in enumerate(label_dirs)}  # Label to index mapping
        self.num_labels = len(label_dirs)

        shape_counter = 0

        for label in label_dirs:
            label_csv_dir = os.path.join(root_folder, label, 'csv_files')

            if not os.path.exists(label_csv_dir):
                print(f"CSV directory missing in {label}. Skipping...")
                continue
            csv_files = sorted([f for f in os.listdir(label_csv_dir) if f.endswith('.csv')])
            # Split csv files into train, val, test: 70%, 20%, 10%
            train_files, temp_files = train_test_split(csv_files, test_size=0.3, random_state=42)
            if self.split == 'train':
                selected_files = train_files
            elif self.split == 'test':
                selected_files = temp_files

            for csv_file in selected_files:
                file_path = os.path.join(label_csv_dir, csv_file)
                df = pd.read_csv(file_path)
                # Assuming columns: label, x, y, sdf
                points = df[['x', 'y']].values  # Shape: (N, 2)
                sdf = df['sdf'].values  # Shape: (N,)
                self.points.append(points)
                self.sdf_values.append(sdf)

                # Track the corresponding shape index and label
                self.shape_indices.extend([shape_counter] * len(df))
                self.labels.extend([self.label_map[label]] * len(df))

                shape_counter += 1

        # Concatenate all data
        self.points = [point for point in self.points if len(point) > 0]
        self.sdf_values = [sdf_value for sdf_value in self.sdf_values if len(sdf_value) > 0]
        self.points = torch.tensor(np.concatenate(self.points, axis=0), dtype=torch.float32)
        print(self.points.shape)
        self.sdf_values = torch.tensor(np.concatenate(self.sdf_values, axis=0), dtype=torch.float32).unsqueeze(1)
        self.shape_indices = torch.tensor(self.shape_indices, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        """
        Fetch a data sample at the given index.
        """
        point = self.points[idx]
        sdf = self.sdf_values[idx]
        shape_idx = self.shape_indices[idx]
        label_idx = self.labels[idx]

        sample = {
            'shape_idx': shape_idx,  # Identifier for which shape/image
            'point': point,  # (x, y) coordinates of the point
            'sdf': sdf,  # Signed distance function value at the point
            'label_idx': label_idx  # Class label index for the sample
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def dump_train_to_csv(train_loader, output_csv_path):
    """
    Dumps the training data into a CSV file, prints the total number of objects,
    and sorts the CSV by object number (shape_idx).
    """
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

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df_sorted = df.sort_values(by=['shape_idx'])
    df_sorted.to_csv(output_csv_path, index=False)

    # Print total number of unique objects
    num_unique_objects = df_sorted['shape_idx'].nunique()
    print(f"Total unique objects in train set: {num_unique_objects}")
    print(f"Data dumped to {output_csv_path}")


if __name__ == "__main__":

    data_folder = '../data_shapeNet'
    train_dataset = DeepSDFDataset2D(data_folder, split='train')
    test_dataset   = DeepSDFDataset2D(data_folder, split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=True)

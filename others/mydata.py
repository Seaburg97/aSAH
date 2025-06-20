# --coding:utf-8--
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class CT3DDataset(Dataset):
    """Dataset class for loading 3D CT images"""
    def __init__(self, folder_path, transform=None):
        self.data = []
        self.labels = []
        self.patient_ids = []  # Store patient IDs
        self.transform = transform
        
        for patient_folder in os.listdir(folder_path):
            label = int(patient_folder.split('_')[0])
            patient_path = os.path.join(folder_path, patient_folder)
            for file in os.listdir(patient_path):
                if 'initial' in file.lower() and file.endswith('.nii.gz'):
                    img_path = os.path.join(patient_path, file)
                    patient_id = file.split('_')[0]
                    self.data.append(img_path)
                    self.labels.append(label)
                    self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return image, label and patient ID for given index"""
        img_path = self.data[idx]
        img = nib.load(img_path).get_fdata()
        img = torch.from_numpy(img).float().unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        return img, label, patient_id

if __name__ == "__main__":
    # Create dataset instance
    dataset = CT3DDataset("output/train/post")

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Test data shapes
    for images, labels, patient_ids in dataloader:
        print("Images shape:", images.shape)  # (batch_size, channels, height, width, depth)
        print("Label:", labels)
        print("Patient ID:", patient_ids)
        break
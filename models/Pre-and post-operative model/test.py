# --coding:utf-8--
# CT 3D model independent prediction application code
# Only needs this file and model weight file to run prediction
import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from tqdm import tqdm
import pandas as pd
import nibabel as nib
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ===============================
# Model Architecture Definition
# ===============================

# CBAM Attention Mechanism Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                              stride=1, padding=(padding, padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_concat = torch.cat([avg_out, max_out], dim=1)
        x_conv = self.conv(x_concat)
        return self.sigmoid(x_conv) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class BottleneckWithCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride):
        super(BottleneckWithCBAM, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1),
                               stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.cbam = CBAM(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class I3Res50forStrokeOutcomeCBAM(nn.Module):
    def __init__(self, input_cha=2, block=BottleneckWithCBAM, layers=[3, 4, 6], num_classes=400):
        super(I3Res50forStrokeOutcomeCBAM, self).__init__()
        self.inplanes = 64
        self.conv1_ = nn.Conv3d(input_cha, 64, kernel_size=(7, 7, 5),
                                stride=(2, 2, 2), padding=(3, 3, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1_ = nn.Linear(512 * round(block.expansion / 2), 1)

        self.drop = nn.Dropout(0.80)
        self.drop3D = nn.Dropout3d(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i]))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)

        x = self.layer2(x)
        x = self.drop3D(x)

        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1_(x)
        return x

    def forward(self, batch):
        return self.forward_single(batch)


# ===============================
# Dataset Class
# ===============================

class CT3DDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.data = []
        self.labels = []
        self.patient_ids = []
        self.transform = transform
        
        if os.path.isdir(folder_path):
            # Simplified batch prediction: directly process all .nii.gz files in folder
            nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
            
            if not nii_files:
                print(f"Error: No .nii.gz files found in folder {folder_path}!")
                return
                
            print(f"Found {len(nii_files)} .nii.gz files")
            for file in nii_files:
                img_path = os.path.join(folder_path, file)
                # Extract patient ID from filename (remove .nii.gz suffix)
                patient_id = os.path.splitext(os.path.splitext(file)[0])[0]
                
                # Try to extract label from filename (if filename starts with a digit)
                try:
                    if file[0].isdigit():
                        label = int(file[0])
                    else:
                        label = 0  # Default label
                except:
                    label = 0  # Default label
                
                self.data.append(img_path)
                self.labels.append(label)
                self.patient_ids.append(patient_id)
        else:
            # If it's a single file
            if folder_path.endswith('.nii.gz'):
                patient_id = os.path.splitext(os.path.splitext(os.path.basename(folder_path))[0])[0]
                self.data.append(folder_path)
                self.labels.append(0)  # Default label
                self.patient_ids.append(patient_id)
            else:
                print(f"Error: {folder_path} is not a valid .nii.gz file!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        
        # Load NIfTI file using TorchIO
        try:
            subject = tio.Subject(image=tio.ScalarImage(img_path))
            if self.transform:
                subject = self.transform(subject)
            img = subject['image'].data
        except:
            # Fallback: load using nibabel
            img_data = nib.load(img_path).get_fdata()
            img = torch.from_numpy(img_data).float().unsqueeze(0)
            if self.transform:
                # Create temporary subject for transformation
                subject = tio.Subject(image=tio.ScalarImage(tensor=img))
                subject = self.transform(subject)
                img = subject['image'].data
        
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        return img, label, patient_id


# ===============================
# Custom Preprocessing
# ===============================

class RandomContrast(tio.Transform):
    def __init__(self, augmentation_factor=(0.75, 1.25), **kwargs):
        super().__init__(**kwargs)
        self.augmentation_factor = augmentation_factor

    def apply_transform(self, subject):
        for image in subject.get_images_dict().values():
            contrast_factor = np.random.uniform(self.augmentation_factor[0], self.augmentation_factor[1])
            array = image.numpy()
            mean = array.mean()
            array = (array - mean) * contrast_factor + mean
            image.set_data(torch.tensor(array))
        return subject


# ===============================
# Evaluation Metrics
# ===============================

class RegressionWithBinaryMetrics:
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.mae_loss = nn.L1Loss()

    def get_binary_predictions(self, continuous_preds):
        return (continuous_preds > self.threshold).float()

    def compute_metrics(self, true_labels, continuous_preds):
        true_labels = true_labels.squeeze()
        continuous_preds = continuous_preds.squeeze()
        
        mae = self.mae_loss(continuous_preds, true_labels).item()

        binary_labels = (true_labels > self.threshold).float()
        binary_preds = self.get_binary_predictions(continuous_preds)

        accuracy = accuracy_score(binary_labels.cpu(), binary_preds.cpu())
        precision = precision_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)
        recall = recall_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)
        f1 = f1_score(binary_labels.cpu(), binary_preds.cpu(), zero_division=0)

        try:
            auc = roc_auc_score(binary_labels.cpu(), continuous_preds.cpu())
        except:
            auc = 0.5

        tn, fp, fn, tp = confusion_matrix(binary_labels.cpu(), binary_preds.cpu()).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'mae': mae,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }


# ===============================
# Predictor Class
# ===============================

class CT3DInference:
    def __init__(self, model_path, device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Model file path (.pth)
            device: Device (None means auto selection)
        """
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data preprocessing - consistent with validation set preprocessing during training
        self.transform = tio.Compose([
            tio.Resize((320, 320, 64)),
            tio.ToCanonical(),
            tio.ZNormalization()
        ])
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"CBAM model loaded to device: {self.device}")
    
    def _load_model(self):
        """Load trained CBAM model"""
        # Initialize CBAM model architecture
        model = I3Res50forStrokeOutcomeCBAM(input_cha=1, num_classes=2)
        
        # Load model weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle DataParallel saved model
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            # If it's a DataParallel saved model, need to remove 'module.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        print(f"Successfully loaded CBAM model, training epochs: {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def preprocess_single_file(self, file_path):
        """
        Preprocess single NIfTI file
        
        Args:
            file_path: NIfTI file path
            
        Returns:
            preprocessed tensor
        """
        try:
            # Load NIfTI file
            subject = tio.Subject(image=tio.ScalarImage(file_path))
            
            # Apply preprocessing transformations
            transformed = self.transform(subject)
            
            # Extract image data and add batch dimension
            image_tensor = transformed['image'].data.unsqueeze(0)  # [1, 1, H, W, D]
            
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing file {file_path}: {str(e)}")
            return None
    
    def predict_single(self, file_path):
        """
        Predict on single file
        
        Args:
            file_path: CT image file path
            
        Returns:
            dict: Dictionary containing prediction results
        """
        # Preprocessing
        input_tensor = self.preprocess_single_file(file_path)
        if input_tensor is None:
            return None
        
        input_tensor = input_tensor.to(self.device)
        
        # Prediction
        with torch.no_grad():
            output = self.model(input_tensor).squeeze()
            # Limit output range to 0-6 (consistent with training)
            prediction = torch.clamp(output, min=0, max=6).cpu().item()
        
        # Binary classification result based on threshold (threshold used during training was 2.5)
        binary_prediction = 1 if prediction > 2.5 else 0
        
        result = {
            'file_path': file_path,
            'regression_score': round(prediction, 4),
            'binary_prediction': binary_prediction,
            'confidence': abs(prediction - 2.5) / 2.5  # Confidence calculation
        }
        
        return result
    
    def predict_batch(self, data_folder, batch_size=4):
        """
        Batch prediction for all CT files in folder
        
        Args:
            data_folder: Folder path containing CT files
            batch_size: Batch size
            
        Returns:
            list: List containing all prediction results
        """
        # Create dataset
        dataset = CT3DDataset(data_folder, transform=self.transform)
        
        if len(dataset) == 0:
            print("No valid CT files found!")
            return []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)
        
        results = []
        
        print(f"Starting batch prediction for {len(dataset)} samples...")
        
        with torch.no_grad():
            for inputs, labels, indices in tqdm(dataloader, desc="Predicting"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).squeeze()
                
                # Limit output range
                predictions = torch.clamp(outputs, min=0, max=6)
                
                # Handle single sample case
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                    indices = [indices]
                
                # Save results
                for i, (pred, idx) in enumerate(zip(predictions, indices)):
                    pred_value = pred.cpu().item()
                    binary_pred = 1 if pred_value > 2.5 else 0
                    
                    result = {
                        'patient_id': idx,
                        'regression_score': round(pred_value, 4),
                        'binary_prediction': binary_pred,
                        'confidence': abs(pred_value - 2.5) / 2.5,
                        'true_label': labels[i].item() if len(labels) > i else None
                    }
                    results.append(result)
        
        return results
    
    def save_results(self, results, output_path):
        """
        Save prediction results to CSV file
        
        Args:
            results: List of prediction results
            output_path: Output file path
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")


# ===============================
# Main Function
# ===============================

def main():
    parser = argparse.ArgumentParser(description='CT 3D CBAM Model Independent Prediction Application')
    parser.add_argument('--model_path', type=str,default="/data0/wjq2/3_Multi_An_prognosis/pengzhannet/test/Pre-operative Model.pth",
                        help='Trained CBAM model file path (.pth)')
    parser.add_argument('--input_path', type=str,default="/data0/wjq2/3_Multi_An_prognosis/pengzhannet/test",
                        help='Input path: single file path or folder path containing files')
    parser.add_argument('--output_path', type=str, default='./predictions.csv',
                        help='Output result file path')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (only for folder prediction)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Specify GPU ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Initialize predictor
    predictor = CT3DInference(model_path=args.model_path)
    
    # Determine if input is file or folder
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file prediction
        print(f"Predicting single file: {input_path}")
        result = predictor.predict_single(str(input_path))
        
        if result:
            print("\nPrediction results:")
            print(f"File: {result['file_path']}")
            print(f"Regression score: {result['regression_score']}")
            print(f"Binary prediction: {result['binary_prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Save single result
            predictor.save_results([result], args.output_path)
        else:
            print("Prediction failed!")
    
    elif input_path.is_dir():
        # Batch prediction
        print(f"Batch predicting folder: {input_path}")
        results = predictor.predict_batch(
            str(input_path),
            batch_size=args.batch_size
        )
        
        if results:
            print(f"\nBatch prediction completed, processed {len(results)} samples")
            
            # Show statistics
            positive_count = sum(1 for r in results if r['binary_prediction'] == 1)
            print(f"Positive predictions: {positive_count}")
            print(f"Negative predictions: {len(results) - positive_count}")
            
            avg_score = np.mean([r['regression_score'] for r in results])
            print(f"Average regression score: {avg_score:.4f}")
            
            # Save results
            predictor.save_results(results, args.output_path)
        else:
            print("Batch prediction failed!")
    
    else:
        print(f"Error: Input path {input_path} does not exist!")


# ===============================
# Usage Examples
# ===============================

def show_usage():
    """Show usage examples"""
    print("="*60)
    print("CT 3D CBAM Model Independent Prediction Application - Usage Examples")
    print("="*60)
    
    print("\n1. Single file prediction:")
    print("python ct3d_inference.py \\")
    print("    --model_path ./f1Best_model.pth \\")
    print("    --input_path ./patient_001.nii.gz \\")
    print("    --output_path ./single_prediction.csv")
    
    print("\n2. Batch prediction:")
    print("python ct3d_inference.py \\")
    print("    --model_path ./f1Best_model.pth \\")
    print("    --input_path ./test_data_folder/ \\")
    print("    --output_path ./batch_predictions.csv \\")
    print("    --batch_size 8")
    print("\n   Folder structure:")
    print("   test_data_folder/")
    print("   ├── patient001.nii.gz")
    print("   ├── patient002.nii.gz")
    print("   ├── case_abc.nii.gz")
    print("   ├── scan_123.nii.gz")
    print("   └── ...")
    
    print("\n3. Direct use in code:")
    print("""
from ct3d_inference import CT3DInference

# Initialize predictor
predictor = CT3DInference(model_path='./f1Best_model.pth')

# Single file prediction
result = predictor.predict_single('./test_image.nii.gz')
print(result)

# Batch prediction
results = predictor.predict_batch('./test_folder/', batch_size=4)
predictor.save_results(results, './predictions.csv')
""")
    
    print("\n4. Input data requirements:")
    print("📁 Batch prediction:")
    print("   - Place all .nii.gz files directly in one folder")
    print("   - File naming format: any_name.nii.gz")
    print("   - Automatically extract patient ID from filename")
    print("")
    print("📄 Single file prediction:")
    print("   - Directly specify .nii.gz file path")
    
    print("\n5. Output results description:")
    print("- regression_score: Regression score (0-6 range)")
    print("- binary_prediction: Binary classification result (0 or 1, based on 2.5 threshold)")
    print("- confidence: Prediction confidence (0-1 range)")
    print("- patient_id: Patient ID automatically extracted from filename")
    
    print("\n6. Notes:")
    print("✅ Designed specifically for CBAM model")
    print("✅ Supports arbitrarily named .nii.gz files")
    print("✅ Simplified folder structure, no nesting required")
    print("✅ Smart extraction of patient ID information")
    print("✅ Data preprocessing consistent with training")
    print("⚠️  Ensure input is valid NIfTI format CT images")
    
    print("\n7. Input data example:")
    print("Simple and direct - place all CT files in one folder:")
    print("my_ct_data/")
    print("├── case001.nii.gz")
    print("├── patient_002.nii.gz") 
    print("├── scan_abc.nii.gz")
    print("├── test_123.nii.gz")
    print("└── any_filename.nii.gz")
    print("\n💡 Tip: Optimized for CBAM model, filename can be any format!")


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) == 1:
    #     show_usage()
    # else:
    main()
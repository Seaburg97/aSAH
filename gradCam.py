import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from GradcamMain.pytorch_grad_cam import GradCAM
from GradcamMain.pytorch_grad_cam.utils.image import show_cam_on_image
from GradcamMain.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchio as tio
import json
import argparse
from data.mydata import CT3DDataset
from model import resnet3D, CBAM_resnet3D
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from datetime import datetime
import nibabel as nib

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode, warmup_steps, warmup_factor, patience, factor, verbose=True):
        super().__init__(optimizer, mode, patience=patience, factor=factor, verbose=verbose)
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.current_step = 0

    def warmup_lambda(self, step):
        return (self.warmup_factor * step / self.warmup_steps) + (1 - self.warmup_factor)

    def step(self, metrics):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            super().step(metrics)

def load_model(model_path, model_type, device):
    """
    Load a pre-trained model

    Args:
    - model_path: Path to the saved model checkpoint
    - model_type: Type of model ('resnet3D' or 'CBAM')
    - device: torch device

    Returns:
    - Loaded model
    """
    # 根据模型类型初始化模型架构
    if model_type == 'resnet3D':
        model = resnet3D.resnet50forOutcome(input_cha=1, num_classes=2).to(device)
    elif model_type == 'CBAM':
        model = CBAM_resnet3D.resnet50forOutcomeCBAM(input_cha=1, num_classes=2).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)

    # 如果模型使用了 DataParallel，需要特殊处理
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model

class RegressorOutputTarget:
    def __init__(self, index):
        self.index = index

    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output[self.index]  # 一维张量直接索引
        elif model_output.dim() == 2 and model_output.shape == (1, 1):
            return model_output.squeeze()  # 将 [1, 1] 形状的张量转换为标量
        else:
            return model_output[:, self.index]

def visualize_gradcam(model, dataset, device, output_dir, fold, num_samples=20):
    """
    Visualize Grad-CAM for 3D medical images across multiple folds

    Args:
    - model: Trained PyTorch model
    - dataset: PyTorch dataset
    - device: torch.device
    - output_dir: Directory to save visualizations
    - fold: Current fold number
    - num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # 选择目标层（根据模型架构调整）
    target_layers = [model.layer3[-1]] if hasattr(model, 'layer3') else [model.features[-1]]

    # 初始化 Grad-CAM
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    # 修改为：
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        # 如果需要指定设备，可以这样做
        # device=device  # 使用您已经定义的 device
    )

    # for i in range(min(num_samples,len(dataset))):
    for i in range(len(dataset)):
        # 获取图像和标签
        image, label, patient_id = dataset[i]
        # image = np.transpose(image, (3, 1, 2, 0))
        if label>2:
            image = image.unsqueeze(0).to(device)  # 添加批次维度
            # 创建病人特定的输出文件夹
            patient_output_dir = os.path.join(output_dir,str(label), patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            # 创建目标
            targets = [RegressorOutputTarget(0)]  # 0 是回归任务的输出索引

            # # 计算 CAM
            grayscale_cam = cam(input_tensor=image)
            grayscale_cam = grayscale_cam[0, :,:,:]  # 批次中的第一个图像
            print(f"生成{patient_id}")
            for slice_idx in range(image.shape[4]):
                # 计算 CAM
                image_slice = image[0, 0, :, :, slice_idx].cpu().numpy()
                cam_slice = grayscale_cam[:, :, slice_idx]

                # 归一化图像切片
                image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())

                # 创建一个包含三个子图的图像
                plt.figure(figsize=(15, 5))

                # 原始图像
                plt.subplot(1, 3, 1)
                plt.imshow(image_slice_normalized, cmap='gray')
                plt.title(f'Original Image\nPatient ID: {patient_id}\nSlice: {slice_idx}\nLabel: {label:.2f}')
                plt.axis('off')

                # Grad-CAM 热力图
                plt.subplot(1, 3, 2)
                plt.imshow(image_slice_normalized, cmap='gray', alpha=0.5)
                plt.imshow(cam_slice, cmap='jet', alpha=0.5)
                plt.title('Grad-CAM Activation')
                plt.axis('off')

                # 单独的热力图
                plt.subplot(1, 3, 3)
                plt.imshow(cam_slice, cmap='jet')
                plt.title('Heatmap')
                plt.axis('off')

                # 调整布局并保存
                plt.tight_layout()
                plt.savefig(os.path.join(patient_output_dir, f'slice_{slice_idx:03d}.png'), dpi=300)
                plt.close()

def main(args):

    # 选择使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 指定使用哪些卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 使用 TorchIO 进行 3D 数据增强和标准化
    data_transform = {
        'train': tio.Compose([
            tio.Resize((320, 320, 64)),
            tio.ToCanonical(),
            tio.ZNormalization()
        ]),'val': tio.Compose([
            tio.Resize((320, 320, 64)),
            tio.ToCanonical(),
            tio.ZNormalization()
        ])
    }

    # 创建数据集
    full_dataset = CT3DDataset(os.path.join(args.train_folder,args.train_mode), transform=None)

    # 设置分层交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)

    # 准备输出目录
    base_output_dir = os.path.join(args.checkpoint_dir, 'gradcam_visualizations',args.model,
                                   args.train_mode,)
    os.makedirs(base_output_dir, exist_ok=True)

    def find_latest_file(directory):
        # 获取文件夹中的所有文件名
        files = os.listdir(directory)

        # 过滤出符合日期时间格式的文件名
        date_files = [f for f in files if is_valid_datetime_format(f)]

        # 如果文件夹中没有符合格式的文件，返回None
        if not date_files:
            return None

        # 解析文件名并找到最近的文件
        latest_file = max(date_files, key=lambda f: datetime.strptime(f, '%Y-%m-%d %H:%M'))

        return latest_file

    def is_valid_datetime_format(filename):
        # 假设文件名格式为 'YYYY-MM-DD_HH-MM-SS'
        try:
            datetime.strptime(filename, '%Y-%m-%d %H:%M')
            return True
        except ValueError:
            return False

    # 遍历每一折
    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset, full_dataset.labels)):
        if fold==0:
            print(f'处理第{fold+1}折......')
            # 创建训练集和验证集
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

            # 应用变换
            train_dataset.dataset.transform = data_transform["train"]
            val_dataset.dataset.transform = data_transform["val"]

            # 构建模型检查点路径
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                args.model,
                args.train_mode,
                find_latest_file(os.path.join(args.checkpoint_dir,args.model,args.train_mode)),
                f'fold_{fold + 1}',
                f'f1Best_model_fold_{fold + 1}.pth'
            )

            # 加载模型
            model = load_model(checkpoint_path, args.model, device)
            print('加载模型的路径为：',checkpoint_path)
            # 准备输出目录
            output_dir = os.path.join(base_output_dir,f'fold_{fold + 1}')
            os.makedirs(output_dir, exist_ok=True)

            # 生成 Grad-CAM 可视化
            visualize_gradcam(model, train_dataset, device, output_dir, fold + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for 3D CT Classification')
    parser.add_argument('--train_folder', type=str, default="./data/output/train/",
                        help='Path to the training data folder')
    parser.add_argument('--train_mode', type=str, default="post",#选择pre还是post
                        help='train mode')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/regression/5fold/',
                        help='Directory to save checkpoints')
    parser.add_argument('--random_seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='CBAM', help="Choose model type: 'resnet3D' or 'CBAM'")

    args = parser.parse_args()
    main(args)
# --coding:utf-8--
import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.calculate import RegressionWithBinaryMetrics
from utils.pre import RandomContrast
from utils.scheduler import WarmupReduceLROnPlateau, WeightedMSELoss
import torchio as tio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from model import resnet3D, CBAM_resnet3D
from data.mydata import CT3DDataset
from datetime import datetime
import logging


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_model_and_metrics(log_dir, epoch, model, optimizer, scheduler, metrics):
    """
    Save model and metrics at early stopping
    """
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
    metrics_json.pop('index', None)
    metrics_json.pop('label', None)
    metrics_json.pop('pre', None)

    with open(os.path.join(log_dir, 'early_stop_results.json'), 'w') as json_file:
        json.dump(metrics_json, json_file, indent=4)

    results_df = pd.DataFrame({
        'Patient_ID': metrics.get('index', []),
        'True_Label': metrics.get('label', []),
        'Prediction': metrics.get('pre', [])
    })
    results_df.to_csv(os.path.join(log_dir, 'early_stop_results.csv'), index=False)

    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': metrics.get('train_loss', 0)
    }, os.path.join(log_dir, 'early_stop_model.pth'))

    return log_dir


def calculate_class_weights(dataset):
    """
    Calculate class weights for imbalanced datasets
    """
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)


def train(model, train_loader, criterion, optimizer, device, epoch, writer, logger, metrics_calculator):
    model.train()
    running_loss = 0.0
    all_true_labels = []
    all_predictions = []

    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch + 1} - LR: {current_lr:.6f}")
    progress_bar = tqdm(train_loader, desc=f"[Train]")
    for inputs, labels, id in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        outputs = torch.clamp(outputs, min=0, max=6)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_true_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.detach().cpu().numpy())

    avg_loss = running_loss / len(train_loader.dataset)
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    metrics = metrics_calculator.compute_metrics(
        torch.tensor(all_true_labels),
        torch.tensor(all_predictions)
    )

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('MAE/train', metrics['mae'], epoch)
    writer.add_scalar('Accuracy/train', metrics['accuracy'], epoch)
    writer.add_scalar('AUC/train', metrics['auc'], epoch)
    writer.add_scalar('f1/train', metrics['f1'], epoch)
    writer.add_scalar('Sensitivity/train', metrics['sensitivity'], epoch)
    writer.add_scalar('Specificity/train', metrics['specificity'], epoch)

    logger.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Train--Loss:{avg_loss:.4f}|MAE:{metrics['mae']:.4f}|AUC:{metrics['auc'] * 100:.2f}%|f1:{metrics['f1'] * 100:.2f}%|acc:{metrics['accuracy'] * 100:.2f}%|speci:{metrics['specificity'] * 100:.2f}%|sensi:{metrics['sensitivity'] * 100:.2f}%")

    return avg_loss, metrics


def validate(model, val_loader, criterion, device, epoch, writer, metrics_calculator):
    model.eval()
    running_loss = 0.0
    all_true_labels = []
    all_predictions = []
    all_index = []

    with torch.no_grad():
        for inputs, labels, index in tqdm(val_loader, desc=f"[Val]"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            outputs = torch.clamp(outputs, min=0, max=6)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            all_index.append(index)

    avg_loss = running_loss / len(val_loader.dataset)
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    all_index_num = np.array([item for sublist in all_index for item in sublist])
    metrics = metrics_calculator.compute_metrics(
        torch.tensor(all_true_labels),
        torch.tensor(all_predictions)
    )

    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('MAE/val', metrics['mae'], epoch)
    writer.add_scalar('Accuracy/val', metrics['accuracy'], epoch)
    writer.add_scalar('AUC/val', metrics['auc'], epoch)
    writer.add_scalar('f1/val', metrics['f1'], epoch)
    writer.add_scalar('Sensitivity/val', metrics['sensitivity'], epoch)
    writer.add_scalar('Specificity/val', metrics['specificity'], epoch)
    return all_true_labels, all_predictions.round(5), all_index_num, avg_loss, metrics['mae'], metrics[
        'accuracy'], metrics['auc'], metrics['f1'], metrics['sensitivity'], metrics['specificity']


def main(args):
    # Setup logging and directories
    current_date = str(datetime.now().strftime('%Y-%m-%d %H:%M'))
    log_dir = os.path.join(args.checkpoint_dir, args.model,args.train_mode,current_date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_logf = log_dir + "/training_log.txt"
    log_debugf = log_dir + "/debug.txt"
    logger = setup_logger(log_logf)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    with open(log_debugf, "w") as log_file:
        json.dump(vars(args), log_file, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")

    # Data augmentation and normalization
    data_transform = {
        'train': tio.Compose([
            tio.Resize((320, 320, 64)),
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
            tio.RandomNoise(mean=0, std=(0, 0.1)),
            RandomContrast(augmentation_factor=(0.75, 1.25)),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=(10, 10, 10)),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
            tio.ToCanonical(),
            tio.ZNormalization()
        ]),
        'val': tio.Compose([
            tio.Resize((320, 320, 64)),
            tio.ToCanonical(),
            tio.ZNormalization()
        ]),
    }

    train_dataset = CT3DDataset(args.train_folder+args.train_mode+"/", transform=data_transform['train'])
    val_dataset = CT3DDataset(args.test_folder+args.train_mode+"/", transform=data_transform['val'])

    class_weights = calculate_class_weights(train_dataset).to(device)
    logger.info(f"Class weights: {class_weights}")

    metrics_calculator = RegressionWithBinaryMetrics(threshold=2.5)
    criterion = WeightedMSELoss()

    if args.model == 'resnet3D':
        model = resnet3D.resnet50forOutcome(input_cha=1, num_classes=2).to(device)
    elif args.model == 'CBAM':
        model = CBAM_resnet3D.resnet50forOutcomeCBAM(input_cha=1, num_classes=2).to(device)
    else:
        raise ValueError("Invalid model name.")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = WarmupReduceLROnPlateau(optimizer, mode='min', warmup_steps=args.warmup_epochs,
                                        warmup_factor=0.9,
                                        patience=12, factor=0.3, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                            pin_memory=True)

    writer = SummaryWriter(log_dir)

    # Training Loop with early stopping based on training loss
    best_train_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(args.num_epochs):
        train_loss, train_metrics = train(model, train_loader, criterion, optimizer, device, epoch, writer, logger,
                                        metrics_calculator)

        label, pre, index, val_loss, mae, acc, auc, f1, sensitivity, specificity = validate(
            model, val_loader, criterion, device, epoch, writer, metrics_calculator)
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Vali--loss:{val_loss:.4f}|MAE:{mae:.4f}|AUC:{auc * 100:.2f}%|f1:{f1 * 100:.2f}%|acc:{acc * 100:.2f}%|speci:{specificity * 100:.2f}%|sensi:{sensitivity * 100:.2f}%")

        scheduler.step(val_loss)

        # Early stopping based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            counter = 0
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "mae": mae,
                "f1": f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "auc": auc,
                "index": index,
                "label": label,
                "pre": pre
            }
            save_model_and_metrics(log_dir, epoch, model, optimizer, scheduler, metrics)
            logger.info("   Saved new best model!")
        else:
            counter += 1

        if counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D CT Classification Training')
    parser.add_argument('--train_folder', type=str, default="./pengzhannet/data/datayjs/",
                        help='Path to the training data folder')
    parser.add_argument('--test_folder', type=str, default="./dataO/datafy/",
                        help='Path to the training data folder')
    parser.add_argument('--train_mode', type=str, default="pre",
                        help='train mode')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=8, help='Number of epochs for warmup')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/regression/whole/',
                        help='Directory to save checkpoints')
    parser.add_argument('--random_seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='CBAM', help="Choose model type: 'resnet18' or 'resnet50'")
    parser.add_argument('--optim', type=str, default='AdamW', help="Choose optimizer type: 'SGD' or 'Adam'")
    parser.add_argument('--gpuid', type=str, default='1,2', help="Choose gpu")

    args = parser.parse_args()
    main(args)
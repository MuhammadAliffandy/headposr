import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights , EfficientNet_B0_Weights 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R  
import numpy as np
import os
import cv2
from scipy.io import loadmat
import time
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from memory_profiler import profile

class LoadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.poses = []
        self._load_data()

    def _load_data(self):
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".jpg"):  
                image_path = os.path.join(self.data_dir, file_name)
                mat_path = image_path.replace(".jpg", ".mat")
                
                image = cv2.imread(image_path)
                mat_data = loadmat(mat_path)
                
                
                pose = mat_data['Pose_Para'][0][:3]  
                
                self.images.append(image)
                self.poses.append(pose)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(pose, dtype=torch.float32)


class LoadBIWIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.poses = []
        self._load_data()

    def _load_data(self):
        
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)

            if os.path.isdir(folder_path):  
                
                for file_name in os.listdir(folder_path):
                    if file_name.endswith("rgb.png"):  
                        image_path = os.path.join(folder_path, file_name)
                        pose_path = image_path.replace("rgb.png", "pose.txt")
                        
                        
                        image = cv2.imread(image_path)
                        
                        
                        if os.path.exists(pose_path):
                            with open(pose_path, 'r') as f:
                                pose_lines = f.readlines()
                                
                                rotation_matrix = [
                                    [float(value) for value in pose_lines[0].strip().split()],
                                    [float(value) for value in pose_lines[1].strip().split()],
                                    [float(value) for value in pose_lines[2].strip().split()]
                                ]
                                
                                
                                r = R.from_matrix(rotation_matrix)
                                yaw, pitch, roll = r.as_euler('zyx', degrees=False)  
                                
                                
                                translation = [float(value) for value in pose_lines[3].strip().split()]
                                
                                
                                self.images.append(image)
                                self.poses.append([yaw, pitch, roll])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(pose, dtype=torch.float32)



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    
])






dataset = LoadBIWIDataset('./BIWI/faces_0/train', transform=transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Connector(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super(Connector, self).__init__()
        self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=1)  
        self.scale_factor = scale_factor  

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.downsampler(x)  
        
        
        if self.scale_factor > 1:
            H = H // self.scale_factor
            W = W // self.scale_factor
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        x = x.view(B, -1, H * W)  
        x = x.permute(0, 2, 1)  
        return x



class HeadPosr(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(HeadPosr, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  
        
        self.connector = Connector(in_channels=1280, out_channels=16)
        
        self.positional_encoding = PositionalEncoding(d_model=16)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, activation='relu', batch_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.pose_head = nn.Linear(16, 3)

    def forward(self, x):
        
        x = self.backbone(x)  
        
        x = self.connector(x)  
        
        x = self.positional_encoding(x)

        x = self.transformer(x)  

        x = self.dropout(x)

        x = x.mean(dim=1)  

        x = self.pose_head(x)  
    
        return x


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def v_loss(outputs, labels):
    diff = outputs - labels
    return torch.mean(diff ** 2)  


@profile
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=90, model_name='headposr_model', patience=5):
    model.train() 
    best_loss = float('inf')
    patience_counter = 0
    total_start_time = time.time()
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  
            running_loss = 0.0
            running_v_loss = 0.0

            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                
                outputs = outputs.view(-1, 3)  
                labels = labels.view(-1, 3)    

                loss = criterion(outputs, labels)  
                loss_v = v_loss(outputs, labels)  

                loss.backward()  
                optimizer.step()  
                running_loss += loss.item()
                running_v_loss += loss_v.item()

                current_lr = get_lr(optimizer)
                torch.cuda.empty_cache()
                if i % 10 == 9:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}, v_loss: {running_v_loss/10:.4f}, LR: {current_lr:.6f}")
                    running_loss = 0.0
            scheduler.step()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

            avg_v_loss = running_v_loss / len(dataloader)  
            if avg_v_loss < best_loss:
                best_loss = avg_v_loss
                patience_counter = 0  
                print(f"Validation loss improved.")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}.")

            
            if patience_counter >= patience:
                print(f"Early stopping triggered. No improvement in {patience} consecutive epochs.")
                torch.save(model.state_dict(), f"{model_name}_over_{epoch}.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving the current model...")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Training completed in {total_duration:.2f} seconds.")
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Final model saved to {model_name}.pth.")




def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def smooth_l1_loss(preds, labels, beta=1.0):
    diff = torch.abs(preds - labels)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return torch.mean(loss)

def mse_loss(preds, labels):
    return torch.mean((preds - labels) ** 2)

def mae_loss(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def angular_distance_loss(preds, labels):
    preds_norm = preds / preds.norm(dim=1, keepdim=True)
    labels_norm = labels / labels.norm(dim=1, keepdim=True)
    cos_sim = torch.sum(preds_norm * labels_norm, dim=1)
    loss = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
    return torch.mean(loss)


def test_model_with_traffic(model, dataloader):
    model.eval()
    
    total_smooth, total_mae, total_mse, total_angular = 0.0, 0.0, 0.0, 0.0
    total_samples = 0

    all_yaw_preds, all_pitch_preds, all_roll_preds = [], [], []
    all_yaw_gt, all_pitch_gt, all_roll_gt = [], [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            
            outputs = outputs.view(-1, 3)  
            labels = labels.view(-1, 3)    

            smooth = smooth_l1_loss(outputs, labels)
            mae = mae_loss(outputs, labels)
            mse = mse_loss(outputs, labels)
            angular = angular_distance_loss(outputs, labels)

            
            total_smooth += smooth.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_mse += mse.item() * inputs.size(0)
            total_angular += angular.item() * inputs.size(0)

            total_samples += inputs.size(0)

            
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            
            all_yaw_preds.extend(outputs_np[:, 0])
            all_pitch_preds.extend(outputs_np[:, 1])
            all_roll_preds.extend(outputs_np[:, 2])

            all_yaw_gt.extend(labels_np[:, 0])
            all_pitch_gt.extend(labels_np[:, 1])
            all_roll_gt.extend(labels_np[:, 2])

            
            for j in range(len(outputs)):
                yaw_pred, pitch_pred, roll_pred = outputs_np[j]
                yaw_gt, pitch_gt, roll_gt = labels_np[j]
                
                print(f"Sample {i*len(outputs)+j+1}:")
                print(f"  Prediksi - Yaw: {yaw_pred:.2f}, Pitch: {pitch_pred:.2f}, Roll: {roll_pred:.2f}")
                print(f"  Ground Truth - Yaw: {yaw_gt:.2f}, Pitch: {pitch_gt:.2f}, Roll: {roll_gt:.2f}")
                print(f"  SMOOTH L1: {smooth.item():.4f}")
                print(f"  MSE: {mse.item():.4f}")
                print(f"  MAE: {mae.item():.4f}")
                print(f"  Angular Distance: {angular.item():.4f}")

    
    avg_smooth = total_smooth / total_samples
    avg_mae = (total_mae / total_samples) * (180 / np.pi)
    avg_mse = (total_mse / total_samples) * (180 / np.pi)
    avg_angular = (total_angular / total_samples) * (180 / np.pi)


    print(f"\nRata-rata SMOOTH L1: {avg_smooth:.4f}")
    print(f"Rata-rata MSE: {avg_mse:.4f}")
    print(f"Rata-rata MAE: {avg_mae:.4f}")
    print(f"Rata-rata Angular Distance: {avg_angular:.4f}")

    
    plot_comparison(all_yaw_preds, all_yaw_gt, 'Yaw')
    plot_comparison(all_pitch_preds, all_pitch_gt, 'Pitch')
    plot_comparison(all_roll_preds, all_roll_gt, 'Roll')


def plot_comparison(preds, gt, title):
    plt.figure(figsize=(10, 5))
    plt.plot(gt, label='Ground Truth', color='b')
    plt.plot(preds, label='Prediksi', color='r')
    plt.title(f'Perbandingan Ground Truth vs Prediksi - {title}')
    plt.xlabel('Sampel')
    plt.ylabel(f'Nilai {title}')
    plt.legend()
    plt.grid(True)
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = HeadPosr().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001 , weight_decay=1e-5 )
    scheduler = get_scheduler(optimizer)

    mode = input("Choose mode (train/test): ").strip().lower()

    if mode == "train":
        print("Starting Training...")
        train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=90 , model_name ='biwi_trainEH64D16_90_epouch')
    
    elif mode == "test":
        model.load_state_dict(torch.load("biwi_trainEH38D16_75_epouch.pth"))
        print("Starting Testing...")
        test_model_with_traffic(model, dataloader)



if __name__ == "__main__":
    main()
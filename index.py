import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from scipy.io import loadmat
import time
import math

# Dataset AFLW2000-3D dengan file .mat untuk pose
class AFLW20003DDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.poses = []
        self._load_data()

    def _load_data(self):
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".jpg"):  # Asumsi gambar berformat .jpg
                image_path = os.path.join(self.data_dir, file_name)
                mat_path = image_path.replace(".jpg", ".mat")
                
                image = cv2.imread(image_path)
                mat_data = loadmat(mat_path)
                
                # Ekstraksi pose kepala (yaw, pitch, roll) dari file .mat
                pose = mat_data['Pose_Para'][0][:3]  # Ambil yaw, pitch, roll
                
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

# Dataset 300W-LP
class Dataset300WLP(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.poses = []
        self._load_data()

    def _load_data(self):
        # Daftar semua folder yang ada di dalam data_dir (afw, helen, ibug, lfpw, dll)
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)

            if os.path.isdir(folder_path):  # Hanya memproses folder
                # Baca semua file gambar dalam folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".jpg"):  # Hanya memproses file .jpg
                        image_path = os.path.join(folder_path, file_name)
                        mat_path = image_path.replace(".jpg", ".mat")
                        
                        # Baca gambar
                        image = cv2.imread(image_path)
                        
                        # Baca file .mat untuk pose (yaw, pitch, roll)
                        if os.path.exists(mat_path):
                            mat_data = loadmat(mat_path)
                            pose = mat_data['Pose_Para'][0][:3]  # Ambil yaw, pitch, roll
                            
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


# Transformasi data dengan augmentasi (cropping dan random scaling)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset dan Dataloader
data_dir = "./300W-LP/300W_LP"
# dataset = AFLW20003DDataset(data_dir, transform=transform)
dataset = Dataset300WLP(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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

# Connector (downsampler and reshaper)
class Connector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Connector, self).__init__()
        self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.downsampler(x)  # Mengurangi channel
        x = x.view(B, -1, H * W)  # Mengubah jadi sekuensial, misalnya B x d x A (A=H/S*W/S)
        x = x.permute(0, 2, 1)  # B x A x d untuk transformer input
        return x

# Arsitektur Model
class HeadPosr(nn.Module):
    def __init__(self):
        super(HeadPosr, self).__init__()
        # Backbone menggunakan ResNet50, dipotong hingga layer tertentu
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Potong sampai layer terakhir sebelum pooling
        
        # Connector untuk downsampling dan reshaping
        self.connector = Connector(in_channels=2048, out_channels=256)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=256)
        
        # Transformer Encoder dengan beberapa lapisan
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Head untuk prediksi yaw, pitch, roll
        self.pose_head = nn.Linear(256, 3)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)  # Output size: [batch_size, 2048, H, W]
        
        # Connector: downsampler and reshaper
        x = self.connector(x)  # Output size: [batch_size, A, 256]
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # Output size: [A, batch_size, 256]

        # Global average pooling over the sequence length dimension (A)
        x = x.mean(dim=1)  # Output size: [batch_size, 256]

        # Pose prediction (yaw, pitch, roll)
        x = self.pose_head(x)  # Output size: [batch_size, 3]
        return x

# Fungsi untuk mendapatkan nilai learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Fungsi untuk melatih model
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=90):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Pastikan output dan label memiliki dimensi yang sama
            outputs = outputs.view(-1, 3)  # Output size: [batch_size, 3]
            labels = labels.view(-1, 3)    # Label size: [batch_size, 3]

            loss = criterion(outputs, labels)  # Hitung loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameter model
            running_loss += loss.item()

            current_lr = get_lr(optimizer)

            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}, LR: {current_lr:.6f}")
                running_loss = 0.0

        scheduler.step()
    
    # Simpan model
    torch.save(model.state_dict(), "headposr_model_300w.pth")
    print("Model saved to headposr_model_300w.pth")

# Scheduler untuk learning rate
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def mae_loss(preds, labels):
    return torch.mean(torch.abs(preds - labels))

# Fungsi untuk menguji model
def test_model_with_predictions(model, dataloader, criterion):
    model.eval()
    running_mae = 0.0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            mae = mae_loss(outputs, labels)
            running_mae += mae.item()
            
            # Print prediksi pose
            yaw, pitch, roll = outputs[0].cpu().numpy()
            print(f"{i}. Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")

    avg_mae = running_mae / len(dataloader)
    print(f"Average MAE: {avg_mae:.4f}")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function
def main():
    model = HeadPosr().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = get_scheduler(optimizer)

    mode = input("Choose mode (train/test): ").strip().lower()

    if mode == "train":
        print("Starting Training...")
        train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=3)
    
    elif mode == "test":
        model.load_state_dict(torch.load("headposr_model_300w.pth"))
        print("Starting Testing...")
        test_model_with_predictions(model, dataloader, criterion)

if __name__ == "__main__":
    main()

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
import matplotlib.pyplot as plt

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

# Transformasi data dengan augmentasi (cropping dan random scaling)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset dan Dataloader
data_dir = "./300W-LP/300W_LP/AFW"
dataset = AFLW20003DDataset(data_dir, transform=transform)
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
    torch.save(model.state_dict(), "headposr_model_300w_90.pth")
    print("Model saved to headposr_model_aflw.pth")

# Scheduler untuk learning rate
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def mae_loss(preds, labels):
    return torch.mean(torch.abs(preds - labels))

# Fungsi untuk menguji model
# def test_model_with_visualization(model, dataloader):
#     model.eval()
#     all_yaw_preds, all_pitch_preds, all_roll_preds = [], [] , []
#     all_yaw_gt, all_pitch_gt, all_roll_gt = [], [], []

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)

#             # Pastikan output dan label memiliki dimensi yang sama
#             outputs = outputs.view(-1, 3)  # Output size: [batch_size, 3]
#             labels = labels.view(-1, 3)    # Label size: [batch_size, 3]

#             mae = mae_loss(outputs, labels)

#             # Ekstrak prediksi dan ground truth
#             outputs_np = outputs.cpu().numpy()
#             labels_np = labels.cpu().numpy()

#             # Simpan prediksi dan ground truth untuk visualisasi
#             all_yaw_preds.extend(outputs_np[:, 0])
#             all_pitch_preds.extend(outputs_np[:, 1])
#             all_roll_preds.extend(outputs_np[:, 2])

#             all_yaw_gt.extend(labels_np[:, 0])
#             all_pitch_gt.extend(labels_np[:, 1])
#             all_roll_gt.extend(labels_np[:, 2])

#             # Cetak prediksi dan ground truth untuk debugging
#             yaw_pred, pitch_pred, roll_pred = outputs[0].cpu().numpy()
#             yaw_gt, pitch_gt, roll_gt = labels[0].cpu().numpy()
            
#             print(f"Sample {i+1}:")
#             print(f"  Prediksi - Yaw: {yaw_pred:.2f}, Pitch: {pitch_pred:.2f}, Roll: {roll_pred:.2f}")
#             print(f"  Ground Truth - Yaw: {yaw_gt:.2f}, Pitch: {pitch_gt:.2f}, Roll: {roll_gt:.2f}")
#             print(f"  MAE: {mae.item():.4f}")

#     # Visualisasi Ground Truth vs Prediksi
#     plot_comparison(all_yaw_preds, all_yaw_gt, 'Yaw')
#     plot_comparison(all_pitch_preds, all_pitch_gt, 'Pitch')
#     plot_comparison(all_roll_preds, all_roll_gt, 'Roll')

# Fungsi untuk memplot Ground Truth vs Prediksi
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

def draw_euler_angles(image, yaw, pitch, roll, center=None, size=100):
    """
    Menggambar garis yang merepresentasikan Yaw, Pitch, dan Roll pada gambar.
    
    Args:
    - image: Gambar yang akan ditampilkan garis Euler angles-nya.
    - yaw: Sudut yaw (radian).
    - pitch: Sudut pitch (radian).
    - roll: Sudut roll (radian).
    - center: Titik pusat untuk menggambar garis. Jika None, gunakan tengah gambar.
    - size: Panjang garis Euler angles.
    """
    h, w, _ = image.shape
    if center is None:
        center = (w // 2, h // 2)  # Default di tengah gambar

    # Konversi Euler angles (yaw, pitch, roll) dari radian ke derajat
    yaw_deg = yaw * (180.0 / np.pi)
    pitch_deg = pitch * (180.0 / np.pi)
    roll_deg = roll * (180.0 / np.pi)
    
    # Menghitung arah garis untuk masing-masing sudut
    yaw_x = int(center[0] + size * math.sin(yaw))
    yaw_y = int(center[1] - size * math.cos(yaw))

    pitch_x = int(center[0] + size * math.sin(pitch))
    pitch_y = int(center[1] + size * math.cos(pitch))

    roll_x = int(center[0] + size * math.cos(roll))
    roll_y = int(center[1] + size * math.sin(roll))

    # Menggambar garis yaw (merah), pitch (hijau), roll (biru)
    image = cv2.line(image, center, (yaw_x, yaw_y), (0, 0, 255), 2)   # Yaw (merah)
    image = cv2.line(image, center, (pitch_x, pitch_y), (0, 255, 0), 2)  # Pitch (hijau)
    image = cv2.line(image, center, (roll_x, roll_y), (255, 0, 0), 2)   # Roll (biru)

    # Tampilkan informasi Euler angles
    cv2.putText(image, f"Yaw: {yaw_deg:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Pitch: {pitch_deg:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Roll: {roll_deg:.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image

# Fungsi untuk menguji model dengan visualisasi Euler angles pada gambar
def test_model_with_visualization(model, dataloader, criterion):
    model.eval()
    running_mae = 0.0
    running_mae_yaw = 0.0
    running_mae_pitch = 0.0
    running_mae_roll = 0.0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Hitung MAE keseluruhan
            mae = mae_loss(outputs, labels)
            running_mae += mae.item()

            # Ekstrak prediksi dan ground truth
            yaw_pred, pitch_pred, roll_pred = outputs[0].cpu().numpy()
            yaw_gt, pitch_gt, roll_gt = labels[0].cpu().numpy()
            
            # Hitung MAE untuk setiap komponen
            running_mae_yaw += abs(yaw_pred - yaw_gt)
            running_mae_pitch += abs(pitch_pred - pitch_gt)
            running_mae_roll += abs(roll_pred - roll_gt)

            # Ambil gambar asli dari dataloader
            image = inputs[0].cpu().numpy().transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ubah ke BGR untuk OpenCV
            
            # Gambar Euler angles berdasarkan prediksi
            image_with_angles = draw_euler_angles(image, yaw_pred, pitch_pred, roll_pred)

            # Tampilkan gambar dengan garis Euler angles
            cv2.imshow(f"Sample {i + 1}: Predicted Euler Angles", image_with_angles)
            cv2.waitKey(0)  # Tekan sembarang tombol untuk lanjut ke gambar berikutnya
            cv2.destroyAllWindows()

    avg_mae = running_mae / len(dataloader)
    avg_mae_yaw = running_mae_yaw / len(dataloader)
    avg_mae_pitch = running_mae_pitch / len(dataloader)
    avg_mae_roll = running_mae_roll / len(dataloader)

    # Cetak nilai MAE rata-rata
    print(f"Average MAE (Overall): {avg_mae:.4f}")
    print(f"Average MAE - Yaw: {avg_mae_yaw:.4f}")
    print(f"Average MAE - Pitch: {avg_mae_pitch:.4f}")
    print(f"Average MAE - Roll: {avg_mae_roll:.4f}")

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
        train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=90)
    
    elif mode == "test":
        model.load_state_dict(torch.load("headposr_model_300w_90.pth"))
        print("Starting Testing...")
        test_model_with_visualization(model, dataloader, criterion)

if __name__ == "__main__":
    main()

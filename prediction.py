import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# ----------------- CONFIG --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = (224, 224)
MEAN_VALUE = 0.5
STD_VALUE = 0.5

# Load class labels from training CSV
train_csv_path = "DATASET_CLEANED/train.csv"
train_csv = pd.read_csv(train_csv_path)
class_labels = sorted(train_csv['diagnosis'].unique())  # Ensure ordered

# ----------------- TRANSFORMS --------------------
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN_VALUE]*3, std=[STD_VALUE]*3)
])

# ----------------- MODEL ARCHITECTURE --------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        max_pool = torch.amax(x, dim=[2, 3], keepdim=True)
        out = self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool)))
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )
        self.cbam = CBAM(64)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

# ----------------- LOAD MODEL --------------------

num_classes = len(class_labels)
model = ImprovedCNNModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("cbam_cnn_model.pth", map_location=device))
model.eval()

# ----------------- PREDICTION FUNCTION --------------------

def predict_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    predictions = {}
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()

        predictions[image_file] = class_labels[pred_class]

    return predictions

# ----------------- RUN PREDICTIONS --------------------

image_folder = "C:/Users/Asus/OneDrive/Documents/1 WINTER SEM/HA/PROJECT/CODE/prediction/Severe DR"
predictions = predict_images(image_folder)

for img, pred in predictions.items():
    print(f"Image: {img} → Predicted Class: {pred}")

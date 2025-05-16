import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- CONFIG ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

IMAGE_SIZE = (224, 224)
MEAN_VALUE = 0.5
STD_VALUE = 0.5
BATCH_SIZE = 8
LEARNING_RATE = 0.0003
EPOCHS = 50
LATENT_DIM = 100
GAN_PRETRAIN_EPOCHS = 20
SYNTH_PER_CLASS = 2

DATASET_PATH = "DATASET_CLEANED"
# Note: CSV files and folders are swapped as per your strategy.
train_csv_path = os.path.join(DATASET_PATH, "test.csv")
val_csv_path = os.path.join(DATASET_PATH, "val.csv")
test_csv_path = os.path.join(DATASET_PATH, "train.csv")
train_folder = os.path.join(DATASET_PATH, "test_images")
val_folder = os.path.join(DATASET_PATH, "val_images")
test_folder = os.path.join(DATASET_PATH, "train_images")

os.makedirs(os.path.join(DATASET_PATH, "synthetic_images"), exist_ok=True)

# ---------------------- TRANSFORMS ----------------------------
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN_VALUE]*3, std=[STD_VALUE]*3)
])

# ---------------------- DATASET ----------------------------
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.dataframe.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

train_csv = pd.read_csv(train_csv_path)
val_csv = pd.read_csv(val_csv_path)
test_csv = pd.read_csv(test_csv_path)

class_counts = Counter(train_csv['diagnosis'])
max_class = max(class_counts.values())
minority_classes = [cls for cls, count in class_counts.items() if count < max_class]
class_weights = torch.tensor([max_class / class_counts[i] for i in range(len(class_counts))],
                               dtype=torch.float).to(device)

# ------------------- MODEL DEFINITIONS ----------------------------
# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        max_pool = torch.amax(x, dim=[2, 3], keepdim=True)
        out = self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool)))
        return x * self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

# CBAM Module combining Channel and Spatial Attention
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Improved CNN Model with CBAM
class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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

# Generator for GAN
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 112 * 112),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 3, 112, 112)

# Discriminator for GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == '__main__':
    # ------------------- PHASE 1: GAN PRE-TRAINING ------------------
    gan_train_dataset = CustomDataset(train_csv, train_folder, transform=data_transforms)
    gan_train_loader = DataLoader(gan_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    # Initialize GAN models
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    adversarial_loss = nn.BCELoss()

    print("Starting GAN pretraining...")
    for epoch in range(GAN_PRETRAIN_EPOCHS):
        for images, _ in gan_train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_images = generator(z)
            fake_images = nn.functional.interpolate(fake_images, size=(224, 224))
            # Train Discriminator
            disc_optimizer.zero_grad()
            d_real = discriminator(images)
            d_fake = discriminator(fake_images.detach())
            d_loss = adversarial_loss(d_real, real_labels) + adversarial_loss(d_fake, fake_labels)
            d_loss.backward()
            disc_optimizer.step()
            # Train Generator
            gen_optimizer.zero_grad()
            d_gen = discriminator(fake_images)
            g_loss = adversarial_loss(d_gen, real_labels)
            g_loss.backward()
            gen_optimizer.step()
        print(f"[GAN Epoch {epoch+1}/{GAN_PRETRAIN_EPOCHS}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}", flush=True)
    print("Saving pretrained generator weights...")
    torch.save(generator.state_dict(), 'generator_pretrained.pth')

    # ------------------- PHASE 2: CNN TRAINING ------------------
    train_dataset = CustomDataset(train_csv, train_folder, transform=data_transforms)
    val_dataset = CustomDataset(val_csv, val_folder, transform=data_transforms)
    test_dataset = CustomDataset(test_csv, test_folder, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    cnn_model = ImprovedCNNModel(num_classes=len(train_csv['diagnosis'].unique())).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='max', factor=0.5, patience=3)

    # Initialize AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    history = {'loss': [], 'val_acc': [], 'val_f1': []}

    print("Starting CNN training with GAN augmentation using AMP...")
    for epoch in range(EPOCHS):
        cnn_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            fake_images, fake_labels = [], []
            # Use GAN augmentation (after initial 5 epochs to allow CNN stabilization)
            if epoch > 5:
                for cls in minority_classes:
                    z = torch.randn(SYNTH_PER_CLASS, LATENT_DIM).to(device)
                    synth_imgs = generator(z)
                    synth_imgs = nn.functional.interpolate(synth_imgs, size=(224, 224))
                    synth_lbls = torch.full((SYNTH_PER_CLASS,), cls, dtype=torch.long).to(device)
                    fake_images.append(synth_imgs)
                    fake_labels.append(synth_lbls)
            if fake_images:
                images = torch.cat([images] + fake_images)
                labels = torch.cat([labels] + fake_labels)
            cnn_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = cnn_model(images)
                loss = classification_loss(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(cnn_optimizer)
            scaler.update()
        cnn_model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = cnn_model(images)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f} | Val Acc: {val_accuracy:.4f} | F1: {val_f1:.4f}")
        history['loss'].append(loss.item())
        history['val_acc'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        scheduler.step(val_accuracy)

    # ------------------- TESTING ---------------------------
    cnn_model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_model(images)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    accuracy_value = accuracy_score(test_labels, test_preds)
    precision_value = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall_value = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1_value = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    print(f"\nTest Accuracy: {accuracy_value:.4f} | Precision: {precision_value:.4f} | Recall: {recall_value:.4f} | F1: {f1_value:.4f}")

    # ----------------- MODEL SAVING -----------------------------
    MODEL_PATH = "cbam_cnn_model.pth"
    torch.save(cnn_model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # ----------------- PLOTTING -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Metrics over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

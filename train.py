import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "Dataset"
MODEL_PATH = "brain_tumor_cnn.pth"
METRICS_PATH = "metrics.txt"

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        conv_out = self.conv(dummy)
        flatten_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Trainer:
    def __init__(self):
        self.train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.test_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_data = datasets.ImageFolder(os.path.join(DATA_PATH, "Training"), transform=self.train_tf)
        self.test_data = datasets.ImageFolder(os.path.join(DATA_PATH, "Testing"), transform=self.test_tf)

        self.train_loader = DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=BATCH_SIZE)

        print("Класи:", self.train_data.classes)

        self.model = BrainTumorCNN().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        self.best_val_loss = float("inf")
        self.patience = 3
        self.counter = 0

        self.train_loss_hist = []
        self.train_acc_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []
        self.precision_hist = []

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, desc=f"Епоха {epoch+1}/{EPOCHS}", unit="batch")

        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            loop.set_postfix(loss=loss.item(), accuracy=f"{acc:.3f}")

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        self.train_loss_hist.append(epoch_loss)
        self.train_acc_hist.append(epoch_acc)

        print(f"\nTrain Loss: {epoch_loss:.3f} | Train Accuracy: {epoch_acc*100:.1f}%")

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        all_true = []
        all_pred = []

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_true.extend(labels.cpu().numpy())
                all_pred.extend(predicted.cpu().numpy())

        val_loss /= len(self.test_loader)
        val_acc = val_correct / val_total

        self.val_loss_hist.append(val_loss)
        self.val_acc_hist.append(val_acc)

        precision = precision_score(all_true, all_pred, average="weighted")
        self.precision_hist.append(precision)

        print(f"Validation Loss: {val_loss:.3f} | Validation Accuracy: {val_acc*100:.1f}%")
        print(f"Precision: {precision*100:.1f}%")

        return val_loss, precision, val_acc

    def save_if_improved(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(self.model.state_dict(), MODEL_PATH)
            print(">>> Модель покращена — збережено.")
        else:
            self.counter += 1
            print(f"Немає покращення ({self.counter}/{self.patience})")

        if self.counter >= self.patience:
            print("\n>>> Early stopping: навчання завершено.")
            return True
        return False

    def save_metrics(self):
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            f.write(f"Validation Accuracy: {self.val_acc_hist[-1]:.4f}\n")
            f.write(f"Precision: {self.precision_hist[-1]:.4f}\n")

        print("\nТочності записано у metrics.txt")

    def plot_graph(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.train_loss_hist, label="Train Loss")
        plt.plot(self.val_loss_hist, label="Validation Loss")
        plt.plot(self.train_acc_hist, label="Train Accuracy")
        plt.plot(self.val_acc_hist, label="Validation Accuracy")
        plt.plot(self.precision_hist, label="Precision")
        plt.legend()
        plt.title("Training Metrics")

        plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
        print("Графік збережено у файл: training_metrics.png")

        plt.show()

    def run(self):
        for epoch in range(EPOCHS):
            self.train_one_epoch(epoch)

            val_loss, precision, val_acc = self.validate()

            if self.save_if_improved(val_loss):
                break

        self.save_metrics()
        self.plot_graph()

        print("\nМодель збережено:", MODEL_PATH)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

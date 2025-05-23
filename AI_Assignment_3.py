import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import csv

# Configuration
class Config:
    n_channels = 32  # channel number
    n_classes = 6  # 6 hand classes and one idle class
    sampling_rate = 500  # sampling rate
    window_duration = 0.3  # Duration of each window in seconds (150 sample per gesture, so 150/500=0.3s)
    batch_size = 512  # batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Dataset Loader
class EEGDataset(Dataset):
    def __init__(self, time_series, labels):
        self.data = [torch.FloatTensor(x.T) for x in time_series]  # Transpose to (channels, time)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# CNN-LSTM Model
class EEG_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Temporal feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(Config.n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)  # Fixed-size output
        )

        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, Config.n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# data normalization
def z_score_normalize(eeg_data):
    """Normalize each channel to zero mean and unit variance"""
    means = np.mean(eeg_data, axis=1, keepdims=True)
    stds = np.std(eeg_data, axis=1, keepdims=True)
    return (eeg_data - means) / (stds + 1e-8)

# seperate data with fixed length windows
def create_windows(data, labels, window_len_samples):
    X, Y = [], []
    for i in range(0, len(data) - window_len_samples + 1, 75):
        window_data = data[i:i + window_len_samples]
        window_label = np.mean(labels[i:i + window_len_samples], axis=0) > 0.3
        X.append(window_data)
        Y.append(window_label.astype(np.float32))
    return X, Y

## main loop
# data reading
temp=pd.DataFrame();
data_train=pd.DataFrame();
label_train=pd.DataFrame();
data_test=pd.DataFrame();
label_test=pd.DataFrame();

for i in range(1,3):
  for j in range(1,4):
    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_data.csv')
    data_train = pd.concat([data_train, temp], axis=0)

    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_events.csv')
    label_train = pd.concat([label_train, temp], axis=0)

for i in range(1,3):
  for j in range(7,9):
    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_data.csv')
    data_test = pd.concat([data_test, temp], axis=0)

    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_events.csv')
    label_test = pd.concat([label_test, temp], axis=0)

n_channels = 32

ch_names = list(data_train.columns[1:])
train_data = np.array(data_train[ch_names], 'float32')

ch_names = list(label_train.columns[1:])
train_label = np.array(label_train[ch_names], 'float32')

ch_names = list(data_test.columns[1:])
test_data = np.array(data_test[ch_names], 'float32')

ch_names = list(label_test.columns[1:])
test_label = np.array(label_test[ch_names], 'float32')

# Filter parameters
fs = 500  # Sampling frequency (Hz)
lowcut = 7  # Lower cutoff frequency (Hz)
highcut = 35  # Upper cutoff frequency (Hz)
order = 4  # Filter order

# Create Butterworth bandpass filter
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(order, [low, high], btype='band')

# Apply filter to each channel
filtered_train_data = np.zeros_like(train_data)
filtered_test_data = np.zeros_like(test_data)

for i in range(n_channels):
    filtered_train_data[:,i] = signal.filtfilt(b, a, train_data[:,i])
    filtered_test_data[:,i] = signal.filtfilt(b, a, test_data[:,i])


# Data normalization
normalized_filtered_train_data = z_score_normalize(filtered_train_data)
normalized_filtered_test_data = z_score_normalize(filtered_test_data)

# create time windows for data
window_len_samples = int(Config.window_duration * Config.sampling_rate)
train_X, train_Y = create_windows(normalized_filtered_train_data, train_label, window_len_samples)
test_X, test_Y = create_windows(normalized_filtered_test_data, test_label, window_len_samples)

# Split data (X = list of epochs, Y = list/array of labels)
train_X, val_X, train_Y, val_Y = train_test_split(
    train_X, train_Y, test_size=0.2, random_state=42, stratify=None  # Set stratify=Y if labels are single-label
)

# Create tensor datasets
train_dataset = EEGDataset(train_X, train_Y)
val_dataset = EEGDataset(val_X, val_Y)
test_dataset = EEGDataset(test_X, test_Y)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
test_loader = DataLoader(test_dataset,batch_size=Config.batch_size)

#  Initialize model
model = EEG_CNN_LSTM().to(Config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

#  Training loop
for epoch in range(1, 5):
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(Config.device), y.to(Config.device)

        if (y.sum(dim=1) != 0).any():
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(Config.device), y.to(Config.device)

            if (y.sum(dim=1) != 0).any():
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.3).float()
                correct += (preds == y).sum().item()
                total += y.numel()

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss / len(val_loader):.4f} | "
              f"Accuracy: {correct/total:.4f}")

## testing loop
model.eval()
# initialization
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(Config.device), y.to(Config.device)

        if (y.sum(dim=1) != 0).any():
            outputs = model(X)
            loss = criterion(outputs, y)
            test_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.3).float()
            correct += (preds == y).sum().item()
            total += y.numel()

            # Store predictions and true labels
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())



    print(f"Test Loss: {test_loss / len(test_loader):.4f} | "
          f"Accuracy: {correct/total:.4f}")


# # ploting ROC Curves + AUC for Each Class
# # Concatenate all batches
# all_preds_np = np.concatenate(all_preds, axis=0)
# all_targets_np = np.concatenate(all_targets, axis=0)

# Convert predictions and targets to numpy arrays
all_preds_np = torch.cat(all_preds, dim=0).cpu().numpy()
all_targets_np = np.concatenate(all_targets, axis=0)

# Create DataFrame
df_results = pd.DataFrame()
for i in range(Config.n_classes):
    df_results[f'pred_class_{i}'] = all_preds_np[:, i]
    df_results[f'true_class_{i}'] = all_targets_np[:, i]

# Save to CSV
df_results.to_csv("eeg_predictions_vs_groundtruth.csv", index=False)
print("Predictions and ground truth saved to 'eeg_predictions_vs_groundtruth.csv'")

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# # Compute ROC curve and AUC for each class
# for i in range(Config.n_classes):
#     fpr[i], tpr[i], _ = roc_curve(all_targets_np[:, i], all_preds_np[:, i])
#     roc_auc[i] = roc_auc_score(all_targets_np[:, i], all_preds_np[:, i])
#
# # Plot
# plt.figure(figsize=(10, 8))
# for i in range(Config.n_classes):
#     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve by Class')
# plt.legend(loc='lower right')
# plt.grid()
# plt.tight_layout()
# plt.show()

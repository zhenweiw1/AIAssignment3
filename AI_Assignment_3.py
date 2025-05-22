import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import pandas as pd

# Configuration
class Config:
    n_channels = 32  # channel number
    n_classes = 6  # 6 hand classes and one idle class
    sampling_rate = 500  # sampling rate
    epoch_duration = 0.3  # Duration of each epoch in seconds (150 sample per gesture, so 150/500=0.3s)
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

def create_epochs(data, labels, epoch_len_samples):
    X, Y = [], []
    for i in range(0, len(data) - epoch_len_samples + 1, epoch_len_samples):
        epoch_data = data[i:i + epoch_len_samples]
        epoch_label = np.mean(labels[i:i + epoch_len_samples], axis=0) > 0.3
        X.append(epoch_data)
        Y.append(epoch_label.astype(np.float32))
    return X, Y


## main
temp=pd.DataFrame();
data_train=pd.DataFrame();
label_train=pd.DataFrame();
data_test=pd.DataFrame();
label_test=pd.DataFrame();

for i in range(1,4):
  for j in range(1,7):
    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_data.csv')
    data_train = pd.concat([data_train, temp], axis=0)

    temp = pd.read_csv('train_data/subj'+str(i)+'_series'+str(j)+'_events.csv')
    label_train = pd.concat([label_train, temp], axis=0)

for i in range(1,4):
  for j in range(7,8):
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

# # if add unkown class to train and test labels
# unknown_mask = np.all(train_label == 0, axis=1)  # Find rows with all zeros
# train_label = np.hstack([train_label, np.zeros((train_label.shape[0], 1))])  # Add column
# train_label[unknown_mask, -1] = 1  # Set 7th column to 1 for unknown samples
#
# unknown_mask = np.all(test_label == 0, axis=1)  # Find rows with all zeros
# test_label = np.hstack([test_label, np.zeros((test_label.shape[0], 1))])  # Add column
# test_label[unknown_mask, -1] = 1  # Set 7th column to 1 for unknown samples

# Data normalization:
normalized_filtered_train_data = z_score_normalize(filtered_train_data)
normalized_filtered_test_data = z_score_normalize(filtered_test_data)

# create epoch
epoch_len_samples = int(Config.epoch_duration * Config.sampling_rate)
train_X, train_Y = create_epochs(normalized_filtered_train_data, train_label, epoch_len_samples)
test_X, test_Y = create_epochs(normalized_filtered_test_data, test_label, epoch_len_samples)

# Create datasets
train_dataset = EEGDataset(train_X, train_Y)
val_dataset = EEGDataset(test_X, test_Y)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
)

#  Initialize model
model = EEG_CNN_LSTM().to(Config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

#  Training loop
for epoch in range(1, 10):
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(Config.device), y.to(Config.device)
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
        for X, y in test_loader:
            X, y = X.to(Config.device), y.to(Config.device)

            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.3).float()
            correct += (preds == y).sum().item()
            total += y.numel()

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss / len(test_loader):.4f} | "
              f"Accuracy: {correct/total:.4f}")


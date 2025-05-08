import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Loading the Data
file_path = 'C:/Users/Windows 10/Downloads/synthetic_iot-2.csv'
df = pd.read_csv(file_path, parse_dates=['time'])

df['time'] = pd.to_datetime(df['time'])

df.shape

# df['target'] = df.apply(lambda row: f"{row['is_anomaly']}_{row['anomaly_type']}"
#                          if row['is_anomaly'] == 1
#                         else "normal", axis=1)
# df['target'] = df['target'].str.replace('True_', '', regex=False)
# df = df.drop(labels=['is_anomaly', 'anomaly_type'], axis=1)

# Removing Outliers
def remove_outliers_iqr(df, columns, whisker_width=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - whisker_width * IQR
        upper_bound = Q3 + whisker_width * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

outlier_cols = ['temperature', 'pressure', 'vibration']
df_no_outliers = remove_outliers_iqr(df.copy(), outlier_cols)
df = df_no_outliers.copy()

# Missing Values Imputation
df['anomaly_type'].fillna('No Anomaly', inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder
df['is_anomaly'] = label_encoder().fit_transform(df['is_anomaly'])

label_encoder = LabelEncoder()
encodered_labels = label_encoder.fit_transform(df['anomaly_type'])
label_encoder.classes_
df['label'] = encodered_labels.copy()
# CLASS_NORMAL = 1

# class_name = ['normal', 'unauthorized_access', 'inventory_mismatch',
#        'inefficient_routing', 'high_humidity', 'gas_leak',
#        'temperature_spike', 'equipment_failure', 'vibration_shock']

# Feature Engineering
rows = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
  datetime_object = pd.to_datetime(index)
  row_data = dict(
      month=datetime_object.month,
      week_of_month=datetime_object.day,
      day_of_week=datetime_object.dayofweek,
      hour=datetime_object.hour,
      minute=datetime_object.minute,
      temperature=row['temperature'],
      humidity=row['humidity'],
      vibration=row['vibration'],
      pressure=row['pressure'],
      gas_levels=row['gas_levels'],
      conveyor_speed=row['conveyor_speed'],
      forklift_movement=row['forklift_movement'],
      stock_level_changes=row['stock_level_changes'],
      order_processing_time=row['order_processing_time'],
  )
  rows.append(row_data)

features_df = pd.DataFrame(rows)

features_df.shape

# Separating features and target
features_df = features_df.reset_index(drop=True)

features_df['is_anomaly'] = df['is_anomaly'].reset_index(drop=True) 

normal_features_df = features_df[features_df['is_anomaly'] == 0].drop(columns=['is_anomaly'])
anomaly_features_df = features_df[features_df['is_anomaly'] == 1].drop(columns=['is_anomaly'])


print(normal_features_df.shape)
print(anomaly_features_df.shape)

# Spliting the Dataset
train_df, val_df = \
train_test_split(normal_features_df, test_size=0.2, shuffle=False, random_state=42)
val_df, test_df = \
train_test_split(val_df, test_size=0.5, shuffle=False, random_state=42)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

# Data Normalization 
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_df)

train_df_scaled = pd.DataFrame(
    scaler.transform(train_df),
    index=train_df.index,
    columns=train_df.columns
)
val_df_scaled = pd.DataFrame(
    scaler.transform(val_df),
    index=val_df.index,
    columns=val_df.columns
)

test_df_scaled = pd.DataFrame(
    scaler.transform(test_df),
    index=test_df.index,
    columns=test_df.columns
)
anomaly_scaled_df = pd.DataFrame(
    scaler.transform(anomaly_features_df),
    index=anomaly_features_df.index,
    columns=anomaly_features_df.columns
)
# Sequence Creation
def create_sliding_sequences(data, seq_len):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        sequences.append(seq)

    return np.array(sequences, dtype=np.float32)

# PyTorch Dataset 
seq_len = 60
batch_size = 64

X_train_seq = torch.tensor(create_sliding_sequences(train_df_scaled, seq_len), dtype=torch.float32)
X_val_seq   = torch.tensor(create_sliding_sequences(val_df_scaled,   seq_len), dtype=torch.float32)
X_test_seq  = torch.tensor(create_sliding_sequences(test_df_scaled,  seq_len), dtype=torch.float32)
anomaly_seq = torch.tensor(create_sliding_sequences(anomaly_scaled_df, seq_len), dtype=torch.float32)

# Wrap in TensorDataset
train_dataset        = TensorDataset(X_train_seq)
val_dataset          = TensorDataset(X_val_seq)
test_normal_dataset  = TensorDataset(X_test_seq)
test_anomaly_dataset = TensorDataset(anomaly_seq)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_normal_loader  = DataLoader(test_normal_dataset, batch_size=batch_size)
test_anomaly_loader = DataLoader(test_anomaly_dataset, batch_size=batch_size)

print(X_train_seq.shape)
print(X_val_seq.shape)
print(X_test_seq.shape)
print(anomaly_seq.shape)

seq_len = X_train_seq.shape[1]
n_features = X_train_seq.shape[2]


 
# Building the LSTM-AE

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=256, dropout_rate=0.4):
        super().__init__()
        self.rnn1 = nn.LSTM(n_features, 2 * embedding_dim, batch_first=True, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(2 * embedding_dim)
        self.rnn2 = nn.LSTM(2 * embedding_dim, embedding_dim, batch_first=True, dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.norm1(x)
        x, (hidden, _) = self.rnn2(x)
        return self.norm2(hidden[-1])

# Updated Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=256, dropout_rate=0.4, output_dim=14):
        super().__init__()
        self.seq_len = seq_len
        self.rnn1 = nn.LSTM(input_dim, input_dim, batch_first=True, dropout=dropout_rate)
        self.rnn2 = nn.LSTM(input_dim, 2 * input_dim, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(2 * input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.linear(x)

# Full Model
device = torch.device("cuda")
if torch.cuda.is_available():
  print("Using GPU for model training")

class LSTMAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=256, dropout_rate=0.4):
        super().__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim, dropout_rate)
        self.decoder = Decoder(seq_len, input_dim=embedding_dim, output_dim=n_features, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

model = LSTMAE(seq_len, n_features, embedding_dim=128).to(device)

# Training Loop
def train_model(model, train_loader, val_loader, n_epochs=150, max_grad_norm=1.0):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss
    scaler = GradScaler()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    early_stop_counter = 0
    patience = 10

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch[0].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(batch)
                loss = criterion(outputs, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Best model saved at Epoch {epoch} | Val Loss: {val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return model.eval(), history



# Training
model, history = train_model(model, train_loader, val_loader, n_epochs=100)

def plot_loss(history):
   plt.figure(figsize=(10, 6))
   plt.plot(history['train_loss'], label='Train Loss', color='blue')
   plt.plot(history['val_loss'], label='Validation Loss', color='red')
   plt.xlabel('Epochs', fontsize=12)
   plt.ylabel('Loss', fontsize=12)
   plt.title('Training vs Validation Loss', fontsize=14)
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   plt.show()
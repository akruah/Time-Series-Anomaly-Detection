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

device = torch.device("cuda")
if torch.cuda.is_available():
  print("Using GPU for model training")

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=128, dropout_rate=0.2):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=2,
      batch_first=True,
      dropout=dropout_rate
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=2,
      batch_first=True,
      dropout=dropout_rate
    )

  def forward(self, x):
    x, _ = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n[-1]

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=128, dropout_rate=0.2, output_dim=n_features):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.output_dim = 2 * input_dim, output_dim
    dropout_rate = dropout_rate

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=2,
      dropout=dropout_rate,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=2,
      dropout=dropout_rate,
      batch_first=True
    )
    self.dense_layers = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(self, x):
    x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
    x, _ = self.rnn1(x)
    x, _ = self.rnn2(x)

    return self.dense_layers(x)

class LSTMAE(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=128, dropout_rate=0.2):
    super(LSTMAE, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim



    self.encoder = Encoder(seq_len, n_features,
                           dropout_rate=dropout_rate,
                           embedding_dim=embedding_dim).to(device)
    self.decoder = Decoder(seq_len, output_dim=n_features,
                           dropout_rate=dropout_rate,
                           input_dim=embedding_dim).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

model = LSTMAE(seq_len, n_features, embedding_dim=128).to(device)

# Training Loop
def train_model(model, train_loader, val_loader, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=5, factor=0.5,
                                                     verbose=True)
    criterion = nn.L1Loss(reduction='mean')

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    checkpoint_path = 'best_model.path'
    patience_counter = 0
    patience_limit = 10

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print(f"Epoch {epoch}: New best model saved with Val Loss: {val_loss:.4f}")
        else:
           patience_counter += 1
           if patience_counter >= patience_limit:
              print(f"Early stopping at epoch {epoch}. val loss did not improve for {patience_limit} epochs.")

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load(checkpoint_path))
    return model.eval(), history


# Training
model, history = train_model(model, train_loader, val_loader, n_epochs=100)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.encoder(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return output

def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch.permute(1, 0, 2))
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch.permute(1, 0, 2))
                loss = criterion(output.squeeze(), y_batch)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch.permute(1, 0, 2))
            predictions.extend(output.squeeze().tolist())
            actuals.extend(y_batch.tolist())

    return predictions, actuals

def prepare_test_data(X_test):
    return torch.tensor(X_test, dtype=torch.float32)

# Dự đoán giá cổ phiếu
def predict_stock_price(model, X_test, scaler_price):
    model.eval()
    with torch.no_grad():
        X_test_tensor = prepare_test_data(X_test)
        predictions = model(X_test_tensor.permute(1, 0, 2))
        predicted_stock_price = predictions.squeeze().numpy()

    # Thêm các cột giá trị giả (zeros) để khớp với hình dạng mà scaler đã được huấn luyện
    if np.array_equal(X_test, np.load('X_test_techtransformer.npy')):
        predicted_stock_price_extended = np.concatenate((predicted_stock_price.reshape(-1, 1), np.zeros((predicted_stock_price.shape[0], 4))), axis=1)
    else:
        predicted_stock_price_extended = np.concatenate((predicted_stock_price.reshape(-1, 1), np.zeros((predicted_stock_price.shape[0], 3))), axis=1)
    # Đảo ngược quá trình chuẩn hóa
    predicted_stock_price_scaled_back = scaler_price.inverse_transform(predicted_stock_price_extended)

    # Loại bỏ các cột giá trị giả
    predicted_stock_price = predicted_stock_price_scaled_back[:, 0]

    return predicted_stock_price
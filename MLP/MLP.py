import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline
import numpy as np
import os
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(200 * 4, 100)  
        self.fc2 = nn.Linear(100, 200)      
        self.fc3 = nn.Linear(200, 100)      
        self.fc4 = nn.Linear(100, 200 * 2)  
        self.relu = nn.ReLU()               

    def forward(self, x):
        x = x.view(-1, 200 * 4)  
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 200, 2)   
        return x

model = MLP()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def list_files(diretorio):
    arquivos = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            arquivos.append(os.path.join(root, file))
    return arquivos

def preprocess_data(X, Y, input_size, output_size):
    X_rec = np.empty((0, 4), int)
    for i in X:
        sample = np.linspace(0.0, 1.0, i.shape[0])

        X_x = CubicSpline(sample, i[:, 0])
        X_y = CubicSpline(sample, i[:, 1])    
        X_r = CubicSpline(sample, i[:, 2])
        X_l = CubicSpline(sample, i[:, 3])

        sample_reduced = np.linspace(0.0, 1.0, input_size)

        X_rec = np.vstack((X_rec, np.array([[X_x(i), X_y(i), X_r(i), X_l(i)] for i in sample_reduced])))

    X_rec = np.reshape(X_rec, (X.shape[0], input_size, 4))

    Y_rec = np.empty((0, 2), int)
    for i in Y:
        sample = np.linspace(0.0, 1.0, i.shape[0])

        Y_x = CubicSpline(sample, i[:, 0])
        Y_y = CubicSpline(sample, i[:, 1])

        sample_reduced = np.linspace(0.0, 1.0, output_size)

        Y_rec = np.vstack((Y_rec, np.array([[Y_x(i), Y_y(i)] for i in sample_reduced])))

    Y_rec = np.reshape(Y_rec, (Y.shape[0], output_size, 2))

    return X_rec, Y_rec

train_X = np.array([np.loadtxt(i, delimiter = ',') for i in list_files('./dataset/train/tracks')])
train_Y = np.array([np.loadtxt(i, delimiter = ',') for i in list_files('./dataset/train/racelines')])

test_X = np.array([np.loadtxt(i, delimiter = ',') for i in list_files('./dataset/test/tracks')])
test_Y = np.array([np.loadtxt(i, delimiter = ',') for i in list_files('./dataset/test/racelines')])

train_X, train_Y = preprocess_data(train_X, train_Y, 200, 200)
test_X, test_Y = preprocess_data(test_X, test_Y, 200, 200)

# Converta os dados para tensores PyTorch
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y = torch.tensor(test_Y, dtype=torch.float32)

def euclidean_distance(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=2)).mean()

def train_model(model, criterion, optimizer, train_data, train_targets, test_data, test_targets, epochs=100):
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        # Treinamento
        optimizer.zero_grad()           
        outputs = model(train_data)           
        loss = criterion(outputs, train_targets)  
        loss.backward()                 
        optimizer.step()                

        # Validação
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            val_loss = euclidean_distance(test_outputs, test_targets)

        # Armazenar os valores para plotagem
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Euclidean Distance: {val_loss.item()}')

    return train_losses, val_losses

train_losses, val_losses = train_model(model, criterion, optimizer, train_X, train_Y, test_X, test_Y, epochs=300)

# Plotagem dos gráficos de Loss e Validation Euclidean Distance por época em gráficos separados

# Gráfico de Training Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de Validation Euclidean Distance
plt.figure(figsize=(10, 5))
plt.plot(val_losses, label='Validation Euclidean Distance')
plt.xlabel('Epochs')
plt.ylabel('Euclidean Distance')
plt.title('Validation Euclidean Distance per Epoch')
plt.legend()
plt.grid(True)
plt.show()

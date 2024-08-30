import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline
import numpy as np
import os
import matplotlib.pyplot as plt
import math

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

def plot_track(track, line_real, line_predicted, num_points=1000):
    def calcular_normais(curva):
        normais = []
        for i in range(len(curva) - 1):
            tangente = np.array(curva[i+1]) - np.array(curva[i])
            normal = np.array([-tangente[1], tangente[0]])
            normal = normal / np.linalg.norm(normal)
            normais.append(normal)
        
        return np.array(normais)
    
    # Interpolação para suavizar as curvas
    sample = np.linspace(0.0, 1.0, track.shape[0])
    smooth_sample = np.linspace(0.0, 1.0, num_points)
    
    cs_x = CubicSpline(sample, track[:, 0])
    cs_y = CubicSpline(sample, track[:, 1])
    cs_r = CubicSpline(sample, track[:, 2])
    cs_l = CubicSpline(sample, track[:, 3])
    
    smooth_track = np.array([[cs_x(i), cs_y(i), cs_r(i), cs_l(i)] for i in smooth_sample])
    
    # Recalcular as bordas com a curva suavizada
    normal = calcular_normais(smooth_track[:, 0:2])
    bound1 = smooth_track[:-1, 0:2] + normal * np.expand_dims(smooth_track[:-1, 2], axis=1)
    bound2 = smooth_track[:-1, 0:2] - normal * np.expand_dims(smooth_track[:-1, 3], axis=1)
    
    # Interpolação para a linha real e a linha prevista
    cs_real_x = CubicSpline(sample, line_real[:, 0])
    cs_real_y = CubicSpline(sample, line_real[:, 1])
    cs_pred_x = CubicSpline(sample, line_predicted[:, 0])
    cs_pred_y = CubicSpline(sample, line_predicted[:, 1])
    
    smooth_real = np.array([[cs_real_x(i), cs_real_y(i)] for i in smooth_sample])
    smooth_pred = np.array([[cs_pred_x(i), cs_pred_y(i)] for i in smooth_sample])
    
    # Plotagem
    plt.plot(smooth_track[:, 0], smooth_track[:, 1], label='Pista', color='gray', linestyle='--', linewidth=0.7)
    plt.plot(bound1[:, 0], bound1[:, 1], color='gray', linewidth=0.7)
    plt.plot(bound2[:, 0], bound2[:, 1], color='gray', linewidth=0.7)
    
    plt.plot(smooth_real[:, 0], smooth_real[:, 1], label='Linha Real', color='blue', linestyle='-', linewidth=0.7)
    plt.plot(smooth_pred[:, 0], smooth_pred[:, 1], label='Linha Prevista', color='red', linestyle='-', linewidth=1)

    plt.grid(True)

def train_model(model, criterion, optimizer, train_data, train_targets, test_data, test_targets, epochs=500, plot_interval=100):
    train_losses = []
    val_losses = []
    plot_data = []  # Armazena os dados para plotar no final
    
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

        # Armazenar as previsões para plotagem
        if (epoch + 1) % plot_interval == 0:
            plot_data.append((epoch + 1, test_X[0].numpy(), test_Y[0].numpy(), test_outputs[0].detach().numpy()))

    return train_losses, val_losses, plot_data

def plot_track_subplot(plot_data):
    num_plots = len(plot_data)
    rows = cols = math.ceil(math.sqrt(num_plots))  # Arranjo mais próximo de um quadrado
    
    plt.figure(figsize=(10, 10))
    
    for i, (epoch, track, line_real, line_predicted) in enumerate(plot_data):
        plt.subplot(rows, cols, i + 1)
        plot_track(track, line_real, line_predicted)
        plt.title(f'Epoch {epoch}')
    
    plt.tight_layout()
    plt.show()

train_losses, val_losses, plot_data = train_model(model, criterion, optimizer, train_X, train_Y, test_X, test_Y, epochs=900, plot_interval=100)
plot_track_subplot(plot_data)
plot_track_subplot([plot_data[-1]])

for i in range(len(test_X)):
    y = model(test_X[i])
    os.mkdir(f'./comparison_data/track{i}')
    np.savetxt(f'comparison_data/track{i}/comparison_expected.csv', test_Y[i].numpy())
    np.savetxt(f'comparison_data/track{i}/comparison_obtained.csv', y[0].detach().numpy())

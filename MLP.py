import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(200 * 2, 100)  # Entrada de 200 vetores de 2 dimensões
        self.fc2 = nn.Linear(100, 200)      # Primeira camada oculta com 100 neurônios
        self.fc3 = nn.Linear(200, 100)      # Segunda camada oculta com 200 neurônios
        self.fc4 = nn.Linear(100, 100)      # Terceira camada oculta com 100 neurônios
        self.output = nn.Linear(100, 100)   # Camada de saída com 100 neurônios
        
        self.relu = nn.ReLU()  # Função de ativação ReLU

    def forward(self, x):
        x = self.flatten(x)     # Achatar a entrada
        x = self.relu(self.fc1(x))  # Primeira camada
        x = self.relu(self.fc2(x))  # Segunda camada
        x = self.relu(self.fc3(x))  # Terceira camada
        x = self.relu(self.fc4(x))  # Quarta camada
        x = self.output(x)          # Camada de saída
        return x

# Criando o modelo
model = MLP()

# Definindo o otimizador e a função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Usando MSELoss como exemplo

# Exemplo de uso do modelo
# Suponha que você tenha um lote de entrada X de tamanho (batch_size, 200, 2)
# E um lote de rótulos/target Y de tamanho (batch_size, 100)

# Exemplo de dados fictícios
X = torch.randn(10, 200, 2)  # Lote de 10 amostras
Y = torch.randn(10, 100)     # Rótulos correspondentes

# Passando os dados pelo modelo
output = model(X)

# Calculando a perda
loss = criterion(output, Y)

# Atualizando os pesos
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')

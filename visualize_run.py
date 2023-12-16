import matplotlib.pyplot as plt
import matplotlib
from genetic import import_save_data

matplotlib.use('Qt5Agg')
data = import_save_data

print(data)



# Dados de exemplo
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Criar o gráfico de dispersão
plt.scatter(x, y, color='blue', label='Pontos de dados')

# Adicionar rótulos e título
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Exemplo de Scatter Plot')

# Adicionar legenda
plt.legend()

# Mostrar o gráfico
plt.show()

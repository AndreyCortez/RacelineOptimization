import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from genetic import import_save_data

matplotlib.use('Qt5Agg')
data = import_save_data('Baseline')['RUN_HISTORY']


# print(data)
arr_1 = np.array([i[0] for i in data])
arr_2 = np.array([i[1] for i in data])

print(arr_1)
print(arr_2)

data = np.column_stack((arr_1, arr_2))
print(data)



# Dados de exemplo
x = data[:,0]
y = data[:,1:]

print(y)

# Criar o gráfico de dispersão
for y_a in [y.T[0]]:
    plt.scatter(x, y_a, color='blue')

# plt.xscale('log')

# Adicionar rótulos e título
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Exemplo de Scatter Plot')

# Adicionar legenda
plt.legend()

# Mostrar o gráfico
plt.show()

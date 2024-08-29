
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# Carregar dados da pista
base = np.loadtxt('./dataset/tracks/Austin.csv', delimiter = ',')

center_line = base[:,0:2]
initial_samples = center_line.shape[0]
sample = np.linspace(0.0, 1.0, center_line.shape[0])

cs_x = CubicSpline(sample, center_line[:, 0])
cs_y = CubicSpline(sample, center_line[:, 1])

# Configurar o gráfico
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.ravel()  # Flatten the 2D array to 1D for easy indexing

for downgrade_level in range(1, 5):
    downgrade_coefficiente = 10 * (downgrade_level)
    resample = np.linspace(0.0, 1.0, int(initial_samples / downgrade_coefficiente))
    
    # Gerar nova curva com menos pontos
    new_curve = np.array([[cs_x(i), cs_y(i)] for i in resample])

    # Reconstruir a curva com o novo conjunto de pontos
    nc_x = CubicSpline(resample, new_curve[:, 0])
    nc_y = CubicSpline(resample, new_curve[:, 1])

    reconstructed = np.array([[nc_x(i), nc_y(i)] for i in sample])

    # Plotar as curvas original (mc) e reconstruída (reconstructed)
    axs[downgrade_level - 1].plot(center_line[:, 0], center_line[:, 1], label='Original', color='blue')
    axs[downgrade_level - 1].plot(reconstructed[:, 0], reconstructed[:, 1], label='Reconstruída', color='red', linestyle='--')
    axs[downgrade_level - 1].set_title(f'Quantidade de Samples do Original: {round(100 / downgrade_coefficiente, 2)} %')
    axs[downgrade_level - 1].legend()
    axs[downgrade_level - 1].grid(True)

# Ajustar layout e exibir o gráfico
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

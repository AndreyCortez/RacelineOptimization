
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

def calcular_normais(curva):
    normais = []
    for i in range(len(curva) - 1):
        tangente = np.array(curva[i+1]) - np.array(curva[i])
        normal = np.array([-tangente[1], tangente[0]])
        normal = normal / np.linalg.norm(normal)
        normais.append(normal)
    
    return np.array(normais)

def plot_track(ax, track, label, color):
    normal = calcular_normais(track[:,0:2])
    bound1 = track[:-1, 0:2] + normal * np.expand_dims(track[:-1, 2], axis=1)
    bound2 = track[:-1, 0:2] - normal * np.expand_dims(track[:-1, 3], axis=1)

    ax.plot(track[:, 0], track[:, 1], label=label, color=color, linestyle='--', linewidth=0.7)
    ax.plot(bound1[:, 0], bound1[:, 1], color=color, linewidth=0.7)
    ax.plot(bound2[:, 0], bound2[:, 1], color=color, linewidth=0.7)


# Carregar dados da pista
base = np.loadtxt('./dataset/train/tracks/SaoPaulo.csv', delimiter = ',')
base[:,2:4] *= 2.5

center_line = base[:,0:2]
initial_samples = base.shape[0]
sample = np.linspace(0.0, 1.0, base.shape[0])

cs_x = CubicSpline(sample, base[:, 0])
cs_y = CubicSpline(sample, base[:, 1])
cs_r = CubicSpline(sample, base[:, 2])
cs_l = CubicSpline(sample, base[:, 3])

# Configurar o gráfico
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.ravel()  # Flatten the 2D array to 1D for easy indexing

for downgrade_level in range(1, 5):
    downgrade_coefficiente = 10 * (downgrade_level)
    resample = np.linspace(0.0, 1.0, int(initial_samples / downgrade_coefficiente))
    
    # Gerar nova curva com menos pontos
    new_curve = np.array([[cs_x(i), cs_y(i), cs_r(i), cs_l(i)] for i in resample])

    # Reconstruir a curva com o novo conjunto de pontos
    nc_x = CubicSpline(resample, new_curve[:, 0])
    nc_y = CubicSpline(resample, new_curve[:, 1])
    nc_r = CubicSpline(resample, new_curve[:, 2])
    nc_l = CubicSpline(resample, new_curve[:, 3])

    reconstructed = np.array([[nc_x(i), nc_y(i), nc_r(i), nc_l(i)] for i in sample])

    # Plotar as curvas original (mc) e reconstruída (reconstructed)
    # axs[downgrade_level - 1].plot(center_line[:, 0], center_line[:, 1], label='Original', color='blue')
    plot_track(axs[downgrade_level - 1], base, 'Original', 'Blue')
    plot_track(axs[downgrade_level - 1], reconstructed, 'Reconstruida', 'Red')
    # axs[downgrade_level - 1].plot(reconstructed[:, 0], reconstructed[:, 1], label='Reconstruída', color='red', linestyle='--')
    axs[downgrade_level - 1].set_title(f'Quantidade de Samples do Original: {round(100 / downgrade_coefficiente, 2)} %')
    axs[downgrade_level - 1].legend()
    axs[downgrade_level - 1].grid(True)

# Ajustar layout e exibir o gráfico
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

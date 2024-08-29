import wrapper_funcs
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Supondo que as funções de wrapper_funcs já foram importadas corretamente
track_name = 'Baseline'

# Carregar dados da pista
base = wrapper_funcs.get_track_data('input/tracks/' + track_name + '.csv')
base[:, :2] *= 2
base = wrapper_funcs.get_essential_curves(base)
mask = wrapper_funcs.get_intersection_interpolation_mask(base['sp'], base['min_curv'])

mc = base['min_curv']
initial_samples = mc.shape[0]
sample = np.linspace(0.0, 1.0, mc.shape[0])

cs_x = CubicSpline(sample, mc[:, 0])
cs_y = CubicSpline(sample, mc[:, 1])

# Configurar o gráfico
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.ravel()  # Flatten the 2D array to 1D for easy indexing

for downgrade_level in range(4):
    downgrade_coefficiente = 4**downgrade_level
    resample = np.linspace(0.0, 1.0, int(initial_samples / downgrade_coefficiente))
    
    # Gerar nova curva com menos pontos
    new_curve = np.array([[cs_x(i), cs_y(i)] for i in resample])

    # Reconstruir a curva com o novo conjunto de pontos
    nc_x = CubicSpline(resample, new_curve[:, 0])
    nc_y = CubicSpline(resample, new_curve[:, 1])

    reconstructed = np.array([[nc_x(i), nc_y(i)] for i in sample])

    # Plotar as curvas original (mc) e reconstruída (reconstructed)
    axs[downgrade_level].plot(mc[:, 0], mc[:, 1], label='Original', color='blue')
    axs[downgrade_level].plot(reconstructed[:, 0], reconstructed[:, 1], label='Reconstruída', color='red', linestyle='--')
    axs[downgrade_level].set_title(f'Quantidade de Samples do Original: {100 / downgrade_coefficiente} %')
    axs[downgrade_level].legend()
    axs[downgrade_level].grid(True)

# Ajustar layout e exibir o gráfico
plt.tight_layout()
plt.show()

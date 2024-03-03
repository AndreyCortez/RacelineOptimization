import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from genetic import import_save_data, gerar_raceline
import wrapper_funcs


track_name = 'Baseline'


base = wrapper_funcs.get_track_data('input/tracks/' + track_name + '.csv')
base[:,:2] *= 2
# base[:,2:4] *= 2
base = wrapper_funcs.get_essential_curves(base)
mask = wrapper_funcs.get_intersection_mask(base['sp'], base['min_curv'])

wrapper_funcs.plot_track(base, [base['sp']], 'Curva de Menor comprimento', True)
wrapper_funcs.plot_track(base, [base['min_curv']], 'Curva de menor curvatura', True)

# ---------------- Salva dados para serem usados no vídeo de explicação -----------------------------

# save = np.column_stack([base['center'][:, 0:2], base['left_border'], base['right_border'], base['sp'], base['min_curv'], mask])
# np.savetxt('Baseline.txt', save)

# ---------------------------------------------------------------------------------------------------


data = import_save_data(track_name)['NEW_BESTS']

sp, mc  = base['sp'], base['min_curv']

initial_curve, final_curve = wrapper_funcs.gerar_raceline(sp, mc, data[0][1]), wrapper_funcs.gerar_raceline(sp, mc, data[-1][1])
base = base['center'][:, 0:2]

arr_1 = np.array([i[0] for i in data])
arr_2 = np.array([i[2] for i in data])


data = np.column_stack((arr_1, arr_2))


# Dados de exemplo
x = data[:,0]
y = data[:,1:]

fig, axs = plt.subplots(1, 2, figsize=(8, 6))

axs[0].scatter(x, y, color='blue')

axs[0].set_xlabel('Batch')
axs[0].set_ylabel('Tempo da Volta')
axs[0].set_title('Evolução do Tempo de Volta')

x, y = zip(*(base))
be, bd = wrapper_funcs.track_borders(base, 5)    
x_1, y_1 = zip(*(be))
x_2, y_2 = zip(*(bd))

axs[1].plot(x, y, marker='', color = "gray", linestyle = '--', linewidth = 1)
axs[1].plot(x_1, y_1, color = "gray", linewidth = 1)
axs[1].plot(x_2, y_2, color = "gray", linewidth = 1)

x, y = zip(*initial_curve)
axs[1].plot(x, y, label = 'Curva Inicial')

x, y = zip(*final_curve)
axs[1].plot(x, y, label = 'Curva Final')

axs[1].axis('equal')

axs[1].set_xlabel('Eixo X')
axs[1].set_ylabel('Eixo Y')

axs[1].set_title('Evolução da linha de corrida')

axs[1].legend()

plt.tight_layout()
plt.show()

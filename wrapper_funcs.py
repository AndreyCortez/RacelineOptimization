import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


import laptimesim
from race_config import *

from helper_functions.opt_min_curvature import opt_min_curv
from helper_functions.opt_shortest_path import opt_shortest_path

import os
import sys
sys.path.append(os.path.dirname(__file__))

from helper_functions.calc_splines import calc_splines


# Por algum motivo esse módulo de laptime sim só funciona com essa bagaça
# Não sei o pq
import matplotlib
matplotlib.use('Qt5Agg')

def get_track_data(trackfile):
    csv_data_temp = np.loadtxt(os.path.dirname(__file__) + '/' + trackfile,
                                comments='#', delimiter=',')
    return csv_data_temp

# TODO: Fazer isso aqui
def filter_track(track_data):
    lista_original = track_data[:,2]
    novos_pontos = np.linspace(np.min(lista_original), np.max(lista_original), len(lista_original) * 2)
    lista_interpolada = np.interp(novos_pontos, range(len(lista_original)), lista_original)
    print(lista_original)
    print(lista_interpolada)


    pass

def get_essential_curves(track_data, plot = False):
    # --- PARAMETERS ---
    CLOSED = True

    # get coords and track widths out of array
    reftrack = track_data[:, 0:4]
    psi_s = 0.0
    psi_e = 2.0

    # --- CALCULATE MIN CURV ---
    if CLOSED:
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])))
    else:
        reftrack = reftrack[200:600, :]
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=reftrack[:, 0:2],
                                                            psi_s=psi_s,
                                                            psi_e=psi_e)

        # extend norm-vec to same size of ref track (quick fix for testing only)
        normvec_norm = np.vstack((normvec_norm[0, :], normvec_norm))

    aplha_sp = opt_shortest_path(reftrack=reftrack,
                                                    normvectors=normvec_norm,
                                                    w_veh=2.0)

    alpha_mincurv, curv_error_max = opt_min_curv(reftrack=reftrack,
                                                    normvectors=normvec_norm,
                                                    A=M,
                                                    kappa_bound=0.4,
                                                    w_veh=2.0,
                                                    closed=CLOSED,
                                                    psi_s=psi_s,
                                                    psi_e=psi_e)

    # --- PLOT RESULTS ---
    curv_result = reftrack[:, 0:2] + normvec_norm * np.expand_dims(alpha_mincurv, axis=1)
    sp_result = reftrack[:, 0:2] + normvec_norm * np.expand_dims(aplha_sp, axis=1)
    bound1 = reftrack[:, 0:2] - normvec_norm * np.expand_dims(reftrack[:, 2], axis=1)
    bound2 = reftrack[:, 0:2] + normvec_norm * np.expand_dims(reftrack[:, 3], axis=1)
    center = reftrack

    if plot:
        plt.plot(reftrack[:, 0], reftrack[:, 1], ":")
        plt.plot(curv_result[:, 0], curv_result[:, 1], label = 'Minimun Curvature')
        plt.plot(sp_result[:, 0], sp_result[:, 1], label = 'Shortest Path')
        plt.plot(bound1[:, 0], bound1[:, 1], 'k')
        plt.plot(bound2[:, 0], bound2[:, 1], 'k')
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    return {'min_curv' : curv_result,
            'sp' : sp_result,
            'left_border' : bound1,
            'right_border' : bound2,
            'center' : center
            }


def gerar_raceline(curva_1, curva_2, alpha):
    mask = get_intersection_mask(curva_1, curva_2)
    alpha = np.array(alpha)[mask]
    alpha = np.column_stack((alpha, alpha))
    vetor_interp = (1 - alpha) * curva_1 + alpha * curva_2
    return vetor_interp

# NOTE: Talvez fazer a primeira mascara e a mascara final serem o mesmo número ajude a acabar com os erros
# dos infs
def get_intersection_mask(path1, path2):
    intersection_indices = np.where(np.linalg.norm(path1 - path2, axis=1) < 0.01)
    aux = intersection_indices[0]
    intersection_indices = np.where(np.linalg.norm(path1[intersection_indices] - np.roll(path1[intersection_indices], 1, axis=0), axis=1) > 2)

    cnt = 0
    mask = []
    for i in range(len(path1)):
        if cnt >= len(intersection_indices[0]):
            mask.append(cnt-1)
            continue
        mask.append(cnt)
        if i > aux[intersection_indices][cnt]:
            cnt += 1

    return mask


def track_borders(base):
    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((base['center'][:, 0:2], base['center'][0, 0:2])))

    bound1 = base['center'][:, 0:2] - normvec_norm * np.expand_dims(base['center'][:, 2], axis=1)
    bound2 = base['center'][:, 0:2] + normvec_norm * np.expand_dims(base['center'][:, 3], axis=1)

    return bound1, bound2

def plot_track(base, racelines = [], title = 'Mapa da Pista', flip_axis = False):

    x, y = base['center'][:,0], base['center'][:,1]
    be, bd = track_borders(base)
    x_1, y_1 = zip(*(be))
    x_2, y_2 = zip(*(bd))
    
    #TODO: trocar o linewidths por track.width
    if not flip_axis:
        plt.plot(x, y, marker='', color = "gray", linestyle = '--', linewidth = 1)
        plt.plot(x_1, y_1, color = "gray", linewidth = 1)
        plt.plot(x_2, y_2, color = "gray", linewidth = 1)
    else:
        plt.plot(y, x, marker='', color = "gray", linestyle = '--', linewidth = 1)
        plt.plot(y_1, x_1, color = "gray", linewidth = 1)
        plt.plot(y_2, x_2, color = "gray", linewidth = 1)
    

    for i in racelines:
        x, y = zip(*i)
        if not flip_axis:
            plt.plot(x, y, marker='')
        else: 
            plt.plot(y, x, marker='')

    plt.axis('equal')

    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')

    plt.title(title)

    plt.show()

def simulate_raceline(raceline):
    global track_opts
    global driver_opts
    global solver_opts
    
    global track_opts
    global driver_opts
    global solver_opts

    repo_path = os.path.dirname(os.path.abspath(__file__))

    parfilepath = os.path.join(repo_path, "input", "track_pars.ini")

    if driver_opts["vel_lim_glob"] is not None:
        vel_lim_glob = driver_opts["vel_lim_glob"]
    elif solver_opts["series"] == "FE":
        vel_lim_glob = np.inf
    else:
        vel_lim_glob = np.inf

    track = laptimesim.src.track.Track(pars_track=track_opts,
                                       parfilepath=parfilepath,
                                       track = raceline,
                                       vel_lim_glob=vel_lim_glob,
                                       yellow_s1=driver_opts["yellow_s1"],
                                       yellow_s2=driver_opts["yellow_s2"],
                                       yellow_s3=driver_opts["yellow_s3"])

    parfilepath = os.path.join(repo_path, "input", "vehicles", solver_opts["vehicle"])

    if solver_opts["series"] == "F1":
        car = laptimesim.src.car_hybrid.CarHybrid(parfilepath=parfilepath)
    elif solver_opts["series"] == "FE":
        car = laptimesim.src.car_electric.CarElectric(parfilepath=parfilepath)
    else:
        raise IOError("Unknown racing series!")

    driver = laptimesim.src.driver.Driver(carobj=car,
                                          pars_driver=driver_opts,
                                          trackobj=track,
                                          stepsize=track.stepsize)

    lap = laptimesim.src.lap.Lap(driverobj=driver,
                                 trackobj=track,
                                 pars_solver=solver_opts,
                                 debug_opts=debug_opts)

    try:
        lap.simulate_lap()
    except:
        return np.inf

    return lap.t_cl[-1]




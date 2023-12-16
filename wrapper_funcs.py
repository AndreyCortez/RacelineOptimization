import os
import numpy as np
import matplotlib.pyplot as plt

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


def get_essential_curves(trackfile, plot = False):
    # --- PARAMETERS ---
    CLOSED = True

    # --- IMPORT TRACK ---
    # load data from csv file
    csv_data_temp = np.loadtxt(os.path.dirname(__file__) + '/' + trackfile,
                                comments='#', delimiter=',')

    # get coords and track widths out of array
    reftrack = csv_data_temp[:, 0:4]
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


# TODO: implementar essa função
def get_raceline(path1, path2, alpha):
    pass


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

def encontrar_centro_circunferencia(ponto1, ponto2, ponto3):
    x1, y1 = ponto1
    x2, y2 = ponto2
    x3, y3 = ponto3
    
    denominador = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    h = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / denominador
    k = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / denominador

    return [h, k]

def orientacao_pontos(ponto1, ponto2, ponto3):
    x1, y1 = ponto1
    x2, y2 = ponto2
    x3, y3 = ponto3
    
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)


def track_borders(centro_da_pista, largura_pista):
    borda_esquerda = []
    borda_direita = []

    count = 0
    for x, y in centro_da_pista:
        x1 = centro_da_pista[count-2]
        x2 = centro_da_pista[count-1]
        x3 = centro_da_pista[count]
        circ_centro = encontrar_centro_circunferencia(x1, x2, x3)
        vetr_central = centro_da_pista[count] - circ_centro
        
        vetr_central = vetr_central / np.linalg.norm(vetr_central)
        count += 1

        
        if orientacao_pontos(x1, x2, x3) > 0:
            borda_esquerda.append([x + vetr_central[0] * largura_pista, y + vetr_central[1] * largura_pista])
            borda_direita.append([x - vetr_central[0] * largura_pista, y - vetr_central[1] * largura_pista])
        else:
            borda_esquerda.append([x - vetr_central[0] * largura_pista, y - vetr_central[1] * largura_pista])
            borda_direita.append([x + vetr_central[0] * largura_pista, y + vetr_central[1] * largura_pista])

    borda_direita = np.array(borda_direita)
    borda_esquerda = np.array(borda_esquerda)

    return borda_esquerda, borda_direita


# TODO: Ajeitar o TrackBorders pra plotar as bordas da pista msm
def plot_track(track, racelines = []):
    x, y = zip(*(track))
    be, bd = track_borders(track, 5)    
    x_1, y_1 = zip(*(be))
    x_2, y_2 = zip(*(bd))
    
    #TODO: trocar o linewidths por track.width
    plt.plot(x, y, marker='', color = "gray", linestyle = '--', linewidth = 1)
    plt.plot(x_1, y_1, color = "gray", linewidth = 1)
    plt.plot(x_2, y_2, color = "gray", linewidth = 1)

    for i in racelines:
        x, y = zip(*i)
        plt.plot(x, y, marker='')

    plt.axis('equal')

    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')

    plt.title('Mapa da pista')

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

    # set velocity limit
    if driver_opts["vel_lim_glob"] is not None:
        vel_lim_glob = driver_opts["vel_lim_glob"]
    elif solver_opts["series"] == "FE":
        vel_lim_glob = np.inf
    else:
        vel_lim_glob = np.inf

    # create instance
    track = laptimesim.src.track.Track(pars_track=track_opts,
                                       parfilepath=parfilepath,
                                       track = raceline,
                                       vel_lim_glob=vel_lim_glob,
                                       yellow_s1=driver_opts["yellow_s1"],
                                       yellow_s2=driver_opts["yellow_s2"],
                                       yellow_s3=driver_opts["yellow_s3"])

    parfilepath = os.path.join(repo_path, "input", "vehicles", solver_opts["vehicle"])

    # create instance
    if solver_opts["series"] == "F1":
        car = laptimesim.src.car_hybrid.CarHybrid(parfilepath=parfilepath)
    elif solver_opts["series"] == "FE":
        car = laptimesim.src.car_electric.CarElectric(parfilepath=parfilepath)
    else:
        raise IOError("Unknown racing series!")

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE DRIVER INSTANCE -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    driver = laptimesim.src.driver.Driver(carobj=car,
                                          pars_driver=driver_opts,
                                          trackobj=track,
                                          stepsize=track.stepsize)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE LAP INSTANCE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    lap = laptimesim.src.lap.Lap(driverobj=driver,
                                 trackobj=track,
                                 pars_solver=solver_opts,
                                 debug_opts=debug_opts)

    # ------------------------------------------------------------------------------------------------------------------
    # CALL SOLVER ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # save start time
    # t_start = time.perf_counter()

    # call simulation
    try:
        lap.simulate_lap()
    except:
        return np.inf

    return lap.t_cl[-1]




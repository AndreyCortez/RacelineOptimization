import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

import laptimesim
from race_config import *

from opt_min_curvature import opt_min_curv
from opt_shortest_path import opt_shortest_path
import os
import sys
sys.path.append(os.path.dirname(__file__))
from calc_splines import calc_splines


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



# Parametros importantes da classe track
# raceline -> retorna uma lista com os pontos (x,y) da curva da pista
# width -> retorna a largura da pista em (m) (TODO: Não implementado)

def import_center(trackname):
    repo_path = os.path.dirname(os.path.abspath(__file__))

    trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines",
                                trackname + ".csv")

    return np.loadtxt(trackfilepath, comments='#', delimiter=',')

def _set_track(raceline):
    global track_opts
    global driver_opts
    global solver_opts

    repo_path = os.path.dirname(os.path.abspath(__file__))

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")

    # set velocity limit
    if driver_opts["vel_lim_glob"] is not None:
        vel_lim_glob = driver_opts["vel_lim_glob"]
    elif solver_opts["series"] == "FE":
        vel_lim_glob = 225.0 / 3.6
    else:
        vel_lim_glob = np.inf

    # print(raceline.shape)

    # create instance
    track = laptimesim.src.track.Track(pars_track=track_opts,
                                       parfilepath=parfilepath,
                                       track = raceline,
                                       vel_lim_glob=vel_lim_glob,
                                       yellow_s1=driver_opts["yellow_s1"],
                                       yellow_s2=driver_opts["yellow_s2"],
                                       yellow_s3=driver_opts["yellow_s3"])

    return track

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
        # # Calcular vetor perpendicular à pista
        # if count == 0:
        #     vetor_perpendicular = [-1 * (y - centro_da_pista[0][1]), x - centro_da_pista[0][0]]
        # else:
        x1 = centro_da_pista[count-2]
        x2 = centro_da_pista[count-1]
        x3 = centro_da_pista[count]
        circ_centro = encontrar_centro_circunferencia(x1, x2, x3)
        vetr_central = centro_da_pista[count] - circ_centro
        
        vetr_central = vetr_central / np.linalg.norm(vetr_central)
        # print(np.linalg.norm(vetr_central))
        count += 1

        
        if orientacao_pontos(x1, x2, x3) > 0:
        # Calcular as coordenadas da borda esquerda
            borda_esquerda.append([x + vetr_central[0] * largura_pista, y + vetr_central[1] * largura_pista])
            borda_direita.append([x - vetr_central[0] * largura_pista, y - vetr_central[1] * largura_pista])
        else:
            borda_esquerda.append([x - vetr_central[0] * largura_pista, y - vetr_central[1] * largura_pista])
            borda_direita.append([x + vetr_central[0] * largura_pista, y + vetr_central[1] * largura_pista])
        # Calcular as coordenadas da borda direita
        # borda_direita.append([0, 0])
        # borda_direita.append([x + vetor_normalizado[0] * largura_pista / 2, y + vetor_normalizado[1] * largura_pista / 2])

    borda_direita = np.array(borda_direita)
    borda_esquerda = np.array(borda_esquerda)

    return borda_esquerda, borda_direita

import cvxpy as cp

def comprimento_curva(pontos):
    comprimento = 0
    for i in range(1, len(pontos)):
        diferenca = pontos[i] - pontos[i - 1]
        comprimento += np.linalg.norm(diferenca)

    return comprimento

def calculate_shortest_path(left_border, right_border):
    if (len(left_border) != len(right_border)):
        raise("As bordas tem que ter o mesmo numero de pontos")
    n = len(left_border)
    delta_x = (left_border - right_border)[:, 0]
    delta_y = (left_border - right_border)[:, 1]
    a = cp.Variable(n)
    # a_t = np.array([0.5 for _ in range(n)])

    delta_Px = []
    delta_Py = []
    S_sq = []

    for i in range(n):
        if i == n - 1:
            i = -1
    
        mat_a = cp.hstack([a[i + 1], a[i]])

        mat_1x = cp.Parameter(2)
        mat_1x.value = np.array([delta_x[i + 1], -delta_x[i]])

        delta_Px.append(cp.sum(cp.multiply(mat_1x, mat_a)) + cp.Constant(right_border[i + 1] - right_border[i]))

        mat_1y = cp.Parameter(2)
        mat_1y.value = np.array([delta_y[i + 1], -delta_y[i]])

        delta_Py.append(cp.sum(cp.multiply(mat_1y, mat_a)) + cp.Constant(right_border[i + 1] - right_border[i]))

    delta_Px = cp.hstack(delta_Px)
    delta_Py = cp.hstack(delta_Py)

    # Função objetivo
    objective = cp.Minimize(cp.sum_squares(delta_Px) + cp.sum_squares(delta_Py))

    # Restrições
    constraints = [0 <= a, a <= 1]

    # Criando o problema de otimização
    problem = cp.Problem(objective, constraints)

    # Resolvendo o problema
    problem.solve()

    res = a.value

    resultados = [(1 - f) * v1 + f * v2 for v1, v2, f in zip(right_border, left_border, res)]

    return resultados

def calculate_least_curvature_path(left_border, right_border):
    if (len(left_border) != len(right_border)):
        raise("As bordas tem que ter o mesmo numero de pontos")
    n = len(left_border)
    # a = cp.Variable((n,2))
    a = np.array([0.5 for i in range(n)])
    # print("passou")
    delta_x = left_border[:, 0] - right_border[:, 0]
    delta_y = left_border[:, 1] - right_border[:, 1]
    
    d_pt = np.column_stack((delta_x, delta_y))
    pt = right_border

    # x = right_border[:, 0] + delta_x.T @ a[:,0]
    # y = right_border[:, 1] + delta_y.T @ a[:,1]

    D = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), k=-1) + np.diag(np.ones(n - 1), k=1)

    Hs = np.linalg.multi_dot([d_pt.T, D.T, D, d_pt])
    Bs = np.linalg.multi_dot([pt.T, D.T, D, d_pt])

    # aux = a.T
    # print(aux)
    
    def curvature(x):
        aux = np.column_stack((x, x)).T
        return np.linalg.multi_dot([aux.T, Hs, aux])[0][0] + np.linalg.multi_dot([Bs, aux])[0][0]

    constraints = [
        a >= 0,
        a <= 1,
    ]

    constraints = ({'type': 'ineq', 'fun': lambda x: x - 1},
               {'type': 'ineq', 'fun': lambda x: 1 - x})

    # Solve the problem

    bounds = [(0, 1) for _ in range(n)]
    result = minimize(curvature, a, bounds=bounds)

    a = result.x
    print(curvature(a))
    print(result.x)

    # Formulate and solve the problem
    # print("Resolvendo")
    # problem = cp.Problem(cp.Minimize(Exp), constraints)
    # problem.solve()

    # # Extract the optimal interpolation vector
    # alpha_optimal = var.value
    # print(alpha_optimal)


    x = right_border[:, 0] + delta_x.T * a
    y = right_border[:, 1] + delta_y.T * a

    # track = np.column_stack((x, y))
    # print(d_pt[:, 0] * a)
    delta = (np.column_stack((d_pt[:, 0] * a, (d_pt[:, 1] * a))))
    track = pt + delta
    # print(T_Sq)
    print(Hs)
    print(Bs)
    pass
    return track

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
    
    track = _set_track(raceline)
    
    results = {}

    repo_path = os.path.dirname(os.path.abspath(__file__))

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "vehicles", solver_opts["vehicle"])

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
    # print("-" * 50)
    # print("Forward/Backward Plus Solver")
    # print("Solver runtime: %.2f s" % (time.perf_counter() - t_start))
    # print("-" * 50)
    print("Lap time: %.3f s" % lap.t_cl[-1])
    return lap.t_cl[-1]
    # print("S1: %.3f s  |  S2: %.3f s  |  S3: %.3f s" %
    #         (lap.t_cl[track.zone_inds["s12"]],
    #         lap.t_cl[track.zone_inds["s23"]] - lap.t_cl[track.zone_inds["s12"]],
    #         lap.t_cl[-1] - lap.t_cl[track.zone_inds["s23"]]))
    # print("-" * 50)
    # v_tmp = lap.vel_cl[0] * 3.6
    # print("Start velocity: %.1f km/h" % v_tmp)
    # v_tmp = lap.vel_cl[-1] * 3.6
    # print("Final velocity: %.1f km/h" % v_tmp)
    # v_tmp = (lap.vel_cl[0] - lap.vel_cl[-1]) * 3.6
    # print("Delta: %.1f km/h" % v_tmp)
    # print("-" * 50)
    # print("Consumption: %.2f kg/lap | %.2f kJ/lap" % (lap.fuel_cons_cl[-1], lap.e_cons_cl[-1] / 1000.0))
    # # [J] -> [kJ]
    # print("-" * 50)



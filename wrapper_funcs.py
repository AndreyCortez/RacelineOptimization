import os
import numpy as np
import matplotlib.pyplot as plt
import time

import laptimesim
from race_config import *

# Por algum motivo esse módulo de laptime sim só funciona com essa bagaça
# Não sei o pq
import matplotlib
matplotlib.use('Qt5Agg')

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
    global track

    repo_path = os.path.dirname(os.path.abspath(__file__))

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")

    # set velocity limit
    if driver_opts["vel_lim_glob"] is not None:
        vel_lim_glob = driver_opts["vel_lim_glob"]
    elif solver_opts["series"] == "FE":
        vel_lim_glob = 225.0 / 3.6
    else:
        vel_lim_glob = np.inf

    # create instance
    global track 
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
        x1 = centro_da_pista[count-7]
        x2 = centro_da_pista[count-4]
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

    return borda_esquerda, borda_direita

def plot_track(track, racelines = []):
    x, y = zip(*(track))
    be, bd = track_borders(track, 5)    
    x_1, y_1 = zip(*(be))
    x_2, y_2 = zip(*(bd))
    
    #TODO: trocar o linewidths por track.width
    plt.plot(x, y, marker='')
    plt.plot(x_1, y_1, marker='')
    plt.plot(x_2, y_2, marker='')
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
    t_start = time.perf_counter()

    # call simulation
    lap.simulate_lap()

    # print("-" * 50)
    # print("Forward/Backward Plus Solver")
    # print("Solver runtime: %.2f s" % (time.perf_counter() - t_start))
    # print("-" * 50)
    # print("Lap time: %.3f s" % lap.t_cl[-1])
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



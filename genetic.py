import numpy as np
from datetime import datetime
import json
import os
from operator import itemgetter
import concurrent.futures

import wrapper_funcs

# NOTE: A função de calcular o melhor caminho não funciona bem em diversas situações e eu não sei o pq
# acho que se tentar dar uma interpolada nos valores de entrada e fazer uma filtragem passa baixa
# 

# Fazer um modelo de algoritmo que tenta ajustar cada pedaço da pista por vez, pois um pedaço,
# na maioria das vezes, não interfere no outro, temos que levar em consideração 
# fazer com que cada pedaço da pista sofra n modificações com intensidades diferentes diferentes 
# vendo quais são mais benéficas para o total

def import_save_data(name):
    arquivos_json = ['runs/' + arquivo for arquivo in os.listdir('runs/') if arquivo.startswith(name) and arquivo.endswith('.json')]
    if not arquivos_json:
        print(f"Nenhum arquivo '{name}' encontrado.")
        return None

    arquivo_mais_recente = max(arquivos_json, key=os.path.getctime)
    with open(arquivo_mais_recente, 'r') as arquivo:
        dicionario_carregado = json.load(arquivo)
        return dicionario_carregado

def save_data(prog_dic, name):
    with open(name, 'w') as arquivo:
        json.dump(prog_dic, arquivo)


def criar_populacao_inicial():
    genes = np.zeros((tam_populacao, gene_size))

    for i in genes:
        i += (np.random.random() * 2 - 1) * desvio_inicial
    genes = np.clip(genes, 0, 1)
    return genes

def reproduzir(gene_escolhido):
    genes = np.zeros((tam_populacao, gene_size))

    for i in genes:
        i += gene_escolhido + (np.random.random(gene_size) * 2 - 1) * desvio_inicial
    genes = np.clip(genes, 0, 1)
    return genes

def vetor_interpolado(vetor1, vetor2, t):
    t = np.clip(t, 0, 1)
    vetor_interpolado = (1 - t) * vetor1 + t * vetor2
    return vetor_interpolado

def gerar_raceline(gene):
    raceline = np.zeros((base.shape[0], 2))
    alpha = gene[mask]
    for i in range(len(raceline)):
        raceline[i] = vetor_interpolado(sp[i], mc[i], alpha[i])
    return np.array(raceline)

if __name__ == '__main__':

    track_name = 'Baseline'


    base = wrapper_funcs.get_track_data('input/tracks/' + track_name + '.csv')
    base[:,:2] *= 2
    # base = wrapper_funcs.filter_track(base)
    base = wrapper_funcs.get_essential_curves(base)

    sp, mc  = base['sp'], base['min_curv']
    mask = wrapper_funcs.get_intersection_mask(mc, sp)
    gene_size = mask[-1] + 1
    base = base['center']

    # TODO: Fazer um método melhor pra isso aqui
    load_latest = False

    prog_dic = import_save_data(track_name)
    if prog_dic == None or not load_latest:
        prog_dic = {
            'BEST_RUN' : 
                (0, [0 for i in range(gene_size)], np.inf)
            ,
            'NEW_BESTS' : [(0, [0 for i in range(gene_size)], np.inf)],
            'RUN_HISTORY' :
            [
                (0, np.inf)
            ]

        }

    data_atual = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    nome_arquivo = f'runs/{track_name}_{data_atual}.json'
    save_data(prog_dic, nome_arquivo)
    prog_dic = import_save_data(track_name)

    # Defines para o algoritmo
    tam_populacao = 5
    desvio_inicial = 0.5
    multithreading = False

    genes = criar_populacao_inicial()
    
    best = {}

    best = {'gene': prog_dic['BEST_RUN'][1], 'tempo' : prog_dic['BEST_RUN'][2]}
    results = [0 for i in range(tam_populacao)]

    best['gene'] = np.array(best['gene'])
    wrapper_funcs.plot_track(base[:,0:2], [mc, sp])

    iterations = prog_dic['RUN_HISTORY'][-1][0]

    while 1:
        print(f"Iteration {iterations}")
        print("-" * 50 + "")
        iterations += 1
        racelines = []    
        for i in genes:
            racelines.append(gerar_raceline(i))
        racelines = np.array(racelines)

        
        if multithreading:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futuros = [executor.submit(wrapper_funcs.simulate_raceline, arg) for arg in racelines]
                resultados = [f.result() for f in concurrent.futures.as_completed(futuros)]
        else:
            resultados = []
            for i in racelines:
                resultados.append(wrapper_funcs.simulate_raceline(i))

        for i in range(len(results)):
            results[i] = {"gene" : genes[i], "tempo" : resultados[i]}

        for i in range(len(results)):
            print(f"{i}/{tam_populacao} => tempo = {results[i]['tempo']}s")

        sorted_results = sorted(results, key=itemgetter("tempo"))
        new_best = False

        if (sorted_results[0]['tempo'] < best['tempo']):
            new_best = True
            best = sorted_results[0]

        print(f"Melhor Tempo: {best['tempo']}")

        if new_best:
            prog_dic['BEST_RUN'] = (iterations, list(best['gene']), best['tempo'])
            prog_dic['NEW_BESTS'].append([iterations, list(best['gene']), best['tempo']])


        prog_dic['RUN_HISTORY'].append([iterations, [i['tempo'] for i in sorted_results]])

        if iterations % 5 == 0 :
            save_data(prog_dic, nome_arquivo)


        genes = reproduzir(best["gene"])


import numpy as np
from operator import itemgetter
import concurrent.futures

import wrapper_funcs

base = wrapper_funcs.import_center("SaoPaulo")
left_border, right_border = wrapper_funcs.track_borders(base, 5)

# Defines para o algoritmo
tam_populacao = 5
desvio_inicial = 0.01

def criar_populacao_inicial():
    genes = np.zeros((tam_populacao, base.shape[0], 2))

    for i in genes:
        for j in i:
            j[0] += (np.random.random() * 2 - 1) * desvio_inicial
            j[1] += (np.random.random() * 2 - 1) * desvio_inicial
        
    #print(genes)
    return genes

def reproduzir(gene_escolhido):
    genes = np.zeros((tam_populacao, base.shape[0], 2))

    for i in genes:
        i += gene_escolhido
        for j in i:
            j[0] = j[0] + (np.random.random() * 2 - 1) * desvio_inicial
            j[1] = j[1] + (np.random.random() * 2 - 1) * desvio_inicial
        #print(i)
    #print(genes)
    return genes

def vetor_interpolado(vetor1, vetor2, t):
    t = np.clip(t, 0, 1)
    vetor_interpolado = (1 - t) * vetor1 + t * vetor2
    return vetor_interpolado

def gerar_raceline(gene):
    raceline = np.zeros((base.shape[0], 2))
    for i in range(len(raceline)):
        raceline[i] = vetor_interpolado(left_border[i], right_border[i], gene[i])
    return np.array(raceline)

genes = criar_populacao_inicial()
best = {'gene': [], 'tempo' : np.inf}
results = np.zeros(tam_populacao)

while 1:
    print("-" * 50 + "")
    racelines = np.array([])    
    for i in genes:
        racelines = np.append(racelines, gerar_raceline(i))

    print(racelines.shape)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futuros = [executor.submit(wrapper_funcs.simulate_raceline, arg) for arg in racelines]
        resultados = [f.result() for f in concurrent.futures.as_completed(futuros)]

    print(resultados)

    for i in range(len(results)):
        results[i] = ({"gene" : genes[i], "tempo" : resultados[i]})

    for i in range(len(results)):
        print(f"{i}/{tam_populacao} => tempo = {results[i]['tempo']}s")

    sorted_results = sorted(results, key=itemgetter("tempo"))
    if (sorted_results[0]['tempo'] < best['tempo']):
        best = sorted_results[0]
    
    print(f"Melhor Tempo: {best['tempo']}")
    genes = reproduzir(best["gene"])


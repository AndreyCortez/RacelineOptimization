import wrapper_funcs
import numpy as np

def list_files(diretorio):
    arquivos = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            arquivos.append(os.path.join(root, file))
    return arquivos

expected_lines = []
obtained_lines = []
for i in range(5):
    files = list_files(f'comparison_data/track{i}')
    print(f"expected:{wrapper_funcs.simulate_raceline(np.loadtxt(files[0]))}")
    print(f"expected:{wrapper_funcs.simulate_raceline(np.loadtxt(files[1]))}")


wrapper_funcs.simulate_raceline(i)
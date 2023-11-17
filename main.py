import wrapper_funcs
import random

track = wrapper_funcs.import_center('SaoPaulo')
wrapper_funcs.plot_track(track)

while 0:
    for i in track[1:-1]:
        pass
        i[0] += (random.random() * 2 - 1) * 0.01
        i[1] += (random.random() * 2 - 1) * 0.01
    print(track[0])

    # wrapper_funcs.plot_track(track)  
    wrapper_funcs.simulate_raceline(track)
    

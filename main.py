import os
import sys
sys.path.append(os.path.dirname(__file__))
import matplotlib
matplotlib.use('Qt5Agg')
import wrapper_funcs

track_info = wrapper_funcs.get_essential_curves('berlin.csv', True) 
# track = track[0:80]
# left_border, rigth_border = wrapper_funcs.track_borders(track, 5)
# sp = wrapper_funcs.calculate_shortest_path(left_border, rigth_border)
# lcp = wrapper_funcs.calculate_least_curvature_path(left_border, rigth_border)
# wrapper_funcs.plot_track(track, [lcp])


while 1:
    pass
    break
    # for i in track:
    #     pass
    #     i[0] += (random.random() * 2 - 1) * 0.01
    #     i[1] += (random.random() * 2 - 1) * 0.01

    # wrapper_funcs.plot_track(track)  
    # (wrapper_funcs.simulate_raceline(track))
    

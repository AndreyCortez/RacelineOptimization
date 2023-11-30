# ------------------------------------------------------------------------------------------------------------------
# USER INPUT -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# F1 qualifying mode:   DRS activated, EM strategy FCFB, initial_energy 4.0MJ, full power of 575kW
# F1 race mode:         DRS as desired, EM strategy LBP, initial_energy 0.0MJ, power reduced to 546kW (-5%)
# FE qualifying mode:   DRS deactivated, EM strategy FCFB
# FE race mode:         DRS deactivated, EM strategy FCFB + lift&coast
# tracks must be unclosed, i.e. last point != first point!

# track options ----------------------------------------------------------------------------------------------------
# trackname:            track name of desired race track (file must be available in the input folder)
# flip_track:           switch to flip track direction if required
# mu_weather:           [-] factor to consider wet track, e.g. by mu_weather = 0.6
# interp_stepsize_des:  [m], desired stepsize after interpolation of the input raceline points
# curv_filt_width:      [m] window width of moving average filter -> set None for deactivation
# use_drs1:             DRS zone 1 switch
# use_drs2:             DRS zone 2 switch
# use_pit:              activate pit stop (requires _pit track file!)

track_opts = {"trackname": "Shanghai",
                "flip_track": False,
                "mu_weather": 1.0,
                "interp_stepsize_des": 5,
                "curv_filt_width": 10.0,
                "use_drs1": False,
                "use_drs2": False,
                "use_pit": False}

# solver options ---------------------------------------------------------------------------------------------------
# vehicle:                  vehicle parameter file
# series:                   F1, FE
# limit_braking_weak_side:  can be None, 'FA', 'RA', 'all' -> set if brake force potential should be determined
#                           based on the weak (i.e. inner) side of the car, e.g. when braking into a corner
# v_start:                  [m/s] ve    wrapper_funcs.plot_track(base, [gerar_raceline(best['gene']))
# velocity at start
# find_v_start:             determine the real velocity at start
# max_no_em_iters:          maximum number of iterations for EM recalculation
# es_diff_max:              [J] stop criterion -> maximum difference between two solver runs

solver_opts = {"vehicle": "FE_Berlin.ini",
                "series": "FE",
                "limit_braking_weak_side": 'FA',
                "v_start": 0 / 3.6,
                "find_v_start": False,
                "max_no_em_iters": 5,
                "es_diff_max": 1.0}

# driver options ---------------------------------------------------------------------------------------------------
# vel_subtr_corner: [m/s] velocity subtracted from max. cornering vel. since drivers will not hit the maximum
#                   perfectly
# vel_lim_glob:     [m/s] velocity limit, set None if unused
# yellow_s1:        yellow flag in sector 1
# yellow_s2:        yellow flag in sector 2
# yellow_s3:        yellow flag in sector 3
# yellow_throttle:  throttle position in a yellow flag sector
# initial_energy:   [J] initial energy (F1: max. 4 MJ/lap, FE Berlin: 4.58 MJ/lap)
# em_strategy:      FCFB, LBP, LS, NONE -> FCFB = First Come First Boost, LBP = Longest (time) to Breakpoint,
#                   LS = Lowest Speed, FE requires FCFB as it only drives in electric mode!
# use_recuperation: set if recuperation by e-motor and electric turbocharger is allowed or not (lift&coast is
#                   currently only considered with FCFB)
# use_lift_coast:   switch to turn lift and coast on/off
# lift_coast_dist:  [m] lift and coast before braking point

driver_opts = {"vel_subtr_corner": 0.5,
                "vel_lim_glob": None,
                "yellow_s1": False,
                "yellow_s2": False,
                "yellow_s3": False,
                "yellow_throttle": 0.3,
                "initial_energy": 4.0e6,
                "em_strategy": "FCFB",
                "use_recuperation": False,
                "use_lift_coast": False,
                "lift_coast_dist": 10.0}

# sensitivity analysis options -------------------------------------------------------------------------------------
# use_sa:   switch to deactivate sensitivity analysis
# sa_type:  'mass', 'aero', 'cog'
# range_1:  range of parameter variation [start, end, number of steps]
# range_2:  range of parameter variation [start, end, number of steps] -> CURRENTLY NOT IMPLEMENTED

sa_opts = {"use_sa": False,
            "sa_type": "mass",
            "range_1": [733.0, 833.0, 5],
            "range_2": None}

# debug options ----------------------------------------------------------------------------------------------------
# use_plot:                 plot results
# use_debug_plots:          plot additional plots for debugging
# use_plot_comparison_tph:  calculate velocity profile with TPH FB solver and plot a comparison
# use_print:                set if prints to console should be used or not (does not suppress hints/warnings)
# use_print_result:         set if result should be printed to console or not

debug_opts = {"use_plot": False,
                "use_debug_plots": False,
                "use_plot_comparison_tph": False,
                "use_print": False,
                "use_print_result": True}

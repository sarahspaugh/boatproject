import numpy as np
from environment_setup import sailing_env
from sailboat_model import sailboat
from memory import MemRecorder
import pickle

START = (21.701746, -160.253116)
FINISH = (5.794623, -161.988956)
lat_low = 3.125
lat_high = 24.125
lon_left = -164.125
lon_right = -158.125

# spawn environment
ocean = sailing_env(START, FINISH, (lat_low, lat_high), (lon_left, lon_right))

#fig1, fig2 = ocean.plot_environment()

# init classes
simboat = sailboat(START[0], START[1])
hist = MemRecorder(simboat)

num_slices = 8 # formerly num waypoints
best_headings = simboat.plan_highlevel_route(FINISH, ocean, num_slices)
lat_benchmarks = simboat.waypoint_lats

for sliceID in range(1,num_slices+1):

    ocean.randomize_slice(lat_benchmarks[sliceID], lat_benchmarks[sliceID-1])

    while_iters = 0
    while simboat.loc[0] > lat_benchmarks[sliceID]:
        simboat.measure_conditions(ocean)
        u0 = simboat.calc_next_action(ocean, 50, best_headings[sliceID-1])
        simboat.update_heading(u0)
        simboat.update_states(ocean)
        hist.update(simboat)
        while_iters += 1
        if while_iters % 50 == 0:
            print("location is currently {},{}".format(simboat.loc[0], simboat.loc[1]))

    print("passed slice number {}".format(sliceID))

loc_hist = hist.loc_history
heading_hist = np.array(hist.heading_hist)
speed_hist = np.array(hist.boatspeed_hist)

np.save("loc_hist.npy", loc_hist)
np.save("H_hist.npy", heading_hist)
np.save("bspeed_hist.npy", speed_hist)

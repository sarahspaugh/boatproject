import numpy as np

class MemRecorder:

    def __init__(self, boat):
        self.loc_history = (boat.loc).reshape((1, 2))
        self.heading_hist = [boat.heading]
        self.boatspeed_hist = [boat.boatspeed]

    def update(self, boat):
        self.loc_history = np.concatenate((self.loc_history, (boat.loc).reshape((1,2))), axis=0)
        self.heading_hist.append(boat.heading)
        self.boatspeed_hist.append(boat.boatspeed)


import numpy as np
from wingsail import wingsail
import cvxpy as cp

"""
Dynamics model and helper functions for playing w/ autonomous sailboat control 
"""

# global #
MAX_SPEED = 4.15 # m/s; equivalent to 8 knots
CRUISE_SPEED = 1.25 # m/s; equivalent to ~2.5 knots

MAX_HEADING_CHANGE = 90 # maximum heading change in one timestep [degrees]
ALPHA_0 = 9.5 # optimal angle of attack [deg]

sail = wingsail()

W1 = 5 # opt weight on beta?
W2 = 5 # opt weight on heading/opt heading err
W3 = np.identity(1) # opt weight on getting south
W4 = np.identity(2)*10 # opt weight on going to endpoint
# # # # # #

def deg_to_meters(deg=None, meters=None):
    """
    if handed degrees (lat/lon), convert to meters
    if handed metres, convert to degrees
    :param deg:
    :param meters:
    :return:
    """
    if (deg is None) and (meters is None):
        print("no values!")
        return None
    elif (deg is not None) and (meters is not None):
        print("too many values")
        return None
    elif (deg is not None):
        return deg * 11.1 / 0.0001
    else:
        return meters * 0.0001 / 11.1

class sailboat:

    def __init__(self, lat, lon):
        self.loc = np.array([lat, lon])
        self.heading = 180 # start pointing south
        self.boatspeed = 0 # u (velocity relative to water)
        self.true_velocity = np.array([0,0]) # [m/s] lat. direction, lon. direction
        self.waypoints = None
        self.dist_to_goal = np.inf
        self.beta = 0 # angle between boat centerline and wind vector, between 0,90 [deg] (or maybe -90,90 but prolly not)
        self.dt = 100 # seconds. arbitrary choice, may need to edit
        self.windspeed = 0 # apparent velocity relative to boat (magnitude)

        # wat is boat like
        self.Cd = 0.01 # hull coefficient of drag
        self.wet_area = 7 * 0.95 # [m^2], area of hull relevant to friction drag
        self.mass = 750 # kg, according to google



    def plan_highlevel_route(self, goal, env, num_waypoints):
        """
        select the best-case headings from start to goal location

        :param goal: final destination
        :param env: coarse map of wind/water conditions
        :param num_waypoints: number of sub-destinations to choose w/ high level path planner
        :param generate_plot: bool, whether to plot the high-level intended trajectory
        :param starter_plot: for overlay purposes
        :return: list of optimal headings;
        """
        start = self.loc # np.array(lat,lon)
        dist = goal - start
        assert dist.shape == (2,), "array is wrong shape, goal or start"

        lat_delta = dist[0] / (num_waypoints + 1)
        waypoint_lats = np.arange(num_waypoints + 1) * lat_delta + start[0]
        waypoint_lats = np.concatenate((waypoint_lats, np.array([goal[0]])))
        self.waypoint_lats = waypoint_lats

        lon_opts = np.arange(-165, -158)
        headings = np.zeros(num_waypoints + 1)

        for i in range(num_waypoints + 1):
            lat_high = waypoint_lats[i]
            lat_low = waypoint_lats[i+1]
            u_avg, v_avg = env.get_avg_conditions(lat_low, lat_high)
            windspeed = np.sqrt(u_avg**2 + v_avg**2)

            wind_angle = np.arctan2(u_avg, v_avg) + np.pi # * 180 / np.pi + 180
            boat_angles = np.array([np.arctan2(lat_delta, x) for x in lon_opts])
            betas = wind_angle - boat_angles
            force_coeffs = np.abs(np.sin(betas))
            max_force_idx = np.argmax(force_coeffs)
            headings[i] = boat_angles[max_force_idx]

        self.opt_headings = headings
        self.goal = goal
        return self.opt_headings

    def measure_conditions(self, env):
        """
        measure environment conditions at current location
        updates windspeed, boatspeed, beta angle
        assumes heading, true velocity are current
        :param env: environment
        :return: none
        """

        # measure windspeed w.r.t. boat (take out our earth velocity from the wind's earth velocity)
        wind, water = env.get_conditions(self.loc)
        relwind_u = wind[0] - self.true_velocity[0]
        relwind_v = wind[1] - self.true_velocity[1]
        self.windspeed = np.sqrt(relwind_u**2 + relwind_v**2)

        # measure boatspeed w.r.t. water (since we secretly only have earth current velocity,
        # subtract this from true earth vel)
        rel_dlat = self.true_velocity[0] - water[0]
        rel_dlon = self.true_velocity[1] - water[1]
        self.boatspeed = np.sqrt(rel_dlat**2 + rel_dlon**2)

        # measure wind angle difference from boat heading
        raw_wind_angle = np.arctan2(wind[0], wind[1]) * 180 / np.pi # should be between -180, 180
        raw_wind_angle += 180 # match heading
        self.beta = raw_wind_angle - self.heading
        # sign shouldn't matter because when we calc forces with the selftrimming sail
        # we always know force directions wrt boat

    def calc_next_action(self, env, horizon_len, opt_heading):
        """
        single iteration of MPC optimization method;
        calculate a short optimal trajectory based on simplified dynamics

        :param horizon_len: number of timesteps for single planning rollout
        :return: first control input from optimal trajectory
        """

        # get current actual-env conditions
        wind, water = env.get_conditions(self.loc)
        wind_angle = np.arctan2(wind[0], wind[1]) * 180 / np.pi + 180

        H = cp.Variable(horizon_len) # headings
        roughloc = cp.Variable((horizon_len, 2)) # amount of south-ness

        obj = cp.Minimize((cp.abs(wind_angle-H-180)-90)*W1 # try to get the sail normal to the wind
                          + cp.square(H-opt_heading)*W2 # try to stick to the optimal heading
                          + cp.quad_form(self.goal - roughloc[-1], W4)) # try to go to dest

        constraints = [cp.abs(H + wind_angle >= 15)] # no go upwind
        constraints += [roughloc[0] == self.loc] # start at start

        for i in range(1, horizon_len):
            constraints += [cp.abs(H[i] - H[i-1])<= 90] # max steer per timestep (don't have to worry about the zero pt because we aren't sailing north)
            constraints += [roughloc[i,0] == roughloc[i-1,0] - cp.abs((wind_angle-H[i-1])*np.pi/180)] # fake latitude update
            constraints += [roughloc[i,1] == roughloc[i-1,1] - cp.sign(H[i-1]-180)*(180 -cp.abs(H[i-1] -180))*np.pi/180 ] # fake long update

        prob = cp.Problem(obj, constraints)
        prob.solve()

        next_action = H.value[0]
        return next_action




    def update_heading(self, new_heading):
        """
        update the heading according to the control plan
        :param new_heading: the new heading
        :return: None
        """
        if new_heading - self.heading > MAX_HEADING_CHANGE:
            self.heading += MAX_HEADING_CHANGE * np.sign(new_heading - self.heading)
        else:
            self.heading = new_heading


    def update_states(self, env):
        """
        advance "actual" boat to next timestep
        I am just euler integrating because i am lame

        :return: none
        """

        FL, FD = sail.get_forces(self.windspeed)
        _, water = env.get_conditions(self.loc)
        d_boatspeed = (np.abs(FL*np.sin(np.deg2rad(self.beta))) - np.abs(FD*np.cos(np.deg2rad(self.beta)))) *1/self.mass# todo hull drag?

        # position derivatives
        d_lat = ((self.boatspeed * np.cos(np.deg2rad(self.heading)) + water[0]) * self.dt +
                 0.5 * d_boatspeed * np.cos(np.deg2rad(self.heading)) * self.dt**2)
        d_lon = ((self.boatspeed * np.sin(np.deg2rad(self.heading)) + water[1]) * self.dt +
                 0.5 * d_boatspeed * np.sin(np.deg2rad(self.heading)) * self.dt**2)

        # change units
        d_lat = deg_to_meters(meters=d_lat)
        d_lon = deg_to_meters(meters=d_lon)

        # update, calc m/s achieved velocity
        newloc = self.loc + np.array([d_lat, d_lon])
        diff = newloc - self.loc
        diff_meters = deg_to_meters(deg=diff)

        # store
        self.true_velocity = np.array([diff_meters[0]/self.dt, diff_meters[1]/self.dt])
        self.loc = newloc














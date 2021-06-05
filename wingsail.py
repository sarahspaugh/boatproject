"""
Modeling tools for free-rotating wingsail/trimsail unit
normal metric units unless otherwise indicated
assumes NACA 0018 symmetric airfoil profile
"""
import numpy as np

# globals #

CHORD = 1
SPAN = 5


###########

class wingsail:

    def __init__(self):
        self.SA = CHORD * SPAN # sail area
        self.chord_width = CHORD * 0.18 # "thickness" of wingsail
        self.AR = SPAN / CHORD # aspect ratio

        """
        # trim sail parameters (not currently in use)
        self.trim_SA = self.SA * 0.10
        self.trim_chord = np.sqrt(self.trim_SA * self.AR)
        self.trim_span = self.AR * self.trim_chord
        self.delta = 0 # angle of trimsail rel. to mainsail (control parameter)
        self.trim_chordwidth = self.trim_chord * 0.18
        """

        # airfoil constants (most accurate @ 3-4 m/s apparent wind)
        self.alpha_0 = 9.5 # degrees
        self.CL = 0.5 # lift coef at optimal alpha
        self.CD = 0.08 # drag coef " " "
        """
        # these are not being used for now (would be for more complex modeling) 
        self.CDp = 0.04 # other drag " " "
        self.CM = 0.02 # moment " " "
        """
        self.variable_alpha = False



    def set_reynolds(self, airspeed):
        """
        calculate the reynolds number given the apparent windspeed
        assumes air at ~10 deg C / 50 deg F
        :param airspeed: apparent wind speed, m/s
        :return: none
        """
        kin_visc = 1.4207e-5 # kinematic viscosity for air, 10deg C. [m^2/s]
        self.Re = self.chord_width * airspeed / kin_visc
        self.trim_Re = self.trim_chordwidth * airspeed / kin_visc


    def get_forces(self, windspeed):
        """
        Return the magnitude of lift/drag forces according to current wingsail params for alpha, Cd/Cl
        :param windspeed: speed of wind relative to boat
        :return: |F_L|, |F_D| (perpendicular/parallel to wind vector)
        """
        rho = 1.225 # air density [kg / m^3]
        F_L = 0.5 * rho * windspeed**2 * self.CL * self.SA
        F_D = 0.5 * rho * windspeed**2 * self.CD * self.SA

        return F_L, F_D


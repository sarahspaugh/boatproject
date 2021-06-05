import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

WIND_CURRENT_SCALE = 1.25 # max val of current in original map

# these scale the noise added to local wind/water when making granular (std devs)
gauss_scale_wind = 1.5
gauss_scale_water = 0.2

# these need to match and set how granular to make local maps/noise additions
GRAN = 0.001
ROUND = 3

"""
The 'environment' class for this group of simulations is built up from global ocean wind data
downloaded in '.nc' format from https://podaac.jpl.nasa.gov/dataset/CCMP_MEASURES_ATLAS_L4_OW_L3_5A_5DAY_WIND_VECTORS_FLK
translated to labeled text format using Panoply (free download) and read as csv
"surface currents" are assumed to roughly match wind vectors, scaled down to reasonable size with added Gaussian noise
"""


def deg_to_meters(deg=None, meters=None):
    if (deg is None) and (meters is None):
        print("no values!")
        return None
    elif deg and meters:
        print("too many values")
        return None
    elif deg:
        return deg * 11.1 / 0.0001
    else:
        return meters * 0.0001 / 11.1


class sailing_env:
    """
    setup sailing environment
    start/finish in lat/long ordered tuples
    lat and long bounds in numerical ordered tuples
    """

    def __init__(self, start, finish, lat_bounds, long_bounds, prev_windmap=None, prev_currents=None):
        if prev_windmap is not None and prev_currents is not None:
            return

        self.start = start
        self.finish = finish

        self.lat_lower, self.lat_upper = lat_bounds
        self.lon_lower, self.lon_upper = long_bounds

        u_wind_raw = pd.read_csv('uwnd.txt', delim_whitespace=True)
        v_wind_raw = pd.read_csv('vwnd.txt', delim_whitespace=True)

        u_wind_raw.drop(columns=["time"], inplace=True)
        v_wind_raw.drop(columns=["time"], inplace=True)

        # fix coord system
        u_wind_raw["lon"] -= 180
        v_wind_raw["lon"] -= 180

        # chop out relevant map area
        uwind_mask = u_wind_raw.loc[u_wind_raw['lat'] >= self.lat_lower].drop_duplicates(subset=["lat", "lon"])
        uwind_mask = uwind_mask.loc[uwind_mask['lat'] <= self.lat_upper]
        uwind_mask = uwind_mask.loc[uwind_mask['lon'] >= self.lon_lower]
        uwind_mask = uwind_mask.loc[uwind_mask['lon'] <= self.lon_upper]

        vwind_mask = v_wind_raw.loc[v_wind_raw['lat'] >= self.lat_lower].drop_duplicates(subset=["lat", "lon"])
        vwind_mask = vwind_mask.loc[vwind_mask['lat'] <= self.lat_upper]
        vwind_mask = vwind_mask.loc[vwind_mask['lon'] >= self.lon_lower]
        vwind_mask = vwind_mask.loc[vwind_mask['lon'] <= self.lon_upper]

        # combine to only one datastructure
        self.wind_vec_map = uwind_mask
        self.wind_vec_map["vwnd"] = vwind_mask["vwnd"]

        # reindex for evenness
        """ jk
        self.wind_vec_map.set_index(["lat"], inplace=True)
        even_lats = pd.Index(np.linspace(lat_lower, lat_upper, num=int((lat_upper-lat_lower)/0.125)))
        self.wind_vec_map.reindex(index=even_lats)
        self.wind_vec_map.reset_index(['lat'], inplace=True).set_index(['lon'], inplace=True)
        even_lons = pd.Index(np.linspace(lon_lower, lon_upper, num=int((lon_upper-lon_lower)/0.125)))
        self.wind_vec_map.reindex(index=even_lons)
        self.wind_vec_map.reset_index(['lon'], inplace=True)
        self.wind_vec_map.drop_duplicates(subset=["lat", "lon"], inplace=True)
        """

        max_wind_u = max(self.wind_vec_map["uwnd"])
        max_wind_v = max(self.wind_vec_map["vwnd"])
        self.max_windspeed = np.sqrt(max_wind_u ** 2 + max_wind_v ** 2)

        # transform windmap to work for currents (max value will be WIND_CURRENT_SCALE before additional manips)
        self.current_vec = self.wind_vec_map.copy()
        self.current_vec["uwnd"] /= (self.max_windspeed / WIND_CURRENT_SCALE)
        self.current_vec["vwnd"] /= (self.max_windspeed / WIND_CURRENT_SCALE)

        # add addtional west-flowing current
        self.current_vec["vwnd"] += np.random.uniform(-0.3, 0, self.current_vec["vwnd"].shape)

        # add noise
        offset_u = np.random.normal(0, 0.2, self.current_vec["uwnd"].shape)
        self.current_vec["uwnd"] += offset_u
        offset_v = np.random.normal(0, 0.1, self.current_vec["vwnd"].shape)
        self.current_vec["vwnd"] += offset_v

        # rename
        self.current_vec.rename(columns={"uwnd": "ucrt", "vwnd": "vcrt"}, inplace=True)

    def randomize_slice(self, lat_high, lat_low, granularity=GRAN):
        """
        fill in map sparseness
        granularity of 0.001 is equivalent to about 110 meters or 10 seconds of max speed sailing
        original granularity ranges between .125 and .25 degrees, or minimum 13,875 m

        :return: None
        """

        lat_range = np.linspace(lat_low, lat_high, int(lat_high / granularity) + 1)
        lon_range = np.linspace(self.lon_lower, self.lon_upper, int(np.abs(self.lon_upper / granularity)) + 1)
        LL_index = pd.MultiIndex.from_product([lat_range, lon_range], names=["lat", "lon"])

        self.water_slice = pd.DataFrame(index=LL_index, columns=["u", "v"])
        self.wind_slice = pd.DataFrame(index=LL_index, columns=["u", "v"])

        coarse_water = self.current_vec.set_index(["lat", "lon"])
        coarse_wind = self.wind_vec_map.set_index(["lat", "lon"])

        last_uwind = 1
        last_vwind = 1
        last_uwatr = 1
        last_vwatr = 1

        for i in range(LL_index.shape[0]):
            coords = LL_index[i]
            try:
                row_water = coarse_water.loc[coords]
                row_wind = coarse_wind.loc[coords]

                print("checkpoint okay")
            except KeyError:
                row_water = None
                row_wind = None

            if row_water is not None:
                last_uwind = row_wind["uwnd"]
                last_vwind = row_wind["vwnd"]
                last_uwatr = row_water["ucrt"]
                last_vwatr = row_water['vcrt']
                self.water_slice[coords, "u"] = last_uwatr
                self.water_slice[coords, "v"] = last_vwatr
                self.wind_slice[coords, "u"] = last_uwind
                self.wind_slice[coords, "v"] = last_vwind

            else:
                self.water_slice[coords, "u"] = last_uwatr + np.random.normal(0, gauss_scale_water / 2)
                self.water_slice[coords, "v"] = last_vwatr + np.random.normal(0, gauss_scale_water)
                self.wind_slice[coords, "u"] = last_uwind + np.random.normal(0, gauss_scale_wind)
                self.wind_slice[coords, "v"] = last_vwind + np.random.normal(0, gauss_scale_wind)

    def get_avg_conditions(self, lat_low, lat_high):
        """
        get average wind for a slice of env

        :param lat_low: lower bound of slice to consider
        :param lat_high: upper bound of same
        :return: average wind vector
        """
        windslice = self.wind_vec_map.loc[self.wind_vec_map["lat"] >= lat_low]
        windslice = windslice.loc[windslice["lat"] <= lat_high]

        u_avg = windslice["uwnd"].mean()
        v_avg = windslice["vwnd"].mean()

        return u_avg, v_avg

    def get_conditions(self, coords):
        """
        Fill in the environment maps
        :param coords: location as numpy array lat/long pair
        :return: local wind, local water
        """
        lat = np.round(coords[0], ROUND)
        long = np.round(coords[0], ROUND)

        LL = (lat, long)

        try:
            wind_vec = self.wind_slice.loc[LL]
            water_vec = self.water_slice.loc[LL]
        except KeyError as e:
            print(e)
            return None

        wind = (wind_vec["u"], wind_vec["v"])
        watr = (water_vec["u"], water_vec["v"])

        return wind, watr

    def plot_environment(self):
        windplot = ff.create_quiver(x=self.wind_vec_map["lon"].to_numpy(),
                                    y=self.wind_vec_map["lat"].to_numpy(),
                                    u=self.wind_vec_map["uwnd"].to_numpy(),
                                    v=self.wind_vec_map["vwnd"].to_numpy(),
                                    arrow_scale=0.05,
                                    name="Wind vectors (m/s)",
                                    line_width=1)
        windplot.add_trace(go.Scatter(x=[self.start[1]], y=[self.start[0]],
                                      mode="markers",
                                      marker_size=12,
                                      fillcolor="red",
                                      name="Start"))
        windplot.add_trace(go.Scatter(x=[self.finish[1]], y=[self.finish[0]],
                                      mode="markers",
                                      marker_size=12,
                                      fillcolor="green",
                                      name="Finish"))
        windplot.update_xaxes(title_text="Longitude")
        windplot.update_yaxes(title_text="Latitude")
        windplot.update_layout(title="Wind pattern over environment space")
        windplot.show()

        waterplot = ff.create_quiver(x=self.current_vec["lon"],
                                     y=self.current_vec["lat"],
                                     u=self.current_vec["ucrt"],
                                     v=self.current_vec["vcrt"],
                                     arrow_scale=0.1,
                                     name="Surface current vectors (m/s)",
                                     line_width=1)
        waterplot.add_trace(go.Scatter(x=[self.start[1]], y=[self.start[0]],
                                       mode="markers",
                                       marker_size=12,
                                       fillcolor="red",
                                       name="Start"))
        waterplot.add_trace(go.Scatter(x=[self.finish[1]], y=[self.finish[0]],
                                       mode="markers",
                                       marker_size=12,
                                       fillcolor="green",
                                       name="Finish"))
        waterplot.update_xaxes(title_text="Longitude")
        waterplot.update_yaxes(title_text="Latitude")
        waterplot.update_layout(title="Surface currents over environment space")
        waterplot.show()

        return windplot, waterplot

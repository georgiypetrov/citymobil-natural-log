import polyline as pl
import pandas as pd
import numpy as np

from geopy import distance

def get_distance(lat1, lon1, lat2, lon2):
    """
    Get distance from two coordinates.
    :param lat1: first latitude coord
    :param lon1: first longitude coord
    :param lat2: second latitude coord
    :param lon2: second longitude coord
    :return: distance in km
    """
    KM = 6371.393
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = KM * c
    return distance

df = pd.read_csv("../data/train.csv/train.csv")

ix = 0

try:
    route = df.iloc[ix].route
    print(route)
    dat = pl.decode(route)
    l = len(dat)

    dist = 0.0

    for i in range(1, l):
        dist += get_distance(dat[i-1][0], dat[i-1][1], dat[i][0], dat[i][1])

    print(dist)
except TypeError:
    print("polyline error")



import polyline as pl
import pandas as pd

from geopy import distance

df = pd.read_csv("../data/train.csv/train.csv")

ix = 0

try:
    route = df.iloc[ix].route
    print(route)
    dat = pl.decode(route)
    l = len(dat)

    dist = 0.0

    for i in range(1, l):
        dist += distance.geodesic(dat[i-1], dat[i]).km

    print(dist)
except TypeError:
    print("polyline error")



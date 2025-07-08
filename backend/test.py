from herbie import Herbie
from herbie.toolbox import EasyMap, pc
from herbie import paint

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

H = Herbie(
    "2021-07-19",
    model="hrrr",
    product="sfc",
    fxx=0,
)
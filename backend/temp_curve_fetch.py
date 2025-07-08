#!/usr/bin/env python3
"""
Fetch HRRR 2-m temperature for Central Park through the Herbie package
"""

import pandas as pd
import xarray as xr
from herbie import FastHerbie
from dask.distributed import Client, LocalCluster
#import matplotlib.pyplot as plt
from time import perf_counter

def build_series():
    # 1. Location & time span
    lat, lon = 40.7790, -73.9693
    point = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    dates = pd.date_range("2016-01-15 00", "2016-01-15 23", freq="1h")

    print("Point:", point.iloc[0].to_dict())
    print("Hours:", len(dates))

    # 2. FastHerbie
    fh = FastHerbie(
        dates,
        model="hrrr",
        product="sfc",
        fxx=[0],
        source="aws",
    )

    print("Herbie objects to fetch:", len(fh.objects))
    fh.download(":TMP:2 m", verbose=True, overwrite=False)
    print("Downloads finished")

    # 3. Open each file (no Dask yet, keep it simple)
    datasets = []
    for H in fh.file_exists:
        try:
            ds = H.xarray(":TMP:2 m", remove_grib=True)
            datasets.append(ds)
        except Exception as e:
            print("⚠️  could not read", H, "--", e)

    if not datasets:
        raise RuntimeError("Nothing readable!")

    print("Datasets read:", len(datasets))

    # 4. Concatenate & pick the point
    ds_all = xr.concat(datasets, dim="valid_time")
    t2m_C = (ds_all.herbie.pick_points(point).t2m - 273.15).rename("t2m_C")
    return t2m_C.to_pandas()


def main():
    series = build_series()

    """
    # --- quick plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))  # one clean figure
    series.plot(ax=ax)  # time on x-axis, °C on y-axis
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Central Park HRRR 2-m Temperature – 15 Jan 2015")
    plt.tight_layout()
    plt.show()
    """

    # 5. (Optional) small parallel compute example
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        processes=False,       # ← keeps life simple for now
        dashboard_address=":0",
    )
    with Client(cluster) as client:
        result = client.gather(client.scatter(series))

    print("\n=== Done ===")
    print(result.head())


if __name__ == "__main__":
    t0 = perf_counter()      # start timer
    main()                   # run everything exactly as before
    t1 = perf_counter()      # stop timer
    print(f"\nTotal runtime: {t1 - t0:.2f} seconds")

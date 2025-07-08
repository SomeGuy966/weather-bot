#!/usr/bin/env python3
"""
latest_knyc_full_obs_with_age.py
Fetch and display the full suite of latest KNYC observations
from api.weather.gov, with both raw SI values and converted units,
plus the age of the observation in seconds.
"""

import requests
from datetime import datetime, timezone

API_URL = "https://api.weather.gov/stations/KNYC/observations/latest"

def c_to_f(c):
    return c * 9/5 + 32

def mps_to_mph(mps):
    return mps * 2.23694

def pa_to_inhg(pa):
    return pa * 0.0002953

def m_to_miles(m):
    return m * 0.000621371

def get_latest_knyc_obs():
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    props = resp.json()["properties"]

    obs_time = datetime.fromisoformat(props["timestamp"])
    temp_c = props["temperature"]["value"]

    return {
        "obs_time":               obs_time,
        "temp_c":                 temp_c,
        "temp_f":                 c_to_f(temp_c) if temp_c is not None else None,
        "dewpoint_c":             props["dewpoint"]["value"],
        "dewpoint_f":             c_to_f(props["dewpoint"]["value"]) if props["dewpoint"]["value"] is not None else None,
        "wind_speed_mps":         props["windSpeed"]["value"],
        "wind_speed_mph":         mps_to_mph(props["windSpeed"]["value"]) if props["windSpeed"]["value"] is not None else None,
        "wind_direction_deg":     props["windDirection"]["value"],
        "wind_gust_mph":          mps_to_mph(props.get("windGust", {}).get("value")) if props.get("windGust", {}).get("value") is not None else None,
        "pressure_pa":            props["barometricPressure"]["value"],
        "pressure_inhg":          pa_to_inhg(props["barometricPressure"]["value"]) if props["barometricPressure"]["value"] is not None else None,
        "sea_level_pressure_inhg": pa_to_inhg(props.get("seaLevelPressure", {}).get("value")) if props.get("seaLevelPressure", {}).get("value") is not None else None,
        "visibility_miles":       m_to_miles(props["visibility"]["value"]) if props["visibility"]["value"] is not None else None,
        "relative_humidity":      props.get("relativeHumidity", {}).get("value"),
        "precip_1hr_inches":      m_to_miles(props.get("precipitationLastHour", {}).get("value")) if props.get("precipitationLastHour", {}).get("value") is not None else None,
        "text_description":       props.get("textDescription"),
        "icon_url":               props.get("icon"),
    }

if __name__ == "__main__":
    obs = get_latest_knyc_obs()
    now_utc = datetime.now(timezone.utc)
    lag_sec = (now_utc - obs["obs_time"]).total_seconds()

    print(f"Observation time (UTC): {obs['obs_time']:%Y-%m-%d %H:%M}")
    print(f"Air temp           : {obs['temp_f']:.1f} °F  ({obs['temp_c']:.1f} °C)")
    print(f"Dew point           : {obs['dewpoint_f']:.1f} °F  ({obs['dewpoint_c']:.1f} °C)")
    print(f"Wind                : {obs['wind_speed_mph']:.1f} mph @ {obs['wind_direction_deg']}°"
          + (f", gust {obs['wind_gust_mph']:.1f} mph" if obs["wind_gust_mph"] is not None else ""))
    print(f"Pressure (station)  : {obs['pressure_inhg']:.2f} inHg")
    if obs["sea_level_pressure_inhg"] is not None:
        print(f"Pressure (sea level): {obs['sea_level_pressure_inhg']:.2f} inHg")
    print(f"Visibility          : {obs['visibility_miles']:.1f} mi")
    if obs["relative_humidity"] is not None:
        print(f"Relative humidity   : {obs['relative_humidity']}%")
    if obs["precip_1hr_inches"] is not None:
        print(f"Precipitation (1 hr): {obs['precip_1hr_inches']:.2f} in")
    print(f"Conditions          : {obs['text_description']}")
    print(f"Icon                : {obs['icon_url']}")
    print(f"Age of obs          : {lag_sec:.0f} s old")

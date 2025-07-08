import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, timezone

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def get_knyc_one_min_window(delta_hours=1):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=delta_hours)
    params = {
        "station": "KNYC",
        "data":    ["tmpf","dwpf"],
        "tz":      "UTC",
        "format":  "onlycomma",
        # ISO timestamps, matching the docs’ sts/ets
        "sts":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ets":     now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = requests.get(IEM_URL, params=params, timeout=10)
    r.raise_for_status()
    print("URL:", r.url)
    print("Raw response:\n", "\n".join(r.text.splitlines()[:5]))
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        print("❗️ No data returned in that window.")
    else:
        df["valid"] = pd.to_datetime(df["valid"], utc=True)
        df = df.set_index("valid").sort_index()
    return df

if __name__ == "__main__":
    df = get_knyc_one_min_window(1)
    print(df)

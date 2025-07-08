import requests
import pandas as pd
from datetime import datetime, timedelta


def daily_highs(station_id, start, end, token, units='standard',
                limit=1000):
    """
    Fetch TMAX over [start, end] by slicing into <=1-year windows
    and concatenating all results into a single DataFrame.
    """
    def year_chunks(s, e):
        s_dt = datetime.fromisoformat(s)
        e_dt = datetime.fromisoformat(e)
        while s_dt <= e_dt:
            # define end of this chunk: one year minus one day from s_dt
            next_year = s_dt.replace(year=s_dt.year + 1) - timedelta(days=1)
            chunk_end = min(next_year, e_dt)
            yield s_dt.date().isoformat(), chunk_end.date().isoformat()
            s_dt = chunk_end + timedelta(days=1)

    all_rows = []
    for cs, ce in year_chunks(start, end):
        print(f"Chunk fetch: {cs} → {ce}")
        # original paging logic for this sub-range
        url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
        params = {
            'datasetid': 'GHCND',
            'datatypeid': 'TMAX',
            'stationid': station_id,
            'startdate': cs,
            'enddate': ce,
            'units': units,
            'limit': limit,
            'offset': 0
        }
        headers = {'token': token}
        rows = []
        page = 1

        while True:
            print(f"  Fetching page {page} (offset={params['offset']})…")
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json().get('results', [])
            rows.extend(data)
            print(f"    Retrieved {len(data)} records")
            if len(data) < limit:
                break
            params['offset'] += limit
            page += 1

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)[['date', 'value']].rename(columns={'value': 'tmax'})
    print(f"Total records after concatenation: {len(df)}")
    return df


if __name__ == "__main__":
    # map user choices to station IDs and short names
    stations = {
        '1': ('GHCND:USW00094728', 'knyc'),
        '2': ('GHCND:USW00014819', 'kmdw'),
        '3': ('GHCND:USW00012839', 'kmia'),
        '4': ('GHCND:USW00013904', 'kaus'),
        '5': ('GHCND:USW00012918', 'khou'),
        '6': ('GHCND:USW00003017', 'kden'),
        '7': ('GHCND:USW00013739', 'kphl'),
        '8': ('GHCND:USW00023174', 'klax'),
    }

    # prompt user for inputs
    start = input('Enter start date (e.g. 2025-06-23): ')
    end   = input('Enter end date (e.g. 2025-06-23): ')
    print("Select a station:")
    print(" 1: KNYC\n 2: KMDW\n 3: KMIA\n 4: KAUS\n 5: KHOU\n 6: KDEN\n 7: KPHL\n 8: KLAX")
    choice = input('Enter station number (1–8): ').strip()

    entry = stations.get(choice)
    if not entry:
        print("Invalid choice; exiting.")
        exit(1)

    station_id, short_name = entry

    token = "kNbLWEOuHkssMJqLXGUOsCTAIrkpyqkd"
    df = daily_highs(station_id, start, end, token)
    print(df.head())

    # save to CSV using the station short name
    filename = f"{short_name}_highs.csv"
    df.to_csv(filename, index=False)
    print(f"Saved highs to {filename}")


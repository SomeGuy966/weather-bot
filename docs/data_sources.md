# Data sources and the settlement chain

Everything below was verified against the live services on 2026-09-12. The point of this
document is to record *what the contract actually pays on* and *which archive reproduces it*,
because the 2025 version of this project lost money on exactly these details.

## What Kalshi settles on

Every `KXHIGH*` market's `rules_primary` reads like:

> If the maximum temperature recorded at New York City (CLINYC) for Sep 11, 2026, is between
> 79-80° fahrenheit according to The Weather Company, then the market resolves to Yes.

The Weather Company's portal (weather.com/kalshi, "Daily Climate" tab) states its source:

> Daily values are official Fahrenheit climate-report figures (the value Kalshi settles on).
> Source: NWS/NOAA official Daily Climate Report (CLI), with WS Form F-6 (CF6) as backup.
> Values reflect the first official value published after the daily cutoff and are not
> restated on later NWS revision.

So the label is the **NWS CLI product's daily maximum**, for the station in the market
rules, first issue only. Two generations of series exist per city; the legacy ones
(`HIGHNY`, `HIGHCHI`, …, settled directly on NWS) are no longer served by the API. The live
series and their stations:

| series | station | CLI product | WFO | LST offset |
| :-- | :-- | :-- | :-- | --: |
| KXHIGHNY | KNYC Central Park | CLINYC | OKX | −5 |
| KXHIGHCHI | KMDW Midway | CLIMDW | LOT | −6 |
| KXHIGHMIA | KMIA | CLIMIA | MFL | −5 |
| KXHIGHAUS | KAUS | CLIAUS | EWX | −6 |
| KXHIGHTHOU | KHOU Hobby | CLIHOU | HGX | −6 |
| KXHIGHDEN | KDEN | CLIDEN | BOU | −7 |
| KXHIGHPHIL | KPHL | CLIPHL | PHI | −5 |
| KXHIGHLAX | KLAX | CLILAX | LOX | −8 |

## Which day

ASOS daily summaries, and therefore the CLI, run over the **local standard time** calendar
day all year. Kalshi's market close times confirm it: NYC markets close at 05:00Z, Chicago
06:00Z, Denver 07:00Z, Los Angeles 08:00Z — midnight LST in each case, which during daylight
time is 1 a.m. on the clock. The evening-before snapshots in the feature table start at
18:00 LST because that is roughly when the next day's market becomes liquid.

## How the official maximum is measured

From Daryl Herzmann's IEM note *Wagering on ASOS Temperatures* (IEM news #1469, Dec 2024):

* the official max/min are **2-minute averages of 2–5-second samples, in whole °F**;
* the sources with that fidelity are the 6-hourly METAR max/min groups, the ASOS Daily
  Summary Message, and the CLI/CF6;
* the METAR body temperature is whole °C and the `T` remark group is 0.1 °C, neither of
  which is the 2-minute-average °F value;
* the IEM 1-minute archive is delayed **12–36 hours** (collected by phone by NCEI) and is
  not 2-minute averaged, so it "does not match" manual max/min calculations.

Consequences for this project:

* The **CLI archive** at `mesonet.agron.iastate.edu/json/cli.py` is the label. It matched the
  values The Weather Company published for 2026-09-11 at all eight stations, and it has full
  coverage for all eight back to at least 2017.
* The **METAR archive** (`cgi-bin/request/asos.py`, routine + SPECI, raw report included) is
  the observation feed, for training and live alike — IEM ingests METARs in real time and was
  as current as `api.weather.gov` when checked. Rows tagged `IEM_GHCNH` are NCEI backfill that
  never existed in the live feed and are dropped. The running maximum derived from METARs
  equals the CLI high on 97.2 % of station-days (2020–2026), is one degree low on 1.7 %, two
  or more low on 0.7 %, and *high* on 0.5 % — the last being sensor faults that the NWS
  quality-controls out of the CLI.
* **1-minute data is not used.** It cannot be a live feature, and training on it would create
  train/serve skew. The original bullet point for this project said "1-minute observations";
  the honest description is "METAR observations (hourly plus specials)".

## Forecast guidance

The **MOS archive** (`cgi-bin/request/mos.py`) serves NBM text bulletins (`NBS`, four cycles a
day, stored at 01/07/13/19Z) back to 2019 and GFS MOS (`GFS`/MAV) back further. The NBM
`txn` field is the 12-hour max/min temperature with `xnd` its standard deviation; the
daytime maximum for a climate day is the `txn` valid at 00Z the following UTC day, which only
the 01Z and 07Z cycles carry (by 13Z the period has started). The 3-hourly `tmp` curve from
the latest cycle of any hour is used for the remaining-day maximum. NBM v3 bulletins (before
late 2020) lack `txn`; those rows fall back to GFS MOS `n_x`.

Availability lags are assumed conservatively: NBM bulletins one hour after the nominal run
time, GFS MOS three and a half hours.

## Kalshi

The public, unauthenticated endpoints of `api.elections.kalshi.com/trade-api/v2` provide
series and market metadata (including `rules_primary`, strikes and results), hourly
candlesticks with bid/ask open/close, volume and open interest, and the full trade tape.
Rate limit is about 10 requests/second. Contracts come in three shapes (`between`,
`greater`, `less`); the subtitle text is the source of truth and is cross-checked against
`floor_strike`/`cap_strike` — note that `T86` with floor 86 means "87° or above".

Only recent markets are exposed: the live series start on 2026-07-07 and the 2025-era
markets return 404. `weatherbot kalshi` is append-only and should be run daily.

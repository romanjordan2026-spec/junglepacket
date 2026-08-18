"""
flip-mini weather page
-----------------------
Fetches live pressure data from the same two NOAA CO-OPS stations as
holland_point_dashboard_v4.py (Port Angeles 9444090, Friday Harbor 9449880),
reproduces v4's exact default math (Hourly resample, Combined average across
stations, MA_WINDOW_HOURS=6 rolling smooth, delta = diff of smoothed series,
CAUTION_THRESHOLD = -1.5 hPa/hr), and writes a plain-text HTML page under
5KB with no JS, no images, no charts -- built for a keypad-navigated,
minimal-browser flip phone on a weak cell connection.

Run hourly by .github/workflows/weather.yml. Output: index.html (repo root),
served by GitHub Pages.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION_IDS = {
    "9444090": "Port Angeles, WA",
    "9449880": "Friday Harbor, WA",
}
MA_WINDOW_HOURS = 6          # same rolling-mean window as v4
CAUTION_THRESHOLD = -1.5     # same caution line as v4 (hPa/hr, smoothed-hourly basis)
LOOKBACK_HOURS = 48          # buffer well beyond the 6h smoothing window
OUTPUT_PATH = "index.html"


def fetch_station_pressure(station_id: str, begin: datetime, end: datetime) -> pd.DataFrame:
    """Same request shape as v4's _fetch_noaa_raw, air_pressure product only."""
    params = {
        "product": "air_pressure",
        "application": "holland_point_dashboard",
        "begin_date": begin.strftime("%Y%m%d %H:%M"),
        "end_date": end.strftime("%Y%m%d %H:%M"),
        "station": station_id,
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    resp = requests.get(NOAA_BASE, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload["data"])
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df[["t", "v"]].dropna()


def combined_hourly_pressure(begin: datetime, end: datetime) -> pd.Series:
    """Fetch both stations, resample each to hourly mean, average across
    whichever stations returned data -- matches v4's 'Combined average' /
    'Hourly' default view. Returns a Series indexed by hour (UTC)."""
    per_station = []
    for station_id in STATION_IDS:
        raw = fetch_station_pressure(station_id, begin, end)
        if raw.empty:
            continue
        hourly = raw.set_index("t")["v"].resample("1h").mean()
        per_station.append(hourly)

    if not per_station:
        raise RuntimeError("No pressure data returned from either NOAA station.")

    combined = pd.concat(per_station, axis=1).mean(axis=1)
    return combined.sort_index()


def compute_latest(pressure_hourly: pd.Series) -> dict:
    smooth = pressure_hourly.rolling(window=MA_WINDOW_HOURS, min_periods=1).mean()
    delta = smooth.diff()

    latest_ts = pressure_hourly.index[-1]
    latest_pressure = pressure_hourly.iloc[-1]
    latest_delta = delta.iloc[-1]

    caution = pd.notna(latest_delta) and latest_delta < CAUTION_THRESHOLD
    return {
        "ts": latest_ts,
        "pressure": latest_pressure,
        "delta": latest_delta,
        "caution": caution,
    }


def render_html(result: dict) -> str:
    ts_local = result["ts"].tz_convert("America/Vancouver")
    pressure = result["pressure"]
    delta = result["delta"]
    caution = result["caution"]

    if caution:
        status_line = "CAUTION: PRESSURE FALLING FAST"
    elif pd.notna(delta) and delta < CAUTION_THRESHOLD / 2:
        status_line = "Watch: pressure falling"
    else:
        status_line = "No active caution signal"

    delta_str = f"{delta:+.2f}" if pd.notna(delta) else "n/a"

    # Deliberately no CSS framework, no external font, no JS. Font size set
    # large and fixed for a fixed small viewport rather than responsive.
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Holland Point - mini</title>
</head>
<body style="font-family:sans-serif;font-size:20px;margin:8px;">
<b>{status_line}</b><br>
Pressure: {pressure:.1f} hPa<br>
Delta: {delta_str} hPa/hr<br>
As of: {ts_local.strftime('%b %d %H:%M %Z')}
</body></html>"""
    return html


def main() -> int:
    end = datetime.now(timezone.utc)
    begin = end - timedelta(hours=LOOKBACK_HOURS)

    try:
        pressure_hourly = combined_hourly_pressure(begin, end)
        result = compute_latest(pressure_hourly)
        html = render_html(result)
    except Exception as exc:  # noqa: BLE001 -- write a visible failure page, not a silent gap
        html = (
            "<!DOCTYPE html><html><body style=\"font-family:sans-serif;font-size:20px;margin:8px;\">"
            f"<b>Fetch failed</b><br>{type(exc).__name__}: {exc}"
            "</body></html>"
        )
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = len(html.encode("utf-8")) / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.2f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

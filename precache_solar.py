"""
precache_solar.py — Fast pre-fetch of 300 days solar data from NASA POWER
Saves to logs/solar_logs/solar_log.csv
"""

import csv
import os
import urllib.request
import json
from datetime import date, timedelta
import numpy as np

# Configuration
LAT = 10.88292448522716
LON = 106.80162959632759
PEAK_WM2 = 950.0
WM2_TO_LUX = 120.0
STEPS_PER_DAY = 144
PRECACHE_DAYS = 365
START_DATE = date(2025, 1, 1)
LOG_FILE = os.path.join("logs", "solar_logs", "solar_log.csv")

def fetch_day(target_date):
    """Fetch one day from NASA POWER."""
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": f"{LON:.4f}",
        "latitude": f"{LAT:.4f}",
        "start": target_date.strftime("%Y%m%d"),
        "end": target_date.strftime("%Y%m%d"),
        "format": "JSON",
        "temporal-api": "hourly",
    }
    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?{param_str}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        hourly = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        values = np.array(list(hourly.values()), dtype=np.float32)
        values = np.clip(values, 0.0, PEAK_WM2)
        # Interpolate to 144 steps
        wm2 = np.interp(np.arange(STEPS_PER_DAY) * 10, np.arange(24) * 60, values).astype(np.float32)
        return wm2
    except Exception as e:
        print(f"  Failed {target_date}: {e}")
        return None

def fallback(target_date):
    """Astronomical fallback."""
    doy = target_date.timetuple().tm_yday
    lat_rad = np.radians(LAT)
    dec_rad = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    cos_ha = np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1.0, 1.0)
    ha_deg = np.degrees(np.arccos(cos_ha))
    b = np.radians(360 / 365 * (doy - 81))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_noon_min = 720 - eot - (LON - 105.0) * 4
    sunrise = solar_noon_min - ha_deg * 4
    sunset = solar_noon_min + ha_deg * 4
    steps = np.arange(STEPS_PER_DAY, dtype=np.float32) * 10
    wm2 = np.zeros(STEPS_PER_DAY, dtype=np.float32)
    day_mask = (steps >= sunrise) & (steps <= sunset)
    if day_mask.any():
        angle = np.pi * (steps[day_mask] - sunrise) / (sunset - sunrise)
        wm2[day_mask] = PEAK_WM2 * 0.75 * np.sin(angle).astype(np.float32)
    return wm2

def main():
    print("=" * 60)
    print("  NASA POWER Solar Pre-cache (300 days)")
    print(f"  Location: {LAT:.4f}, {LON:.4f}")
    print(f"  Output: {LOG_FILE}")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Collect all data first, then write once
    all_rows = []
    
    for i in range(PRECACHE_DAYS):
        d = START_DATE + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        
        wm2 = fetch_day(d)
        if wm2 is None:
            wm2 = fallback(d)
        
        peak = wm2.max()
        print(f"[{i+1:3d}/300] {date_str} - Peak: {peak:6.1f} W/m²")
        
        # Generate 144 rows for this day
        for step in range(STEPS_PER_DAY):
            minutes = step * 10
            hours = minutes // 60
            mins = minutes % 60
            tod = minutes / (24 * 60)
            lux = wm2[step] * WM2_TO_LUX
            solar_norm = (wm2[step] / PEAK_WM2) * 0.08
            
            all_rows.append([
                i + 1, date_str, step,
                f"{tod:.6f}", f"{hours:02d}:{mins:02d}",
                f"{lux:.2f}", f"{wm2[step]:.4f}", f"{solar_norm:.6f}"
            ])
        
        if (i + 1) % 50 == 0:
            print(f"  → {i+1}/300 collected...")
    
    # Write all at once
    print(f"\nWriting {len(all_rows)} rows to {LOG_FILE}...")
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "date", "step", "time_of_day", "hour_min", "lux", "wm2", "solar_norm"])
        writer.writerows(all_rows)
    
    print("\n" + "=" * 60)
    print(f"  Complete! 300 days cached ({len(all_rows)} rows)")
    print(f"  File size: {os.path.getsize(LOG_FILE) / 1024 / 1024:.1f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()

import requests
import pandas as pd
from io import StringIO

url = "https://data.cityofchicago.org/resource/6dvr-xwnh.csv"
all_chunks = []
offset = 0
chunk_size = 50000

while True:
    params = {
        "$where": "trip_start_timestamp >= '2025-01-01T00:00:00' AND trip_start_timestamp < '2026-01-01T00:00:00'",
        "$limit": chunk_size,
        "$offset": offset,
        "$order": "trip_start_timestamp ASC",
        "$$app_token": "YOUR_APP_TOKEN_HERE"
    }
    
    response = requests.get(url, params=params, timeout=120)
    chunk_df = pd.read_csv(StringIO(response.text))
    
    if len(chunk_df) == 0:
        break
    
    all_chunks.append(chunk_df)
    offset += chunk_size
    print(f"Downloaded {offset:,} rows so far...")

results_df = pd.concat(all_chunks, ignore_index=True)
results_df.to_csv("data/raw/tnc_trips.csv", index=False)
print(f"Done — {len(results_df):,} total rows saved")
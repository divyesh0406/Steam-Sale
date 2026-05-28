"""Check columns in eda_treatment_control.parquet and stg.games."""
import os, sys
from pathlib import Path
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

tc = pd.read_parquet(Path("results/eda_treatment_control.parquet"))
print("=== eda_treatment_control.parquet ===")
print(f"Shape: {tc.shape}")
print(f"Columns: {tc.columns.tolist()}")
print(tc.dtypes)
print()

conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='stg' AND table_name='games' ORDER BY ordinal_position")
cols = [r[0] for r in cur.fetchall()]
print(f"=== stg.games columns ===\n{cols}")
conn.close()

"""Truncate fct_reviews_daily to free up space, then VACUUM."""
import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM mart.fct_reviews_daily")
before = cur.fetchone()[0]
print(f"Rows before: {before:,}")
cur.execute("TRUNCATE mart.fct_reviews_daily")
cur.execute("SELECT COUNT(*) FROM mart.fct_reviews_daily")
after = cur.fetchone()[0]
print(f"Rows after:  {after:,}")
cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
print(f"DB size after truncate: {cur.fetchone()[0]}")
conn.close()

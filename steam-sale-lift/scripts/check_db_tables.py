import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10)
rows = conn.execute(
    "SELECT table_schema, table_name FROM information_schema.tables "
    "WHERE table_schema IN ('stg','mart') ORDER BY 1,2"
).fetchall()
for r in rows:
    n = conn.execute(f"SELECT COUNT(*) FROM {r[0]}.{r[1]}").fetchone()[0]
    print(f"  {r[0]}.{r[1]}: {n:,} rows")
conn.close()

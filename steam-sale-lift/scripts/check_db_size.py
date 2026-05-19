import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
    SELECT schemaname, tablename,
           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
           pg_total_relation_size(schemaname||'.'||tablename) AS bytes
    FROM pg_tables
    WHERE schemaname IN ('mart','raw','stg')
    ORDER BY bytes DESC
""")
print(f"{'Table':<40} {'Size':>10}")
print("-" * 52)
total = 0
for r in cur.fetchall():
    print(f"{r[0]+'.'+r[1]:<40} {r[2]:>10}")
    total += r[3]

cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
print(f"\nTotal DB size: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM mart.fct_reviews_daily")
print(f"fct_reviews_daily rows: {cur.fetchone()[0]:,}")
conn.close()

import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
    SELECT DATE_TRUNC('month', review_date)::DATE AS month,
           SUM(review_count) AS reviews,
           COUNT(DISTINCT appid) AS games
    FROM mart.fct_reviews_daily
    WHERE review_date BETWEEN '2025-01-01' AND '2026-06-01'
    GROUP BY 1 ORDER BY 1
""")
print("Month         Reviews   Games")
for r in cur.fetchall():
    print(f"{r[0]}   {r[1]:>8,}   {r[2]:>5,}")
conn.close()

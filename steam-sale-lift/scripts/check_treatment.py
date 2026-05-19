import os, psycopg, pandas as pd, datetime as dt
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])

WINTER_START = dt.date(2024, 12, 19)
WINTER_END   = dt.date(2025,  1,  2)
PRE_START    = dt.date(2024, 11, 19)
PRE_END      = dt.date(2024, 12, 18)

rw = pd.read_sql("""
    SELECT r.appid,
        SUM(CASE WHEN r.review_date BETWEEN %(pre_start)s AND %(pre_end)s
                 THEN r.review_count ELSE 0 END) AS pre,
        SUM(CASE WHEN r.review_date BETWEEN %(sale_start)s AND %(sale_end)s
                 THEN r.review_count ELSE 0 END) AS sale
    FROM mart.fct_reviews_daily r
    JOIN stg.games g USING (appid)
    WHERE g.base_price_cents >= 500
    GROUP BY r.appid
""", conn, params={"pre_start": PRE_START, "pre_end": PRE_END,
                   "sale_start": WINTER_START, "sale_end": WINTER_END})

print(f"Games with review data: {len(rw):,}")
print(f"Pre-period  — mean: {rw['pre'].mean():.1f}, median: {rw['pre'].median():.0f}, max: {rw['pre'].max():.0f}")
print(f"Sale-period — mean: {rw['sale'].mean():.1f}, median: {rw['sale'].median():.0f}, max: {rw['sale'].max():.0f}")
print(f"Games with 0 pre reviews:  {(rw['pre']==0).sum():,}")
print(f"Games with 0 sale reviews: {(rw['sale']==0).sum():,}")

# Distribution of sale review counts
print("\nSale review count distribution:")
for threshold in [0, 1, 5, 10, 50, 100]:
    n = (rw["sale"] > threshold).sum()
    print(f"  > {threshold:>4}: {n:,} ({n/len(rw):.1%})")

conn.close()

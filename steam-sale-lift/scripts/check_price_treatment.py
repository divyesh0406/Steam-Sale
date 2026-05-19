"""Check how many games have price records for winter_2024."""
import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
    SELECT
        COUNT(DISTINCT p.appid) AS treated,
        (SELECT COUNT(*) FROM stg.games WHERE base_price_cents >= 500) AS total_paid
    FROM mart.fct_prices_daily p
    JOIN stg.games g USING (appid)
    WHERE p.sale_event_id = 'winter_2024'
      AND g.base_price_cents >= 500
""")
r = cur.fetchone()
print(f"Treated (discounted in Winter 2024): {r[0]:,}")
print(f"Total paid games:                    {r[1]:,}")
print(f"Control (paid, not discounted):      {r[1]-r[0]:,}")
print(f"Treatment rate: {r[0]/r[1]:.1%}")

# Also check discount depths for treated games
cur.execute("""
    SELECT
        ROUND(AVG(p.discount_pct)) AS avg_discount,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.discount_pct)::NUMERIC) AS median_discount,
        MIN(p.discount_pct) AS min_discount,
        MAX(p.discount_pct) AS max_discount
    FROM mart.fct_prices_daily p
    JOIN stg.games g USING (appid)
    WHERE p.sale_event_id = 'winter_2024'
      AND g.base_price_cents >= 500
      AND p.discount_pct > 0
""")
r = cur.fetchone()
print(f"\nDiscount depth for treated games:")
print(f"  Avg: {r[0]}%  Median: {r[1]}%  Range: {r[2]}–{r[3]}%")
conn.close()

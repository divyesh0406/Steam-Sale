"""Check SteamSpy review totals available for synthetic review generation."""
import os, psycopg
from dotenv import load_dotenv
load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
    SELECT
        COUNT(*) AS games,
        COUNT(CASE WHEN positive + negative > 0 THEN 1 END) AS with_reviews,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY positive + negative) AS median_total,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY positive + negative) AS p90_total,
        MAX(positive + negative) AS max_total
    FROM mart.fct_steamspy
""")
r = cur.fetchone()
print(f"Games total:           {r[0]:,}")
print(f"With review data:      {r[1]:,}")
print(f"Median lifetime total: {r[2]:,.0f}")
print(f"P90 lifetime total:    {r[3]:,.0f}")
print(f"Max lifetime total:    {r[4]:,}")

# Age distribution
cur.execute("""
    SELECT
        COUNT(*) AS games,
        AVG(CURRENT_DATE - release_date) AS avg_age_days,
        COUNT(CASE WHEN release_date IS NOT NULL THEN 1 END) AS with_release_date
    FROM mart.dim_games
""")
r = cur.fetchone()
print(f"\nGames with release date: {r[2]:,}")
print(f"Average age (days):      {r[1]:.0f}")
conn.close()

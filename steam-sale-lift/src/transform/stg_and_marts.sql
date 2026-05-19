-- Phase 2: Staging views + analytics mart views
CREATE SCHEMA IF NOT EXISTS stg;
-- Run via: psql $DATABASE_URL -f src/transform/stg_and_marts.sql
-- All objects are CREATE OR REPLACE / IF NOT EXISTS so re-running is safe.

-- ─────────────────────────────────────────────────────────────────────────────
-- STG layer — light cleaning on top of mart base tables
-- ─────────────────────────────────────────────────────────────────────────────

-- Games with nulls filled and derived flags validated
CREATE OR REPLACE VIEW stg.games AS
SELECT
    appid,
    name,
    release_date,
    COALESCE(developer, 'Unknown')              AS developer,
    COALESCE(publisher, 'Unknown')              AS publisher,
    base_price_cents,
    ROUND(base_price_cents / 100.0, 2)          AS base_price_usd,
    COALESCE(is_indie, FALSE)                   AS is_indie,
    COALESCE(is_aaa, FALSE)                     AS is_aaa,
    -- Games that are neither indie nor AAA (mid-tier)
    NOT COALESCE(is_indie, FALSE)
        AND NOT COALESCE(is_aaa, FALSE)         AS is_midtier,
    COALESCE(primary_genre, 'Unknown')          AS primary_genre,
    COALESCE(genres, ARRAY[]::TEXT[])           AS genres,
    owners_lower,
    owners_upper,
    -- Midpoint estimate for owner counts
    ROUND((owners_lower + owners_upper) / 2.0)  AS owners_mid,
    -- Age of game in days at time of analysis
    CURRENT_DATE - release_date                 AS age_days
FROM mart.dim_games
WHERE name IS NOT NULL;


-- Sale events with duration derived
CREATE OR REPLACE VIEW stg.sale_events AS
SELECT
    sale_event_id,
    name,
    start_date,
    end_date,
    sale_type,
    end_date - start_date + 1                   AS duration_days,
    EXTRACT(YEAR FROM start_date)::INT          AS sale_year,
    EXTRACT(MONTH FROM start_date)::INT         AS sale_month
FROM mart.dim_sale_events;


-- Prices with participation flag clarified
-- NOTE: because we use synthetic price history, sale_event_id presence
-- indicates the game existed during a sale window, not confirmed participation.
-- Use mart.fct_steamspy.discount > 0 for confirmed current discounts.
CREATE OR REPLACE VIEW stg.prices AS
SELECT
    p.appid,
    p.date,
    p.price_cents,
    ROUND(p.price_cents / 100.0, 2)             AS price_usd,
    p.discount_pct,
    p.is_on_sale,
    p.sale_event_id,
    s.sale_type,
    s.start_date                                AS sale_start,
    s.end_date                                  AS sale_end,
    -- Window position: days relative to sale start (negative = pre, positive = post)
    p.date - s.start_date                       AS days_from_sale_start
FROM mart.fct_prices_daily p
LEFT JOIN mart.dim_sale_events s USING (sale_event_id);


-- Review velocity cleaned
CREATE OR REPLACE VIEW stg.review_velocity AS
SELECT
    appid,
    review_date,
    review_count,
    positive_count,
    review_count - positive_count               AS negative_count,
    ROUND(positive_count::NUMERIC
          / NULLIF(review_count, 0), 4)         AS positive_rate,
    avg_playtime_min
FROM mart.fct_reviews_daily
WHERE review_count > 0;


-- Player history with month parsed to a proper date
CREATE OR REPLACE VIEW stg.players AS
SELECT
    appid,
    year_month,
    -- Convert "April 2024" → first day of that month
    TO_DATE(year_month, 'Month YYYY')           AS month_start_date,
    ROUND(avg_players)::INT                     AS avg_players,
    peak_players
FROM mart.fct_players_monthly
WHERE (avg_players IS NOT NULL OR peak_players IS NOT NULL)
  AND year_month ~ '^\w+ \d{4}$';  -- exclude "Last 30 Days" and similar non-date strings


-- ─────────────────────────────────────────────────────────────────────────────
-- MART analytics views — what the notebooks query directly
-- ─────────────────────────────────────────────────────────────────────────────

-- Core DiD panel: one row per game × sale event
-- treatment = participated in that sale (proxied by SteamSpy current discount
-- for summer_2024; for historical sales all games are assumed potential participants)
CREATE OR REPLACE VIEW mart.v_did_panel AS
WITH sale_review_windows AS (
    SELECT
        g.appid,
        g.name,
        g.primary_genre,
        g.is_indie,
        g.is_aaa,
        g.base_price_cents,
        g.owners_mid,
        s.sale_event_id,
        s.sale_type,
        s.start_date,
        s.end_date,
        -- Pre-period: 30 days before sale
        s.start_date - INTERVAL '30 days'       AS pre_start,
        s.start_date - INTERVAL '1 day'         AS pre_end,
        -- Post-period: 30 days after sale
        s.end_date + INTERVAL '1 day'           AS post_start,
        s.end_date + INTERVAL '30 days'         AS post_end
    FROM stg.games g
    CROSS JOIN stg.sale_events s
    WHERE g.base_price_cents IS NOT NULL
      AND g.base_price_cents > 0
      AND g.release_date < s.start_date        -- game existed before the sale
),
pre_reviews AS (
    SELECT
        sw.appid,
        sw.sale_event_id,
        COALESCE(SUM(r.review_count), 0)        AS pre_review_count,
        COALESCE(AVG(r.positive_rate), NULL)    AS pre_positive_rate
    FROM sale_review_windows sw
    LEFT JOIN stg.review_velocity r
        ON r.appid = sw.appid
       AND r.review_date BETWEEN sw.pre_start AND sw.pre_end
    GROUP BY sw.appid, sw.sale_event_id
),
post_reviews AS (
    SELECT
        sw.appid,
        sw.sale_event_id,
        COALESCE(SUM(r.review_count), 0)        AS post_review_count,
        COALESCE(AVG(r.positive_rate), NULL)    AS post_positive_rate
    FROM sale_review_windows sw
    LEFT JOIN stg.review_velocity r
        ON r.appid = sw.appid
       AND r.review_date BETWEEN sw.post_start AND sw.post_end
    GROUP BY sw.appid, sw.sale_event_id
)
SELECT
    sw.appid,
    sw.name,
    sw.primary_genre,
    sw.is_indie,
    sw.is_aaa,
    sw.base_price_cents,
    sw.owners_mid,
    sw.sale_event_id,
    sw.sale_type,
    sw.start_date,
    sw.end_date,
    pre.pre_review_count,
    pre.pre_positive_rate,
    post.post_review_count,
    post.post_positive_rate,
    -- Outcome: change in review velocity (proxy for demand lift)
    post.post_review_count - pre.pre_review_count   AS review_count_lift,
    post.post_positive_rate - pre.pre_positive_rate AS positive_rate_change,
    -- Log outcome for DiD regression (adds 1 to handle zeros)
    LN(post.post_review_count + 1)
        - LN(pre.pre_review_count + 1)              AS log_review_lift
FROM sale_review_windows sw
JOIN pre_reviews  pre  USING (appid, sale_event_id)
JOIN post_reviews post USING (appid, sale_event_id);


-- RDD view: one row per game × sale event, with discount depth as running variable
-- Used for H2: effect of crossing the 50% discount threshold on review velocity
CREATE OR REPLACE VIEW mart.v_rdd_discount AS
SELECT
    d.appid,
    g.name,
    g.primary_genre,
    g.is_indie,
    g.is_aaa,
    g.base_price_cents,
    d.sale_event_id,
    sp.discount                                 AS current_discount_pct,
    -- Reviews in 14 days after sale start (outcome variable)
    COALESCE((
        SELECT SUM(r.review_count)
        FROM stg.review_velocity r
        JOIN mart.dim_sale_events se
            ON se.sale_event_id = d.sale_event_id
        WHERE r.appid = d.appid
          AND r.review_date BETWEEN se.start_date
                                AND se.start_date + INTERVAL '14 days'
    ), 0)                                       AS reviews_14d_post,
    -- Reviews in 14 days before sale (baseline)
    COALESCE((
        SELECT SUM(r.review_count)
        FROM stg.review_velocity r
        JOIN mart.dim_sale_events se
            ON se.sale_event_id = d.sale_event_id
        WHERE r.appid = d.appid
          AND r.review_date BETWEEN se.start_date - INTERVAL '14 days'
                                AND se.start_date - INTERVAL '1 day'
    ), 0)                                       AS reviews_14d_pre
FROM mart.fct_prices_daily d
JOIN stg.games g USING (appid)
LEFT JOIN mart.fct_steamspy sp USING (appid)
WHERE d.sale_event_id IS NOT NULL
  AND sp.discount IS NOT NULL
  AND sp.discount > 0;


-- Heterogeneity summary: avg review lift by genre × sale type
CREATE OR REPLACE VIEW mart.v_genre_heterogeneity AS
SELECT
    primary_genre,
    is_indie,
    is_aaa,
    sale_type,
    COUNT(*)                                    AS n_games,
    ROUND(AVG(review_count_lift)::NUMERIC, 2)            AS avg_review_lift,
    ROUND(STDDEV(review_count_lift)::NUMERIC, 2)         AS sd_review_lift,
    ROUND(AVG(log_review_lift)::NUMERIC, 4)              AS avg_log_review_lift,
    ROUND(AVG(pre_review_count)::NUMERIC, 1)             AS avg_pre_reviews,
    ROUND(AVG(post_review_count)::NUMERIC, 1)            AS avg_post_reviews
FROM mart.v_did_panel
GROUP BY primary_genre, is_indie, is_aaa, sale_type
HAVING COUNT(*) >= 10
ORDER BY avg_log_review_lift DESC;


-- Game-level summary for the Streamlit explorer
CREATE OR REPLACE VIEW mart.v_game_summary AS
SELECT
    g.appid,
    g.name,
    g.primary_genre,
    g.is_indie,
    g.is_aaa,
    g.base_price_cents,
    ROUND(g.base_price_cents / 100.0, 2)        AS base_price_usd,
    g.owners_mid,
    g.release_date,
    sp.discount                                 AS current_discount_pct,
    sp.positive                                 AS total_positive_reviews,
    sp.negative                                 AS total_negative_reviews,
    ROUND(sp.positive::NUMERIC
          / NULLIF(sp.positive + sp.negative, 0) * 100, 1)
                                                AS overall_positive_rate_pct,
    sp.average_playtime_2weeks,
    -- Latest monthly player count
    (SELECT avg_players FROM stg.players p
     WHERE p.appid = g.appid
     ORDER BY month_start_date DESC LIMIT 1)    AS latest_avg_players
FROM stg.games g
LEFT JOIN mart.fct_steamspy sp USING (appid);

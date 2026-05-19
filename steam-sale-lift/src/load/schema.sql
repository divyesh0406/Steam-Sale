-- Steam Sale Lift — warehouse schema
-- Run once against your Neon Postgres database.
-- All tables use ON CONFLICT DO UPDATE (upsert) during ingest, so re-running is safe.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS mart;

-- ─────────────────────────────────────────────────────────────────────────────
-- RAW layer — minimal parsing, close to source
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw.app_list (
    appid       BIGINT PRIMARY KEY,
    name        TEXT,
    owners_raw  TEXT,       -- SteamSpy range string: "200000 .. 500000"
    positive    INT,
    negative    INT,
    scraped_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- MART layer — analytics-ready star schema
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mart.dim_sale_events (
    sale_event_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    sale_type       TEXT NOT NULL  -- 'major', 'minor'
);

CREATE TABLE IF NOT EXISTS mart.dim_games (
    appid               BIGINT PRIMARY KEY,
    name                TEXT NOT NULL,
    release_date        DATE,
    developer           TEXT,
    publisher           TEXT,
    base_price_cents    INT,        -- price in US cents at launch / before first sale
    is_indie            BOOLEAN,
    is_aaa              BOOLEAN,
    primary_genre       TEXT,
    genres              TEXT[],
    tags                TEXT[],
    owners_lower        INT,        -- from SteamSpy range lower bound
    owners_upper        INT,        -- from SteamSpy range upper bound
    scraped_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mart.fct_prices_daily (
    appid           BIGINT  REFERENCES mart.dim_games(appid),
    date            DATE    NOT NULL,
    price_cents     INT,
    discount_pct    NUMERIC(5,2),   -- 0.00 = full price, 50.00 = 50% off
    is_on_sale      BOOLEAN GENERATED ALWAYS AS (discount_pct > 0) STORED,
    sale_event_id   TEXT    REFERENCES mart.dim_sale_events(sale_event_id),
    PRIMARY KEY (appid, date)
);

-- Daily aggregate review counts (replaces per-review table to fit 512 MB limit)
CREATE TABLE IF NOT EXISTS mart.fct_reviews_daily (
    appid               BIGINT  REFERENCES mart.dim_games(appid),
    review_date         DATE    NOT NULL,
    review_count        INT     NOT NULL,
    positive_count      INT     NOT NULL,
    avg_playtime_min    INT,
    PRIMARY KEY (appid, review_date)
);

CREATE TABLE IF NOT EXISTS mart.fct_players_monthly (
    appid           BIGINT  REFERENCES mart.dim_games(appid),
    year_month      TEXT    NOT NULL,   -- 'April 2024'
    avg_players     NUMERIC(12,2),
    peak_players    INT,
    PRIMARY KEY (appid, year_month)
);

CREATE TABLE IF NOT EXISTS mart.fct_steamspy (
    appid           BIGINT  PRIMARY KEY REFERENCES mart.dim_games(appid),
    owners_lower    INT,
    owners_upper    INT,
    positive        INT,
    negative        INT,
    discount        INT,    -- current discount % at scrape time (0 = full price)
    average_playtime_2weeks INT,
    median_playtime_2weeks  INT,
    scraped_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_prices_appid_date
    ON mart.fct_prices_daily(appid, date);

CREATE INDEX IF NOT EXISTS idx_prices_sale_event
    ON mart.fct_prices_daily(sale_event_id)
    WHERE sale_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_reviews_appid_date
    ON mart.fct_reviews_daily(appid, review_date);

CREATE INDEX IF NOT EXISTS idx_players_appid_month
    ON mart.fct_players_monthly(appid, year_month);

-- ─────────────────────────────────────────────────────────────────────────────
-- Analytics views
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW mart.v_game_sale_windows AS
SELECT
    g.appid,
    g.name,
    g.primary_genre,
    g.is_indie,
    g.is_aaa,
    s.sale_event_id,
    s.name           AS sale_name,
    s.sale_type,
    s.start_date,
    s.end_date,
    s.start_date - INTERVAL '30 days' AS pre_window_start,
    s.end_date   + INTERVAL '30 days' AS post_window_end,
    EXISTS (
        SELECT 1
        FROM mart.fct_prices_daily p
        WHERE p.appid          = g.appid
          AND p.sale_event_id  = s.sale_event_id
          AND p.is_on_sale
    ) AS participated
FROM mart.dim_games g
CROSS JOIN mart.dim_sale_events s;


CREATE OR REPLACE VIEW mart.v_review_velocity AS
-- Daily review counts per game — useful for DiD and RDD outcomes
SELECT
    appid,
    review_date                                              AS date,
    review_count,
    positive_count,
    ROUND(positive_count::numeric / NULLIF(review_count, 0), 4) AS positive_rate,
    avg_playtime_min                                         AS avg_playtime_at_review
FROM mart.fct_reviews_daily;


CREATE OR REPLACE VIEW mart.v_discount_depth AS
-- One row per game × sale event: max discount seen during that event
SELECT
    p.appid,
    p.sale_event_id,
    MAX(p.discount_pct)                          AS max_discount_pct,
    MIN(p.price_cents)                           AS min_price_cents,
    COUNT(*)                                     AS days_on_sale,
    MIN(p.date)                                  AS first_sale_date
FROM mart.fct_prices_daily p
WHERE p.sale_event_id IS NOT NULL
  AND p.is_on_sale
GROUP BY p.appid, p.sale_event_id;

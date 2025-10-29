import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import date
from typing import Optional, Dict, Any, List, Tuple
import math
import traceback
import datetime
import re, calendar, datetime as dt

import psycopg2
import psycopg2.extras
from psycopg2.extras import DictCursor

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ==== Optional (AI) ====
try:
    from openai import OpenAI  # installed via requirements.txt
except Exception:
    OpenAI = None  # handled later

# =========================
# Config / Environment
# =========================
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "cmci")
DB_USER = os.getenv("DB_USER", "doku")
DB_PASS = os.getenv("DB_PASS", "doku")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

def conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT,
        connect_timeout=5,
        options="-c search_path=public"
    )

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")  # mount to host via docker-compose
START_TIME = time.time()

# =========================
# Logging setup (console + daily rotating file)
# =========================
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "app.log")

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(fmt)
root_logger.addHandler(ch)

fh = TimedRotatingFileHandler(log_file, when="midnight", backupCount=14, encoding="utf-8")
fh.setFormatter(fmt)
root_logger.addHandler(fh)

logger = logging.getLogger("cmci")
logger.info("Logger initialized. Level=%s, File=%s", LOG_LEVEL, log_file)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="CMCI API (Seasonal + Holidays + AI + Logging)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


import re
from datetime import datetime, timezone

# normalisasi nama channel
_CHANNEL_ALIASES = {
    "qris": "QRIS", "gopay": "QRIS",  # boleh kustomisasi
    "va": "VA", "virtual account": "VA", "bank": "VA",
    "cc": "CC", "credit": "CC", "kartu kredit": "CC",
}

_ID_RE   = re.compile(r"(clientid|client|merchant|mid)\s*[:=]?\s*(\d+)", re.I)
_CH_RE   = re.compile(r"(channel|payment|metode)\s*[:=]?\s*([a-zA-Z +]+)", re.I)
_MON_RE1 = re.compile(r"(jan|feb|mar|apr|mei|may|jun|jul|aug|agu|sep|oct|okt|nov|dec|des)[a-z]*\s*(\d{4})?", re.I)
_MON_RE2 = re.compile(r"bulan\s+(jan|feb|mar|apr|mei|may|jun|jul|aug|agu|sep|oct|okt|nov|dec|des)[a-z]*", re.I)
_MON_RE3 = re.compile(r"(september|oktober|november|desember|januari|februari|maret|april|mei|juni|juli|agustus)", re.I)
_YR_RE   = re.compile(r"(20\d{2})")

def _norm_channel(txt: str|None) -> str|None:
    if not txt: return None
    t = txt.strip().lower()
    t = t.replace("payment channel", "").replace("channel", "").strip()
    t = t.replace("+", " ")
    # ambil kata pertama relevan
    for word in re.split(r"[^a-z]+", t):
        if not word: continue
        if word in _CHANNEL_ALIASES:
            return _CHANNEL_ALIASES[word]
        if word.upper() in ("QRIS","VA","CC"):
            return word.upper()
    return None

# map bulan indo/eng → angka
_MONTHS = {
    # id
    "januari":1,"februari":2,"maret":3,"april":4,"mei":5,"juni":6,"juli":7,"agustus":8,"september":9,"oktober":10,"november":11,"desember":12,
    # en short/var
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"mei":5,"jun":6,"jul":7,"aug":8,"agu":8,"sep":9,"oct":10,"okt":10,"nov":11,"dec":12,"des":12,
}

def _month_to_range(mon: int, year: int):
    start = datetime(year, mon, 1, tzinfo=timezone.utc).date()
    if mon == 12:
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc).date()
    else:
        end = datetime(year, mon+1, 1, tzinfo=timezone.utc).date()
    return start, end

def parse_user_query(q: str) -> dict:
    q = q.strip()

    # merchant/client id
    m = _ID_RE.search(q)
    client_id = int(m.group(2)) if m else None

    # channel (optional)
    ch = None
    m = _CH_RE.search(q)
    if m:
        ch = _norm_channel(m.group(2))

    # bulan & tahun (default: bulan berjalan)
    now = datetime.now(timezone.utc)
    mon = now.month
    yr  = now.year

    for rx in ( _MON_RE1, _MON_RE2, _MON_RE3 ):
        m = rx.search(q)
        if m:
            token = (m.group(1) or "").lower()
            if token in _MONTHS:
                mon = _MONTHS[token]
            # coba cari tahun di dekatnya
            my = _YR_RE.search(q)
            if my:
                yr = int(my.group(1))
            break

    start, end = _month_to_range(mon, yr)
    return {"merchant_id": client_id, "channel": ch, "start": start, "end": end}


def parse_params(msg: str):
    msg_low = msg.lower()
    # clientid
    m = re.search(r'client\s*id\s*(\d+)|clientid\s*(\d+)', msg_low)
    client_id = int(next(g for g in (m.group(1), m.group(2)) if g)) if m else None

    # channel
    ch = None
    for k in CHANNEL_MAP:
        if k.lower() in msg_low:
            ch = CHANNEL_MAP[k]; break
    ch = ch or "QRIS"

    # bulan & tahun
    bln = None
    for nama, num in BULAN_ID.items():
        if nama in msg_low: bln = num; break
    today = dt.date.today()
    tahun = today.year
    if "tahun " in msg_low:
        mt = re.search(r'tahun\s*(\d{4})', msg_low)
        if mt: tahun = int(mt.group(1))
    if bln is None:
        bln = today.month  # default: bulan ini

    # rentang tanggal bulan tsb
    start = dt.date(tahun, bln, 1)
    last_day = calendar.monthrange(tahun, bln)[1]
    end = dt.date(tahun, bln, last_day)
    return client_id, ch, start, end

def _sr_sql_and_params(merchant_id: int, start, end, channel: str|None):
    # SR = sukses/total, sukses = status 'SALE'
    if channel:
        sql = """
        WITH f AS (
          SELECT status
          FROM fact_tx
          WHERE merchant_id = %s
            AND paid_at >= %s AND paid_at < %s
            AND channel = %s
        )
        SELECT
          COUNT(*)                              AS total,
          COUNT(*) FILTER (WHERE status='SALE') AS success,
          ROUND(100.0 * COUNT(*) FILTER (WHERE status='SALE') / NULLIF(COUNT(*),0), 2) AS sr
        FROM f;
        """
        params = (merchant_id, start, end, channel)
        return sql, params

    # tanpa channel → overall + breakdown per-channel
    sql_overall = """
      WITH f AS (
        SELECT status
        FROM fact_tx
        WHERE merchant_id = %s
          AND paid_at >= %s AND paid_at < %s
      )
      SELECT
        COUNT(*)                              AS total,
        COUNT(*) FILTER (WHERE status='SALE') AS success,
        ROUND(100.0 * COUNT(*) FILTER (WHERE status='SALE') / NULLIF(COUNT(*),0), 2) AS sr
      FROM f;
    """
    sql_by_ch = """
      SELECT channel,
             COUNT(*)                              AS total,
             COUNT(*) FILTER (WHERE status='SALE') AS success,
             ROUND(100.0 * COUNT(*) FILTER (WHERE status='SALE') / NULLIF(COUNT(*),0), 2) AS sr
      FROM fact_tx
      WHERE merchant_id = %s
        AND paid_at >= %s AND paid_at < %s
      GROUP BY channel
      ORDER BY sr DESC NULLS LAST, channel;
    """
    return (sql_overall, (merchant_id, start, end)), (sql_by_ch, (merchant_id, start, end))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    try:
        response = await call_next(request)
        dur_ms = (time.time() - start) * 1000
        logger.info("HTTP %s %s -> %s (%.1f ms)", request.method, path, response.status_code, dur_ms)
        return response
    except Exception:
        dur_ms = (time.time() - start) * 1000
        logger.exception("Unhandled error at %s %s (%.1f ms)", request.method, path, dur_ms)
        raise

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    logger.exception("UNHANDLED: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "hint": "Cek container logs untuk detail."},
    )

# =========================
# DB helper (compat)
# =========================
def db():
    return conn()

# =========================
# Seasons: table + seeding + helpers
# =========================
DDL_SEASON = """
CREATE TABLE IF NOT EXISTS dim_season (
  id SERIAL PRIMARY KEY,
  season_name VARCHAR(50),
  start_date DATE,
  end_date DATE
);
"""

DEFAULT_SEASONS = [
    ("Lebaran", "2025-03-15", "2025-04-30"),
    ("Libur Sekolah", "2025-06-15", "2025-07-15"),
    ("Natal & Tahun Baru", "2025-12-15", "2026-01-05"),
]

def ensure_season_table_and_seed():
    con = db(); cur = con.cursor()
    cur.execute(DDL_SEASON); con.commit()
    cur.execute("SELECT COUNT(*) FROM dim_season;")
    cnt = cur.fetchone()[0]
    if cnt == 0:
        cur.executemany(
            "INSERT INTO dim_season (season_name, start_date, end_date) VALUES (%s,%s,%s)",
            DEFAULT_SEASONS
        )
        con.commit()
        logger.info("dim_season seeded with default rows")
    cur.close(); con.close()

def current_season_row() -> Optional[Dict[str, Any]]:
    ensure_season_table_and_seed()
    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT season_name, start_date, end_date
        FROM dim_season
        WHERE %s BETWEEN start_date AND end_date
        ORDER BY start_date DESC
        LIMIT 1
    """, (date.today(),))
    row = cur.fetchone()
    cur.close(); con.close()
    return dict(row) if row else None

@app.get("/seasons")
def seasons():
    ensure_season_table_and_seed()
    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT season_name, start_date, end_date FROM dim_season ORDER BY start_date;")
    rows = cur.fetchall()
    cur.close(); con.close()
    return {"today": str(date.today()), "current": current_season_row(), "seasons": rows}

# =========================
# Holidays: table + seeding + helpers (Indonesia 2025)
# =========================
DDL_HOLIDAY = """
CREATE TABLE IF NOT EXISTS dim_holiday (
  id SERIAL PRIMARY KEY,
  holi_name VARCHAR(80) NOT NULL,
  holi_date DATE NOT NULL,
  is_joint_leave BOOLEAN DEFAULT FALSE
);
"""

DEFAULT_HOLIDAYS_2025 = [
    ("Tahun Baru 2025",               "2025-01-01", False),
    ("Isra Mi'raj",                   "2025-01-27", False),
    ("Tahun Baru Imlek 2576",         "2025-01-29", False),
    ("Nyepi (Saka 1947)",             "2025-03-29", False),
    ("Idulfitri 1446 H (1)",          "2025-03-31", False),
    ("Idulfitri 1446 H (2)",          "2025-04-01", False),
    ("Wafat Isa Almasih",             "2025-04-18", False),
    ("Paskah",                        "2025-04-20", False),
    ("Hari Buruh",                    "2025-05-01", False),
    ("Waisak 2569 BE",                "2025-05-12", False),
    ("Kenaikan Isa Almasih",          "2025-05-29", False),
    ("Hari Lahir Pancasila",          "2025-06-01", False),
    ("Iduladha 1446 H",               "2025-06-09", False),
    ("1 Muharram 1447 H",             "2025-06-27", False),
    ("HUT RI",                        "2025-08-17", False),
    ("Maulid Nabi",                   "2025-09-05", False),
    ("Natal",                         "2025-12-25", False),
    # Cuti Bersama (ringkas)
    ("Cuti Bersama Imlek",            "2025-01-28", True),
    ("Cuti Bersama Nyepi",            "2025-03-28", True),
    ("Cuti Bersama Idulfitri (1)",    "2025-04-02", True),
    ("Cuti Bersama Idulfitri (2)",    "2025-04-03", True),
    ("Cuti Bersama Idulfitri (3)",    "2025-04-04", True),
    ("Cuti Bersama Waisak",           "2025-05-13", True),
    ("Cuti Bersama Kenaikan",         "2025-05-30", True),
    ("Cuti Bersama Iduladha",         "2025-06-09", True),
    ("Cuti Bersama Natal",            "2025-12-26", True),
]

def ensure_holiday_table_and_seed():
    con = db(); cur = con.cursor()
    cur.execute(DDL_HOLIDAY); con.commit()
    cur.execute("SELECT COUNT(*) FROM dim_holiday;")
    cnt = cur.fetchone()[0]
    if cnt == 0:
        cur.executemany(
            "INSERT INTO dim_holiday (holi_name, holi_date, is_joint_leave) VALUES (%s,%s,%s)",
            DEFAULT_HOLIDAYS_2025
        )
        con.commit()
        logger.info("dim_holiday seeded for 2025")
    cur.close(); con.close()

def next_holiday(from_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    ensure_holiday_table_and_seed()
    from_date = from_date or date.today()
    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT holi_name, holi_date, is_joint_leave
        FROM dim_holiday
        WHERE holi_date >= %s
        ORDER BY holi_date ASC
        LIMIT 1
    """, (from_date,))
    row = cur.fetchone()
    cur.close(); con.close()
    if not row: return None
    days_left = (row["holi_date"] - from_date).days
    row["days_left"] = days_left
    return dict(row)

@app.get("/holidays")
def get_holidays():
    ensure_holiday_table_and_seed()
    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT holi_name, holi_date, is_joint_leave FROM dim_holiday ORDER BY holi_date;")
    rows = cur.fetchall()
    cur.close(); con.close()
    return rows

@app.get("/next_holiday")
def get_next_holiday():
    return next_holiday()

# =========================
# Common endpoints
# =========================
@app.get("/")
def root():
    return {"ok": True, "msg": "CMCI Seasonal API is running"}

@app.get("/debug/health")
def health():
    return {"ok": True}

@app.get("/debug/env")
def debug_env():
    return {
        "LOG_LEVEL": LOG_LEVEL,
        "LOG_DIR": LOG_DIR,
        "OPENAI_KEY_SET": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "DB_HOST": DB_HOST,
        "DB_NAME": DB_NAME,
        "DB_USER": DB_USER,
    }

# =========================
# Refresh scores (baseline 30d -> pair_metrics)
# =========================
SQL_REFRESH = """
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_tx_30d_hourly AS
SELECT merchant_id,
       date_trunc('hour', paid_at) AS hour_bucket,
       COUNT(*) AS tx_count,
       SUM(amount) AS total_amount,
       COUNT(DISTINCT user_hash) AS unique_users
FROM fact_tx
WHERE paid_at >= now() - interval '30 days'
GROUP BY merchant_id, date_trunc('hour', paid_at);

CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_tx_30d_hourly
  ON mv_tx_30d_hourly (merchant_id, hour_bucket);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_tx_30d_channel AS
SELECT merchant_id, channel,
       COUNT(*) AS tx_count,
       SUM(amount) AS total_amount
FROM fact_tx
WHERE paid_at >= now() - interval '30 days'
GROUP BY merchant_id, channel;

CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_tx_30d_channel
  ON mv_tx_30d_channel (merchant_id, channel);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_tx_30d_users AS
SELECT merchant_id, COUNT(DISTINCT user_hash) AS users_30d
FROM fact_tx
WHERE paid_at >= now() - interval '30 days'
GROUP BY merchant_id;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_tx_30d_users
  ON mv_tx_30d_users (merchant_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_tx_30d_user_overlap AS
SELECT a.merchant_id AS m1, b.merchant_id AS m2,
       COUNT(DISTINCT a.user_hash) AS users_both
FROM fact_tx a
JOIN fact_tx b
  ON a.user_hash = b.user_hash
 AND a.merchant_id < b.merchant_id
WHERE a.paid_at >= now() - interval '30 days'
  AND b.paid_at >= now() - interval '30 days'
GROUP BY a.merchant_id, b.merchant_id;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_tx_30d_user_overlap
  ON mv_tx_30d_user_overlap (m1, m2);

REFRESH MATERIALIZED VIEW mv_tx_30d_hourly;
REFRESH MATERIALIZED VIEW mv_tx_30d_channel;
REFRESH MATERIALIZED VIEW mv_tx_30d_users;
REFRESH MATERIALIZED VIEW mv_tx_30d_user_overlap;

CREATE TABLE IF NOT EXISTS pair_metrics (
  m1            INT NOT NULL,
  m2            INT NOT NULL,
  users_both    BIGINT,
  users_m1      BIGINT,
  users_m2      BIGINT,
  jaccard_30d   DOUBLE PRECISION,
  score         DOUBLE PRECISION,
  updated_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (m1, m2)
);

WITH base AS (
  SELECT o.m1,
         o.m2,
         o.users_both,
         u1.users_30d AS users_m1,
         u2.users_30d AS users_m2,
         (o.users_both::double precision) /
         NULLIF(u1.users_30d + u2.users_30d - o.users_both, 0) AS jaccard_30d
  FROM mv_tx_30d_user_overlap o
  JOIN mv_tx_30d_users u1 ON u1.merchant_id = o.m1
  JOIN mv_tx_30d_users u2 ON u2.merchant_id = o.m2
)
INSERT INTO pair_metrics (m1, m2, users_both, users_m1, users_m2, jaccard_30d, score, updated_at)
SELECT
  b.m1, b.m2, b.users_both, b.users_m1, b.users_m2,
  COALESCE(b.jaccard_30d, 0.0) AS jaccard_30d,
  COALESCE(b.jaccard_30d, 0.0) AS score,
  now()
FROM base b
ON CONFLICT (m1, m2) DO UPDATE
SET users_both  = EXCLUDED.users_both,
    users_m1    = EXCLUDED.users_m1,
    users_m2    = EXCLUDED.users_m2,
    jaccard_30d = EXCLUDED.jaccard_30d,
    score       = EXCLUDED.score,
    updated_at  = now();
"""

@app.post("/refresh_scores")
def refresh_scores():
    logger.info("refresh_scores: start")
    t0 = datetime.datetime.utcnow()
    try:
        with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
            logger.info("refresh_scores: executing SQL_REFRESH...")
            cur.execute(SQL_REFRESH)
        dt = (datetime.datetime.utcnow() - t0).total_seconds()
        logger.info(f"refresh_scores: done in {dt:.2f}s")
        return {"ok": True, "elapsed_sec": dt}
    except Exception as e:
        logger.error("refresh_scores: FAILED %s", e, exc_info=True)
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, f"refresh-{date.today()}.log"), "a") as f:
                f.write(f"[{datetime.datetime.utcnow().isoformat()}] {traceback.format_exc()}\n")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="refresh_scores failed; check logs")

# =========================
# Recommendations (auto season / all / custom)
# =========================
@app.get("/recommendations")
def recommendations(
    top: int = Query(20, ge=1, le=200),
    mode: str = Query("auto", regex="^(auto|all|custom)$"),
    season: Optional[str] = Query(None)
):
    if mode == "all":
        sql = """
        SELECT
          pm.m1, dm1.name AS m1_name,
          pm.m2, dm2.name AS m2_name,
          pm.score, pm.jaccard_30d,
          NULL::text as season_name
        FROM pair_metrics pm
        JOIN dim_merchant dm1 ON dm1.merchant_id = pm.m1
        JOIN dim_merchant dm2 ON dm2.merchant_id = pm.m2
        ORDER BY pm.score DESC
        LIMIT %s;
        """
        con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, (top,))
        rows = cur.fetchall(); cur.close(); con.close()
        return {"mode": "all", "season": None, "rows": rows}

    season_row = None
    if mode == "custom" and season:
        ensure_season_table_and_seed()
        con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT season_name, start_date, end_date FROM dim_season WHERE season_name=%s LIMIT 1", (season,))
        season_row = cur.fetchone(); cur.close(); con.close()
    else:
        season_row = current_season_row()

    if not season_row:
        logger.info("No current season; fallback to baseline (mode=all)")
        return recommendations(top=top, mode="all")

    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql_season = """
    WITH u AS (
      SELECT DISTINCT user_hash, merchant_id
      FROM fact_tx
      WHERE paid_at::date BETWEEN %s AND %s
    ),
    pairs AS (
      SELECT a.merchant_id AS m1, b.merchant_id AS m2, COUNT(*) AS users_both
      FROM u a
      JOIN u b ON a.user_hash = b.user_hash AND a.merchant_id < b.merchant_id
      GROUP BY 1,2
    ),
    mstats AS (
      SELECT merchant_id, COUNT(DISTINCT user_hash) AS users_uniq
      FROM u GROUP BY 1
    )
    SELECT
      p.m1, dm1.name AS m1_name,
      p.m2, dm2.name AS m2_name,
      (p.users_both::double precision) /
        NULLIF(m1.users_uniq + m2.users_uniq - p.users_both,0) AS jaccard_30d,
      (p.users_both::double precision) /
        NULLIF(m1.users_uniq + m2.users_uniq - p.users_both,0) AS score
    FROM pairs p
    JOIN mstats m1 ON m1.merchant_id = p.m1
    JOIN mstats m2 ON m2.merchant_id = p.m2
    JOIN dim_merchant dm1 ON dm1.merchant_id = p.m1
    JOIN dim_merchant dm2 ON dm2.merchant_id = p.m2
    ORDER BY score DESC
    LIMIT %s;
    """
    cur.execute(sql_season, (season_row["start_date"], season_row["end_date"], top))
    rows = cur.fetchall(); cur.close(); con.close()
    return {"mode": "season", "season": season_row, "rows": rows}

# =========================
# Pair Detail & Audience
# =========================
def _merchant_names(m1:int, m2:int) -> Tuple[str,str]:
    with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT merchant_id, name FROM dim_merchant WHERE merchant_id IN (%s,%s)", (m1, m2))
        found = {r["merchant_id"]: (r["name"] or str(r["merchant_id"])) for r in cur.fetchall()}
    return (found.get(m1, str(m1)), found.get(m2, str(m2)))

@app.get("/pair/{m1}/{m2}/detail")
def pair_detail(m1: int, m2: int):
    con = db(); cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # meta dari pair_metrics (optional)
    cur.execute("""
      SELECT dm1.name AS m1_name, dm2.name AS m2_name,
             pm.score, pm.jaccard_30d
      FROM pair_metrics pm
      JOIN dim_merchant dm1 ON dm1.merchant_id = pm.m1
      JOIN dim_merchant dm2 ON dm2.merchant_id = pm.m2
      WHERE pm.m1=%s AND pm.m2=%s
    """, (m1, m2))
    meta = cur.fetchone()

    # hour histogram 30d
    cur.execute("""
      WITH hh AS (
        SELECT merchant_id, date_part('hour', paid_at)::int AS hr, COUNT(*)::double precision AS cnt
        FROM fact_tx
        WHERE paid_at >= now() - interval '30 days'
          AND merchant_id IN (%s,%s)
        GROUP BY 1,2
      ),
      base AS (
        SELECT merchant_id, hr, cnt,
               cnt / NULLIF(SUM(cnt) OVER (PARTITION BY merchant_id),0) AS share
        FROM hh
      ),
      hrs AS (SELECT generate_series(0,23) AS hr)
      SELECT b.merchant_id, h.hr, COALESCE(b.share,0.0) AS share
      FROM hrs h
      LEFT JOIN base b ON b.hr = h.hr AND b.merchant_id IN (%s,%s)
      ORDER BY b.merchant_id, h.hr;
    """, (m1, m2, m1, m2))
    hh_rows = cur.fetchall()

    # channel share 30d
    cur.execute("""
      WITH ch AS (
        SELECT merchant_id, channel, COUNT(*)::double precision AS cnt
        FROM fact_tx
        WHERE paid_at >= now() - interval '30 days'
          AND merchant_id IN (%s,%s)
        GROUP BY 1,2
      )
      SELECT merchant_id, channel,
             cnt / NULLIF(SUM(cnt) OVER (PARTITION BY merchant_id),0) AS share
      FROM ch
      ORDER BY merchant_id, channel;
    """, (m1, m2))
    ch_rows = cur.fetchall()

    cur.close(); con.close()

    # nama fallback kalau meta kosong
    if meta:
        m1_name, m2_name = meta["m1_name"], meta["m2_name"]
    else:
        m1_name, m2_name = _merchant_names(m1, m2)

    def to_hist(rows, mid): return [r["share"] for r in rows if r["merchant_id"] == mid]
    channels: Dict[int, Dict[str, float]] = {}
    for r in ch_rows:
        channels.setdefault(r["merchant_id"], {})[r["channel"]] = float(r["share"])

    return {
        "m1": {"id": m1, "name": m1_name,
               "hour_hist": to_hist(hh_rows, m1), "channel_share": channels.get(m1, {})},
        "m2": {"id": m2, "name": m2_name,
               "hour_hist": to_hist(hh_rows, m2), "channel_share": channels.get(m2, {})},
        "metrics": {"score": float((meta or {}).get("score") or 0),
                    "jaccard_30d": float((meta or {}).get("jaccard_30d") or 0)}
    }

def _interval(text: str) -> str:
    if text.endswith("d"): return f"{int(text[:-1])} days"
    if text.endswith("h"): return f"{int(text[:-1])} hours"
    return "30 days"

@app.get("/audience")
def audience(m1: int, m2: int, winA: str = "30d", winB: str = "90d",
             limit: Optional[int] = Query(None, ge=1, le=200000)):
    interA = _interval(winA); interB = _interval(winB)
    sql = f"""
    WITH a AS (
      SELECT DISTINCT user_hash
      FROM fact_tx
      WHERE merchant_id=%s AND paid_at >= now() - interval '{interA}'
    ),
    b AS (
      SELECT DISTINCT user_hash
      FROM fact_tx
      WHERE merchant_id=%s AND paid_at >= now() - interval '{interB}'
    )
    SELECT a.user_hash
    FROM a LEFT JOIN b USING (user_hash)
    WHERE b.user_hash IS NULL
    {"LIMIT %s" if limit else ""}
    ;
    """
    con = db(); cur = con.cursor()
    params = (m1, m2) if not limit else (m1, m2, limit)
    cur.execute(sql, params)
    users = [u[0] for u in cur.fetchall()]
    cur.close(); con.close()
    return {"count": len(users), "users": users}

# =========================
# Helpers for friendly Rule Insight
# =========================
def _dominant_channels(ch_share: dict, top=2):
    if not ch_share: return []
    return sorted(ch_share.items(), key=lambda x: x[1], reverse=True)[:top]

def _top_hours(hist: List[float], k=3):
    if not hist: return []
    pairs = sorted([(i, (hist[i] or 0.0)) for i in range(len(hist))], key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:k]]

def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "-"

def _join_channels(ch_keys: List[str]) -> str:
    if not ch_keys: return "-"
    if len(ch_keys) == 1: return ch_keys[0]
    return ", ".join(ch_keys[:-1]) + " & " + ch_keys[-1]

def _plain_overlap_label(jac: float) -> str:
    if jac >= 0.20: return "tinggi"
    if jac >= 0.10: return "menengah"
    if jac >= 0.05: return "cukup"
    if jac >  0.00: return "rendah"
    return "sangat rendah"

# =========================
# Rule-based Insight (non-teknis)
# =========================
@app.get("/insight/{m1}/{m2}")
def insight(m1:int, m2:int):
    d = pair_detail(m1, m2)
    season = current_season_row()
    near   = next_holiday()

    m1n, m2n = d["m1"]["name"], d["m2"]["name"]
    h1, h2   = d["m1"]["hour_hist"] or [], d["m2"]["hour_hist"] or []
    ch1, ch2 = d["m1"]["channel_share"] or {}, d["m2"]["channel_share"] or {}
    jac      = float(d["metrics"]["jaccard_30d"] or 0.0)
    score    = float(d["metrics"]["score"] or 0.0)

    top_h1 = _top_hours(h1); top_h2 = _top_hours(h2)
    if top_h1 and top_h2:
        offs = sum([min([abs(a-b), 24-abs(a-b)]) for a in top_h1 for b in top_h2]) / (len(top_h1)*len(top_h2))
    else:
        offs = 0.0

    dom1 = [x[0] for x in _dominant_channels(ch1)]
    dom2 = [y[0] for y in _dominant_channels(ch2)]
    synergy = len(set(dom1).intersection(set(dom2))) == 0

    season_name = season["season_name"] if season else None
    near_line   = f"{near['holi_name']} ({near['holi_date']}) – {near['days_left']} hari lagi" if near else None

    parts = []
    header = f"Peluang kolaborasi: **{m1n} × {m2n}**"
    context = []
    context.append(f"Tingkat kesamaan pelanggan: **{_plain_overlap_label(jac)}**")
    if season_name: context.append(f"Musim saat ini: **{season_name}**")
    if near_line:   context.append(f"Libur terdekat: **{near_line}**")
    parts.append(header + "\n" + " · ".join(context))

    ideas = []

    if season_name in ["Lebaran", "Natal & Tahun Baru", "Libur Sekolah"]:
        ideas.append({
            "judul": "Paket musiman lintas merchant",
            "inti":  f"Gabungkan penawaran {m1n} dan {m2n} dalam 1 paket hemat.",
            "kenapa": "Di musim ini, pelanggan cenderung belanja lebih banyak dan tertarik bundling.",
            "aksi":  "Buat voucher paket (tersedia via QRIS) + stok kreatif banner di jam ramai.",
            "target": f"Konversi: 8–12%, Uplift GMV: {_fmt_pct(max(jac,0.02))}"
        })

    if offs >= 6:
        ideas.append({
            "judul": "Happy Hours silang",
            "inti":  f"Tawarkan promo {m2n} saat jam ramai {m1n}, dan sebaliknya.",
            "kenapa": "Jam ramai kedua merchant berbeda jauh, jadi bisa saling mengisi trafik.",
            "aksi":  "Jadwalkan push/in-app banner 2–3 jam sebelum puncak masing-masing.",
            "target": f"Klik: 10–15%, Konversi: 3–7%"
        })

    if synergy:
        ideas.append({
            "judul": "Sinergi kanal pembayaran",
            "inti":  f"{m1n} kuat di {_join_channels(dom1)} sedangkan {m2n} kuat di {_join_channels(dom2)}.",
            "kenapa": "Perbedaan kanal ini bisa saling melengkapi audiens.",
            "aksi":  f"Buat promo silang: QRIS-only di {m1n} memberi voucher {m2n} (VA/CC) & sebaliknya.",
            "target": "Repeat rate: 10–15% dari pembeli promo"
        })

    if not ideas:
        ideas.append({
            "judul": "Tes cepat audience builder",
            "inti":  f"Ajak pelanggan {m1n} yang belum pernah belanja di {m2n}.",
            "kenapa": "Cara cepat validasi minat dengan biaya kecil.",
            "aksi":  "Kirim 2x kampanye (reminder 48 jam). Uji A/B nilai voucher.",
            "target": "Sampel min. 2.000 user, Konversi awal 2–4%"
        })

    for i, it in enumerate(ideas, 1):
        parts.append(
            f"{i}) **{it['judul']}**\n"
            f"   • Inti: {it['inti']}\n"
            f"   • Kenapa cocok: {it['kenapa']}\n"
            f"   • Cara eksekusi cepat: {it['aksi']}\n"
            f"   • Target hasil: {it['target']}"
        )

    note = (
        f"\nCatatan: Potensi kolaborasi **{_plain_overlap_label(jac)}** "
        f"(indikasi skor {score:.4f}). Mulai dari eksperimen ringan lalu iterasi mingguan."
    )
    parts.append(note)

    return {"pair": f"{m1n} × {m2n}", "idea": "\n\n".join(parts)}

# =========================
# AI Insight (OpenAI)
# =========================
def build_campaign_prompt(pair_detail: Dict[str, Any],
                          season: Optional[Dict[str, Any]],
                          horizon: str = "next_month") -> str:
    m1 = pair_detail["m1"]["name"]; m2 = pair_detail["m2"]["name"]
    jac = pair_detail["metrics"]["jaccard_30d"]; score = pair_detail["metrics"]["score"]
    h1  = pair_detail["m1"]["hour_hist"] or []; h2 = pair_detail["m2"]["hour_hist"] or []
    ch1 = pair_detail["m1"]["channel_share"] or {}; ch2 = pair_detail["m2"]["channel_share"] or {}
    top_h1 = _top_hours(h1); top_h2 = _top_hours(h2)
    season_name = season["season_name"] if season else "—"
    near = next_holiday()
    trend_context = None
    if near and near["days_left"] <= 30:
        near_line = f"{near['holi_name']} on {near['holi_date']} (D-{near['days_left']})"
    else:
        # fallback: cari tren viral via AI
        try:
            trend_prompt = "Apa tren viral di Indonesia bulan ini yang bisa digunakan untuk kampanye marketing digital?"
            trend_resp = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": trend_prompt}],
                max_tokens=60,
                temperature=0.7,
            )
            trend_context = trend_resp.choices[0].message.content.strip()
            near_line = f"Trending Topic: {trend_context}"
        except Exception as e:
            logger.warning(f"Trend fetch fallback: {e}")
            near_line = "No nearby holiday or trend data."


    return f"""
You are a marketing data strategist for a payments company (DOKU).
Generate 3 concise, high-impact cross-merchant campaign ideas for the {horizon.replace('_',' ')}.

PAIR: {m1} × {m2}
SEASON: {season_name}
UPCOMING_HOLIDAY: {near_line}
SCORES: jaccard_30d={jac:.4f}, score={score:.4f}
PEAK_HOURS:
- {m1}: top {top_h1}
- {m2}: top {top_h2}
CHANNEL_SHARE:
- {m1}: {ch1}
- {m2}: {ch2}

CONSTRAINTS & STYLE:
- Ideas must be grounded in the data above (overlap, peak hours, channels, season/holiday).
- Prefer QRIS/VA/CC mechanics depending on channel dominance.
- Include a short “Why now” tied to season/nearest holiday.
- Each idea must include: title, mechanic (≤2 sentences), incentive, and suggested KPIs (CTR/CVR or GMV lift).
- Output as bullet list, no JSON, no preamble.
"""

def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or OpenAI is None:
        logger.warning("OPENAI_API_KEY not set or openai lib not available")
        return ("[Local Rule-Based Fallback]\n"
                "Set OPENAI_API_KEY untuk menggunakan AI generatif.")
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a senior growth strategist for payments and ecommerce."},
                {"role":"user","content":prompt}
            ],
            temperature=0.6,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logger.exception("OpenAI API call failed (model=%s)", model)
        raise

@app.get("/ai_insight/{m1}/{m2}")
def ai_insight(m1:int, m2:int, horizon: str = "next_month"):
    try:
        pd = pair_detail(m1, m2)
        season = current_season_row()
        prompt = build_campaign_prompt(pd, season, horizon=horizon)

        logger.info("AI_INSIGHT req pair=(%s,%s) horizon=%s season=%s",
                    m1, m2, horizon, (season or {}).get("season_name"))

        text = call_llm(prompt)
        logger.info("AI_INSIGHT ok pair=(%s,%s)", m1, m2)

        return {
            "pair": f"{pd['m1']['name']} × {pd['m2']['name']}",
            "season": season,
            "horizon": horizon,
            "insight": text
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI_INSIGHT failed pair=(%s,%s)", m1, m2)
        raise HTTPException(
            status_code=500,
            detail=f"AI insight gagal: {type(e).__name__}. Cek logs untuk detail."
        )

# ====== AI Auto Insights Log ======
DDL_AI_LOG = """
CREATE TABLE IF NOT EXISTS ai_insight_log (
  id SERIAL PRIMARY KEY,
  m1 INT NOT NULL,
  m2 INT NOT NULL,
  season TEXT,
  insight TEXT,
  generated_at TIMESTAMPTZ DEFAULT now()
);
"""

def ensure_ai_log():
    with conn() as c, c.cursor() as cur:
        cur.execute(DDL_AI_LOG)

@app.post("/auto_insight/generate")
def auto_insight_generate(limit:int=5):
    ensure_ai_log()
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    if not api_key or OpenAI is None:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT m1, m2 FROM pair_metrics ORDER BY score DESC LIMIT %s;", (limit,))
        pairs = cur.fetchall()

        client = OpenAI(api_key=api_key)
        for m1, m2 in pairs:
            # pakai nama merchant biar lebih enak
            m1n, m2n = _merchant_names(m1, m2)
            season = (current_season_row() or {}).get("season_name")
            prompt = (
                f"Buat 1 insight kampanye kolaborasi ringkas untuk merchant '{m1n}' dan '{m2n}'. "
                f"Fokus pada musim: {season or '-'}, gunakan mekanik kanal pembayaran (QRIS/VA/CC) yang cocok, "
                "sertakan kenapa sekarang (holiday/season), dan KPI singkat. Format bullet, max 4 baris."
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.6,
                    max_tokens=250
                )
                idea = resp.choices[0].message.content.strip()
            except Exception:
                logger.exception("auto_insight_generate: OpenAI fail for pair=(%s,%s)", m1, m2)
                idea = "[fallback] insight tidak tersedia"
            cur.execute(
                "INSERT INTO ai_insight_log (m1,m2,season,insight) VALUES (%s,%s,%s,%s)",
                (m1, m2, season, idea)
            )
    return {"ok": True, "count": len(pairs)}

@app.get("/auto_insights")
def auto_insights():
    ensure_ai_log()
    with conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT dm1.name AS m1_name, dm2.name AS m2_name, a.season, a.insight, a.generated_at
            FROM ai_insight_log a
            JOIN dim_merchant dm1 ON dm1.merchant_id=a.m1
            JOIN dim_merchant dm2 ON dm2.merchant_id=a.m2
            ORDER BY a.generated_at DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
    return rows

# =========================
# Metrics (simple)
# =========================
@app.get("/metrics")
def metrics():
    return {
        "uptime_sec": time.time() - START_TIME,
        "env": {"db": DB_NAME, "host": DB_HOST},
    }

# =========================
# Geo helpers
# =========================
def _safe_latlon(lat, lon) -> Optional[Tuple[float, float]]:
    try:
        lat = float(lat); lon = float(lon)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        return lat, lon
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    a = _safe_latlon(lat1, lon1)
    b = _safe_latlon(lat2, lon2)
    if a is None or b is None:
        return None
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    s = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2.0)**2
    s = max(0.0, min(1.0, s))
    c = 2.0 * math.atan2(math.sqrt(s), math.sqrt(1.0 - s))
    d = R * c
    return d if math.isfinite(d) else None

def _jaccard_live(cur, m1: int, m2: int) -> float:
    cur.execute("""
      WITH
      u1 AS (
        SELECT COUNT(DISTINCT user_hash) AS n
        FROM fact_tx
        WHERE merchant_id = %s
          AND status = 'SALE'
          AND paid_at >= now() - interval '30 day'
      ),
      u2 AS (
        SELECT COUNT(DISTINCT user_hash) AS n
        FROM fact_tx
        WHERE merchant_id = %s
          AND status = 'SALE'
          AND paid_at >= now() - interval '30 day'
      ),
      ub AS (
        SELECT COUNT(DISTINCT a.user_hash) AS n
        FROM fact_tx a
        JOIN fact_tx b ON a.user_hash = b.user_hash
        WHERE a.merchant_id = %s
          AND b.merchant_id = %s
          AND a.status = 'SALE'
          AND b.status = 'SALE'
          AND a.paid_at >= now() - interval '30 day'
          AND b.paid_at >= now() - interval '30 day'
      )
      SELECT (SELECT n FROM u1) AS u1,
             (SELECT n FROM u2) AS u2,
             (SELECT n FROM ub) AS u_both;
    """, (m1, m2, m1, m2))
    row = cur.fetchone() or {"u1": 0, "u2": 0, "u_both": 0}
    u1 = int(row["u1"] or 0); u2 = int(row["u2"] or 0); u_both = int(row["u_both"] or 0)
    denom = (u1 + u2 - u_both)
    return float(u_both) / denom if denom > 0 else 0.0

# =========================
# GEO Recommendations (live)
# =========================
@app.get("/geo/recommendations")
def geo_recommendations(
    city: str = Query(..., description="m1 dari kota ini"),
    top: int = Query(5, ge=1, le=50),
    same_city: bool = Query(False, description="true=pasangan sekota"),
    radius_km: float | None = Query(None, description="maks jarak km (opsional)"),
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="bobot geo_score")
):
    log = logging.getLogger("cmci")
    try:
        # m1 candidates
        with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
              SELECT merchant_id, COALESCE(NULLIF(name,''), merchant_id::text) AS name,
                     city, latitude::float AS lat, longitude::float AS lon
              FROM dim_merchant
              WHERE city = %s AND latitude IS NOT NULL AND longitude IS NOT NULL
            """, (city,))
            m1s = [dict(r) for r in cur.fetchall()]

            # m2 candidates
            if same_city:
                cur.execute("""
                  SELECT merchant_id, COALESCE(NULLIF(name,''), merchant_id::text) AS name,
                         city, latitude::float AS lat, longitude::float AS lon
                  FROM dim_merchant
                  WHERE city = %s AND latitude IS NOT NULL AND longitude IS NOT NULL
                """, (city,))
            else:
                cur.execute("""
                  SELECT merchant_id, COALESCE(NULLIF(name,''), merchant_id::text) AS name,
                         city, latitude::float AS lat, longitude::float AS lon
                  FROM dim_merchant
                  WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                """)
            m2s = [dict(r) for r in cur.fetchall()]

        if not m1s or not m2s:
            return []

        rows = []
        with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
            for a in m1s:
                for b in m2s:
                    if a["merchant_id"] == b["merchant_id"]:
                        continue
                    d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                    if d is None:
                        continue
                    if radius_km is not None and d > radius_km:
                        continue
                    # geo score: 1 at 0 km; → linearly decay to 0 at 1000 km
                    geo_score = max(0.0, 1.0 - min(d/1000.0, 1.0))
                    jac = _jaccard_live(cur, a["merchant_id"], b["merchant_id"])
                    final_score = alpha*geo_score + (1.0-alpha)*jac
                    rows.append({
                        "m1": a["merchant_id"], "m2": b["merchant_id"],
                        "m1_name": a["name"], "m2_name": b["name"],
                        "m1_city": a["city"], "m2_city": b["city"],
                        "distance_km": round(d, 2),
                        "geo_score": round(geo_score, 4),
                        "jaccard_30d": round(jac, 6),
                        "final_score": round(final_score, 6),
                    })

        rows.sort(key=lambda r: r["final_score"], reverse=True)
        return rows[:top]
    except Exception:
        log.exception("geo live failed city=%s top=%s same_city=%s radius=%s alpha=%s",
                      city, top, same_city, radius_km, alpha)
        raise HTTPException(status_code=500, detail="Geo live failed")

@app.get("/merchant/{mid}/geo_recommendations")
def merchant_geo_recs(mid: int, radius_km: float = 25.0, top: int = 5, alpha: float = 0.6):
    """
    Versi live: ambil koordinat merchant {mid}, hitung jarak & jaccard ke merchant lain dalam radius.
    """
    top = max(1, min(top, 50))
    with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT name, city, latitude::float AS lat, longitude::float AS lon FROM dim_merchant WHERE merchant_id=%s", (mid,))
        me = cur.fetchone()
        if not me or me["lat"] is None or me["lon"] is None:
            return []

        cur.execute("""
          SELECT merchant_id, COALESCE(NULLIF(name,''), merchant_id::text) AS name,
                 city, latitude::float AS lat, longitude::float AS lon
          FROM dim_merchant
          WHERE merchant_id <> %s AND latitude IS NOT NULL AND longitude IS NOT NULL
        """, (mid,))
        candidates = [dict(r) for r in cur.fetchall()]

        rows = []
        for b in candidates:
            d = haversine_km(me["lat"], me["lon"], b["lat"], b["lon"])
            if d is None or d > radius_km:
                continue
            geo_score = max(0.0, 1.0 - min(d/1000.0, 1.0))
            jac = _jaccard_live(cur, mid, b["merchant_id"])
            final_score = alpha*geo_score + (1.0-alpha)*jac
            rows.append({
                "m1": mid, "m2": b["merchant_id"],
                "m1_name": me["name"] or str(mid), "m2_name": b["name"],
                "m1_city": me["city"], "m2_city": b["city"],
                "distance_km": round(d, 2),
                "geo_score": round(geo_score, 4),
                "jaccard_30d": round(jac, 6),
                "final_score": round(final_score, 6),
            })

    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows[:top]

# =========================
# Static files (UI)
# =========================
app.mount("/app", StaticFiles(directory="static", html=True), name="app")

@app.post("/scheduler/seed")
def scheduler_seed():
    with conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO dim_merchant (merchant_id, name, city, latitude, longitude)
            SELECT generate_series(2000,2005), 'Demo-' || generate_series(2000,2005), 'Jakarta', 
                   6.20 + random()/10, 106.80 + random()/10
            ON CONFLICT DO NOTHING;
        """)
        c.commit()
    return {"ok": True, "msg": "Dummy merchant inserted."}

# ====== Merge AI: /chat intent-aware monthly SR ======
from fastapi import Body
from datetime import datetime, timedelta
import calendar, json, re, math
from typing import Optional, Tuple, Dict, Any, List

MONTH_ID = {
    # id
    "january":1,"jan":1,"januari":1,
    "february":2,"feb":2,"februari":2,
    "march":3,"mar":3,"maret":3,
    "april":4,"apr":4,
    "may":5,"mei":5,
    "june":6,"jun":6,"juni":6,
    "july":7,"jul":7,"juli":7,
    "august":8,"aug":8,"agustus":8,
    "september":9,"sep":9,
    "october":10,"oct":10,"oktober":10,"okt":10,
    "november":11,"nov":11,
    "december":12,"dec":12,"desember":12,"des":12,
}

CHANNEL_ALIASES = {
    "QRIS":["qris"],
    "VA":["va","virtual account","virtualaccount"],
    "CC":["cc","credit card","kartu kredit","creditcard"],
}

def _prev_month(year:int, month:int)->Tuple[int,int]:
    if month==1: return (year-1,12)
    return (year, month-1)

def _parse_intent(txt:str) -> Dict[str,Any]:
    t = txt.lower().strip()

    # clientid / merchant / client
    cid = None
    m = re.search(r"(?:clientid|client|merchant)\s*[:=]?\s*(\d+)", t)
    if m: cid = int(m.group(1))
    else:
        m = re.search(r"\b(\d{4,})\b", t)  # fallback angka 4+ digit
        if m: cid = int(m.group(1))

    # month / year
    now = datetime.now()
    year = now.year; month = now.month
    # nama bulan
    for k,v in MONTH_ID.items():
        if re.search(rf"\b{k}\b", t):
            month = v
            break
    # angka MM-YYYY
    m = re.search(r"(0?[1-9]|1[0-2])[-/ ](20\d{2})", t)
    if m:
        month = int(m.group(1)); year = int(m.group(2))
    # kata “bulan ini/kemarin”
    if re.search(r"bulan\s*(ini|current)", t):
        year, month = now.year, now.month
    if re.search(r"(bulan\s*kemarin|last month)", t):
        year, month = _prev_month(now.year, now.month)

    # channel (normalize)
    channel = None
    for can, aliases in CHANNEL_ALIASES.items():
        for a in [can] + aliases:
            if re.search(rf"\b{re.escape(a.lower())}\b", t):
                channel = can
                break
        if channel: break

    # compare?
    compare = bool(re.search(r"(bandingkan|compare|bulan sebelumnya|prev month|previous)", t))

    return {"clientid": cid, "year": year, "month": month, "channel": channel, "compare": compare}

def _month_window(y:int,m:int)->Tuple[datetime,datetime]:
    start = datetime(y,m,1)
    last_day = calendar.monthrange(y,m)[1]
    end = datetime(y,m,last_day) + timedelta(days=1)  # exclusive
    return start, end

def _fetch_month_summary(cur, clientid:int, year:int, month:int, channel:Optional[str]=None) -> Dict[str,Any]:
    start, end = _month_window(year, month)

    # base filter
    ch_where = " AND channel=%s " if channel else ""
    ch_param = [channel] if channel else []

    # totals
    cur.execute(f"""
      WITH base AS (
        SELECT user_hash, amount, status, channel, date_trunc('day', paid_at) AS d
        FROM fact_tx
        WHERE merchant_id=%s AND paid_at >= %s AND paid_at < %s {ch_where}
      )
      SELECT
        COUNT(*)                           AS total_tx,
        COUNT(*) FILTER (WHERE status='SALE') AS success_tx,
        COALESCE(SUM(amount),0)            AS gmv,
        COUNT(DISTINCT user_hash)          AS unique_users
      FROM base;
    """, [clientid, start, end] + ch_param)
    tot = dict(cur.fetchone())

    # by channel (always, untuk saran)
    cur.execute("""
      WITH base AS (
        SELECT channel, status, amount
        FROM fact_tx
        WHERE merchant_id=%s AND paid_at >= %s AND paid_at < %s
      )
      SELECT channel,
             COUNT(*) AS total_tx,
             COUNT(*) FILTER (WHERE status='SALE') AS success_tx,
             COALESCE(SUM(amount),0) AS gmv
      FROM base
      GROUP BY channel
      ORDER BY channel;
    """, [clientid, start, end])
    by_channel = [dict(r) for r in cur.fetchall()]

    # by day (untuk UI detail/AI konteks)
    cur.execute(f"""
      WITH days AS (
        SELECT generate_series(%s::date, (%s::date - interval '1 day'), interval '1 day') AS d
      ),
      base AS (
        SELECT date_trunc('day', paid_at)::date AS d, status
        FROM fact_tx
        WHERE merchant_id=%s AND paid_at >= %s AND paid_at < %s {ch_where}
      )
      SELECT d::date AS day,
             COUNT(b.*) AS total_tx,
             COUNT(*) FILTER (WHERE b.status='SALE') AS success_tx
      FROM days
      LEFT JOIN base b USING (d)
      GROUP BY 1 ORDER BY 1;
    """, [start, end, clientid, start, end] + ch_param)
    by_day = [dict(r) for r in cur.fetchall()]

    # merchant name
    cur.execute("SELECT COALESCE(NULLIF(name,''), merchant_id::text) FROM dim_merchant WHERE merchant_id=%s LIMIT 1", (clientid,))
    merchant = (cur.fetchone() or [str(clientid)])[0]

    sr = 0.0
    if tot["total_tx"]:
        sr = tot["success_tx"] / tot["total_tx"] * 100.0

    return {
        "merchant": merchant,
        "clientid": clientid,
        "year": year,
        "month": month,
        "channel": channel,          # bisa None
        "totals": {
            "total_tx": int(tot["total_tx"] or 0),
            "success_tx": int(tot["success_tx"] or 0),
            "gmv": int(tot["gmv"] or 0),
            "unique_users": int(tot["unique_users"] or 0),
            "success_rate_pct": round(sr, 2)
        },
        "by_channel": [
            {
              "channel": r["channel"],
              "total_tx": int(r["total_tx"] or 0),
              "success_tx": int(r["success_tx"] or 0),
              "gmv": int(r["gmv"] or 0),
              "success_rate_pct": round((r["success_tx"]/(r["total_tx"] or 1))*100.0, 2) if r["total_tx"] else 0.0
            } for r in by_channel
        ],
        "by_day": [
            {
              "day": str(r["day"]),
              "total_tx": int(r["total_tx"] or 0),
              "success_tx": int(r["success_tx"] or 0),
              "success_rate_pct": round(( (r["success_tx"] or 0) / (r["total_tx"] or 1) )*100.0, 2) if r["total_tx"] else 0.0
            } for r in by_day
        ]
    }

def _md_stat_block(s:Dict[str,Any])->str:
    t = s["totals"]
    head = f"**{s['merchant']}** • Periode **{calendar.month_name[s['month']]} {s['year']}**"
    ch = f"\nChannel: **{s['channel']}**" if s.get("channel") else ""
    body = (
        f"{head}{ch}\n"
        f"Success Rate: **{t['success_rate_pct']:.2f}%** "
        f"({t['success_tx']}/{t['total_tx']})\n"
        f"GMV: **Rp {t['gmv']:,}** • Unique Users: **{t['unique_users']:,}**"
    )
    return body

def _ai_summary(pair_payload:Dict[str,Any]) -> str:
    """
    pair_payload = {
      "current": <summary dict>,           # wajib
      "previous": <summary dict or None>,  # opsional
    }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # fallback rule-based singkat
        cur = pair_payload["current"]["totals"]
        note = []
        if cur["success_rate_pct"] < 75: note.append("SR rendah — cek error rate dan jam spike.")
        if cur["total_tx"] < 50: note.append("Volume kecil; lakukan kampanye akuisisi.")
        base = " • ".join(note) if note else "Performa stabil. Lanjutkan optimasi kanal dominan."
        return f"[Local] {_md_stat_block(pair_payload['current'])}\n\nRingkasan: {base}"

    # prompt ringkas + data JSON
    data_json = json.dumps(pair_payload, ensure_ascii=False)
    sys = "You are a concise analytics assistant for payments. Write in Indonesian. Be crisp and actionable."
    usr = f"""
Berikut data ringkas transaksi (current & optional previous). Tugasmu:
1) Berikan 2–3 poin insight utama (<= 50 kata total).
2) Jika ada previous, sebutkan perubahan SR/volume secara singkat.
3) Tutup dengan 1 rekomendasi praktis.

DATA:
{data_json}
"""

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.4, max_tokens=220,
    )
    return resp.choices[0].message.content.strip()

def _channel_suggestions(all_channels:List[Dict[str,Any]])->List[str]:
    labels = []
    for r in all_channels:
        c = r["channel"]
        sr = r["success_rate_pct"]
        labels.append(f"channel {c} (SR {sr:.1f}%)")
    if not labels: labels = ["channel QRIS","channel VA","channel CC"]
    return labels

# ==== simple in-memory session context (TTL optional) ====
SESSION: dict[str, dict] = {}

def _get_sid(payload: dict) -> str:
    # frontend kirim "sid" (random uuid) per user/browser
    return (payload.get("sid") or "default").strip()

def _norm_channel_word(s: str | None) -> str | None:
    if not s: return None
    s = s.strip().lower()
    if s in ("qris","qr","qr code"): return "QRIS"
    if s in ("va","virtual account"): return "VA"
    if s in ("cc","card","credit card","kartu kredit"): return "CC"
    return s.upper()

def _mk_choices(labels: list[str]) -> list[dict]:
    # tombol yang menempelkan teks ke input saat diklik
    return [{"label": lab, "text": lab} for lab in labels]

# ==== simple in-memory session context (TTL optional) ====
SESSION: dict[str, dict] = {}

def _get_sid(payload: dict) -> str:
    # frontend kirim "sid" (random uuid) per user/browser
    return (payload.get("sid") or "default").strip()

def _norm_channel_word(s: str | None) -> str | None:
    if not s: return None
    s = s.strip().lower()
    if s in ("qris","qr","qr code"): return "QRIS"
    if s in ("va","virtual account"): return "VA"
    if s in ("cc","card","credit card","kartu kredit"): return "CC"
    return s.upper()

def _mk_choices(labels: list[str]) -> list[dict]:
    # tombol yang menempelkan teks ke input saat diklik
    return [{"label": lab, "text": lab} for lab in labels]


def _extract_channel_loose(text: str | None) -> str | None:
    """
    Tangkap channel dari frasa pendek:
      - "channel qris", "qris", "va", "cc", "kartu kredit", dst.
    Return: "QRIS" | "VA" | "CC" | None
    """
    if not text:
        return None
    t = text.strip().lower()

    # pola "channel <xxx>"
    m = re.search(r"\bchannel\s+([a-zA-Z ]+)\b", t)
    cand = (m.group(1) if m else t).strip()

    for canon, aliases in CHANNEL_ALIASES.items():
        if cand == canon.lower() or cand in aliases:
            return

@app.post("/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    """
    Mode 2-step interaktif + session:
      - Step 1 (tanpa channel): ringkasan + tombol channel.
      - Step 2 (pilih channel): detail + insight.
    """
    try:
        sid = _get_sid(payload)
        q = (payload.get("text") or "").strip()
        if not q:
            return {"reply":"(pesan kosong)", "suggestions":[], "actions":[], "choices":[]}

        # ---- parse intent utama
        intent = _parse_intent(q)
        cid    = intent["clientid"]
        year, month = intent["year"], intent["month"]
        channel = intent.get("channel")
        channel = (_norm_channel_word(channel)
           if channel else _extract_channel_loose(q))
        compare = intent["compare"]

        # ---- jika user hanya ketik "qris"/"va"/"cc", ambil dari session
        if not cid and not year and not month and channel:
            ctx = SESSION.get(sid) or {}
            if ctx.get("clientid"):
                cid   = ctx["clientid"]
                year  = ctx["year"]
                month = ctx["month"]
            else:
                return {
                    "reply": "Aku butuh *clientid* dulu sebelum memilih channel. Contoh: `clientid 1001`.",
                    "suggestions": ["clientid 1001", "clientid 1002 Oktober 2025"],
                    "actions": ["clientid 1001", "clientid 1002 Oktober 2025"],
                    "choices": _mk_choices(["clientid 1001","clientid 1002 Oktober 2025"])
                }

        # ---- wajib punya clientid
        if not cid:
            return {
                "reply":"Aku tidak menemukan *clientid*. Contoh: `clientid 1001`.",
                "suggestions":["clientid 1001 bulan ini","clientid 1002 Oktober 2025"],
                "actions":["clientid 1001 bulan ini","clientid 1002 Oktober 2025"],
                "choices": _mk_choices(["clientid 1001 bulan ini","clientid 1002 Oktober 2025"])
            }

        # default periode kalau user tidak sebut
        if not year or not month:
            today = date.today()
            year, month = today.year, today.month

        with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
            current = _fetch_month_summary(cur, cid, year, month, channel=channel)

            # simpan konteks supaya langkah berikutnya bisa pakai
            SESSION[sid] = {"clientid": cid, "year": year, "month": month}

            # ---- STEP 1: belum ada channel → header + tombol pilihan real
            if channel is None:
                t = current["totals"]
                header = (
                    f"**{current['merchant']}** • Periode **"
                    f"{calendar.month_name[current['month']]} {current['year']}**\n"
                    f"Total: **{t['total_tx']}** trx • SR agregat: **{t['success_rate_pct']:.2f}%** • "
                    f"GMV: **Rp {t['gmv']:,}**\n\n"
                    f"Untuk channel mana yang mau dicek?"
                )

                chans = [r["channel"] for r in current["by_channel"] if r["channel"]]
                if not chans:
                    return {
                        "reply": header + "\n(Tidak ada pembagian channel pada periode ini.)",
                        "suggestions":["bandingkan dengan bulan sebelumnya","bulan kemarin"],
                        "actions":["bandingkan dengan bulan sebelumnya","bulan kemarin"],
                        "choices": _mk_choices(["bandingkan dengan bulan sebelumnya","bulan kemarin"])
                    }

                chan_labels = [f"channel {c}" for c in chans]  # contoh: "channel QRIS"
                # tambahkan opsi umum
                chan_labels += ["bandingkan dengan bulan sebelumnya","bulan ini","bulan kemarin"]

                return {
                    "reply": header,
                    "suggestions": chan_labels,
                    "actions": chan_labels,
                    "choices": _mk_choices(chan_labels)
                }

            # ---- STEP 2: sudah ada channel → detail + AI
            prev = None
            if compare:
                py, pm = _prev_month(year, month)
                prev = _fetch_month_summary(cur, cid, py, pm, channel=channel)

        md_head = _md_stat_block(current)
        ai_md = _ai_summary({"current": current, "previous": prev})
        reply = f"{md_head}\n\n{ai_md}"

        # tombol lanjutan
        other = [r["channel"] for r in current["by_channel"] if r["channel"] and r["channel"] != channel]
        next_labels = [f"channel {c}" for c in other] + \
                      ["bandingkan dengan bulan sebelumnya","lihat detail by day","bulan ini","bulan kemarin"]

        return {
            "reply": reply,
            "suggestions": next_labels,
            "actions": next_labels,
            "choices": _mk_choices(next_labels)
        }

    except Exception as e:
        logger.exception("chat handler failed")
        return {
            "reply": f"Terjadi error di sisi server: **{type(e).__name__}**. Coba lagi ya.",
            "suggestions":[], "actions":[], "choices":[]
        }


# ====== end /chat ======






from fastapi import Body

@app.post("/agent/sr")
def agent_sr(payload: dict = Body(...)):
    """
    payload = { "q": "Hi Merge, berapa SR clientid 1234 CC bulan Agustus?" }
    """
    q = (payload or {}).get("q","")
    parsed = parse_user_query(q)

    if not parsed["merchant_id"]:
        # Kalau user belum sebut clientid, beri jawaban langsung & contoh
        return {
            "reply": ("Aku butuh *clientid*-nya ya. Contoh: "
                      "`SR clientid 1001 channel QRIS bulan September`."),
            "meta": {"need_client_id": True}
        }

    mid     = parsed["merchant_id"]
    start   = parsed["start"]
    end     = parsed["end"]
    channel = parsed["channel"]

    with conn() as c, c.cursor(cursor_factory=DictCursor) as cur:
        if channel:
            sql, params = _sr_sql_and_params(mid, start, end, channel)
            cur.execute(sql, params)
            row = cur.fetchone()
            total   = int(row["total"])
            success = int(row["success"])
            sr      = float(row["sr"] or 0.0)
            when = f"{start.strftime('%B %Y')}"
            ch = channel
            reply = (f"MERGE: Hasil untuk **client {mid}**, **channel {ch}**, "
                     f"periode **{when}** → Success Rate: **{sr:.2f}%** "
                     f"({success}/{total}).")
            return {"reply": reply, "data": {"total":total,"success":success,"sr":sr}}
        else:
            (sql_all, p_all), (sql_ch, p_ch) = _sr_sql_and_params(mid, start, end, None)
            cur.execute(sql_all, p_all)
            o = cur.fetchone()
            total   = int(o["total"])
            success = int(o["success"])
            sr      = float(o["sr"] or 0.0)

            cur.execute(sql_ch, p_ch)
            bych = cur.fetchall()
            lines = []
            for r in bych:
                lines.append(f"- {r['channel']}: {float(r['sr'] or 0.0):.2f}% ({int(r['success'])}/{int(r['total'])})")
            when = f"{start.strftime('%B %Y')}"
            reply = (
              f"MERGE: **client {mid}** periode **{when}**\n"
              f"• Overall SR: **{sr:.2f}%** ({success}/{total})\n"
              + ("• Per-channel:\n" + "\n".join(lines) if lines else "• Per-channel: (tidak ada data)")
            )
            return {"reply": reply, "data": {"overall":{"total":total,"success":success,"sr":sr}, "by_channel": bych}}





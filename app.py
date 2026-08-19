import os
import io
import re
import json
import math
import time
import base64
import hashlib
import secrets as pysecrets
import sqlite3
import threading
import textwrap
from pathlib import Path
from datetime import datetime, date, timedelta

import asyncio
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
from groq import Groq, AuthenticationError, RateLimitError, APIConnectionError, BadRequestError
try:
    import edge_tts
    EDGE_TTS_OK = True
except Exception:
    edge_tts = None
    EDGE_TTS_OK = False

# ============================================================
# IA KSC / NutriVision MGP — V6.0 (rediseño)
# EUREKA 2026
# ============================================================

APP_VERSION = "6.0"
AI_MODEL = "qwen/qwen3.6-27b"
USDA_BASE = "https://api.nal.usda.gov/fdc/v1"
OFF_BASE = "https://world.openfoodfacts.org/api/v2/product"

DATA_DIR = Path(".ksc_data")
PROFILE_DIR = DATA_DIR / "profile_photos"
MEAL_DIR = DATA_DIR / "meal_photos"
PUSHUP_DIR = DATA_DIR / "pushup_videos"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "ksc.db"
BARCODE_CACHE_PATH = DATA_DIR / "barcode_cache.json"
COMMUNITY_PATH = DATA_DIR / "community_profiles.json"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

for d in (DATA_DIR, PROFILE_DIR, MEAL_DIR, PUSHUP_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="IA KSC · NutriVision",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# NOTA DE PERSISTENCIA
# ============================================================
# Todo (perfiles, comidas, puntos, retos) se guarda en .ksc_data/ksc.db
# (SQLite) en el disco donde corre la app. Mientras esa carpeta no se
# borre, los perfiles NUNCA se pierden, aunque cierres la pestaña o
# reinicies el navegador. Si despliegas esto en un hosting con disco
# "efímero" (se borra en cada reinicio del servidor), debes montar un
# volumen persistente apuntando a .ksc_data — si no, el hosting es el
# que borra los datos, no la app.

# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{
 --bg:#060f0b;--panel:#0b1b13;--panel2:#0e2117;--line:rgba(255,255,255,.09);
 --text:#f4fff8;--muted:#9cb7a8;--green:#56f09f;--green2:#22c98a;--yellow:#ffd166;
 --blue:#6ab8ff;--purple:#b28dff;--orange:#ff9d5c;--red:#ff6b7a;
 --radius:20px;--ease:cubic-bezier(.16,1,.3,1);
}
*{ scrollbar-width:thin; scrollbar-color:rgba(86,240,159,.35) transparent; }
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:rgba(86,240,159,.35);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:rgba(86,240,159,.55)}

html, body, [class*="css"], .stApp{
 font-family:'Manrope','Plus Jakarta Sans',-apple-system,sans-serif;
}

@keyframes fadeInUp{ from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn{ from{opacity:0} to{opacity:1} }
@keyframes floatSoft{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
@keyframes pulseGlow{
 0%,100%{box-shadow:0 0 0 0 rgba(86,240,159,.35)}
 50%{box-shadow:0 0 0 10px rgba(86,240,159,0)}
}
@keyframes flameFlicker{
 0%,100%{transform:scale(1) rotate(-2deg)}
 25%{transform:scale(1.08) rotate(2deg)}
 50%{transform:scale(0.96) rotate(-1deg)}
 75%{transform:scale(1.05) rotate(1deg)}
}
@keyframes gradientShift{
 0%{background-position:0% 50%}
 50%{background-position:100% 50%}
 100%{background-position:0% 50%}
}
@keyframes popIn{
 0%{opacity:0;transform:scale(.85)}
 70%{transform:scale(1.03)}
 100%{opacity:1;transform:scale(1)}
}

.stApp{
 background:
 radial-gradient(circle at 8% 0%,rgba(86,240,159,.14),transparent 30%),
 radial-gradient(circle at 94% 5%,rgba(106,184,255,.09),transparent 24%),
 radial-gradient(circle at 50% 100%,rgba(178,141,255,.06),transparent 35%),
 linear-gradient(180deg,#060f0b 0%,#07150f 75%);
 animation:fadeIn .5s var(--ease);
}
.block-container{max-width:1320px;padding-top:1.1rem;padding-bottom:4rem}
.block-container > div{ animation:fadeInUp .45s var(--ease) both; }

[data-testid="stSidebar"]{
 background:linear-gradient(180deg,#07140e,#091a12);
 border-right:1px solid var(--line);
}
[data-testid="stSidebar"] *{color:#effff6}
[data-testid="stSidebar"] [role="radiogroup"] label{
 border-radius:14px;padding:11px 14px;margin-bottom:5px;
 transition:all .22s var(--ease);
 border:1px solid transparent;font-size:1.02rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
 background:rgba(86,240,159,.10);border-color:rgba(86,240,159,.22);
 transform:translateX(3px);
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
 background:linear-gradient(90deg,rgba(86,240,159,.18),rgba(86,240,159,.03));
 border-color:rgba(86,240,159,.35);
 box-shadow:0 4px 14px rgba(86,240,159,.10);
}

h1,h2,h3{letter-spacing:-.035em}

.hero{
 padding:34px 36px;border:1px solid var(--line);border-radius:28px;position:relative;overflow:hidden;
 background:radial-gradient(circle at 84% 18%,rgba(86,240,159,.19),transparent 28%),
 linear-gradient(135deg,rgba(17,44,29,.97),rgba(6,17,12,.98));
 box-shadow:0 24px 80px rgba(0,0,0,.28);margin-bottom:20px;
 animation:fadeInUp .55s var(--ease) both;
}
.hero::after{
 content:"";position:absolute;inset:0;pointer-events:none;
 background:linear-gradient(120deg,transparent 30%,rgba(86,240,159,.06) 50%,transparent 70%);
 background-size:220% 220%;animation:gradientShift 9s ease-in-out infinite;
}
.badge{
 display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:999px;
 background:rgba(86,240,159,.09);border:1px solid rgba(86,240,159,.25);
 color:#87f6ba;font-size:.80rem;font-weight:850;position:relative;z-index:1;
}
.badge::before{
 content:"";width:7px;height:7px;border-radius:99px;background:var(--green);
 animation:pulseGlow 1.8s ease-in-out infinite;
}
.hero-title{font-size:3.3rem;font-weight:950;line-height:1;letter-spacing:-.055em;color:white;margin:15px 0 10px;position:relative;z-index:1}
.green{
 background:linear-gradient(90deg,var(--green),#8bffce,var(--green));
 background-size:200% auto;-webkit-background-clip:text;background-clip:text;color:transparent;
 animation:gradientShift 4s linear infinite;
}
.hero-sub{color:#b8d3c4;max-width:920px;font-size:1.04rem;line-height:1.6;position:relative;z-index:1}

.mini{
 height:100%;background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:18px;padding:17px;
 transition:transform .28s var(--ease),border-color .28s var(--ease),box-shadow .28s var(--ease);
}
.mini:hover{transform:translateY(-4px);border-color:rgba(86,240,159,.30);box-shadow:0 16px 40px rgba(0,0,0,.22)}

.kicker{font-size:.74rem;font-weight:900;letter-spacing:.11em;color:#77ecab}
.big{font-size:1.55rem;font-weight:900;color:white;margin-top:4px}
.note{font-size:.84rem;color:var(--muted);line-height:1.5;margin-top:4px}

.chip{
 display:inline-block;padding:6px 10px;margin:3px;border-radius:999px;
 background:rgba(86,240,159,.08);border:1px solid rgba(86,240,159,.18);font-size:.78rem;font-weight:800;
 transition:all .2s var(--ease);
}
.chip:hover{background:rgba(86,240,159,.16);transform:translateY(-1px)}

.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 22px}
@media (max-width:900px){ .stat-grid{grid-template-columns:repeat(2,1fr)} }
.stat-card{
 position:relative;overflow:hidden;border-radius:20px;padding:18px 18px 16px;
 background:linear-gradient(160deg,rgba(255,255,255,.045),rgba(255,255,255,.015));
 border:1px solid var(--line);animation:popIn .5s var(--ease) both;
 transition:transform .25s var(--ease),box-shadow .25s var(--ease),border-color .25s var(--ease);
}
.stat-card:hover{transform:translateY(-5px) scale(1.015);box-shadow:0 18px 46px rgba(0,0,0,.30);border-color:rgba(86,240,159,.30)}
.stat-card .icon{font-size:1.5rem;display:inline-block;animation:floatSoft 3.4s ease-in-out infinite}
.stat-card .label{font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:8px}
.stat-card .value{font-size:1.7rem;font-weight:950;color:white;letter-spacing:-.03em;margin-top:2px}
.stat-card .sub{font-size:.78rem;color:var(--muted);margin-top:3px}
.stat-card.accent-green{border-color:rgba(86,240,159,.22)}
.stat-card.accent-blue{border-color:rgba(106,184,255,.22)}
.stat-card.accent-purple{border-color:rgba(178,141,255,.22)}
.stat-card.accent-orange{border-color:rgba(255,157,92,.22)}

.streak-card{
 display:flex;align-items:center;gap:14px;padding:16px 20px;border-radius:20px;
 background:linear-gradient(120deg,rgba(255,157,92,.14),rgba(255,209,102,.05));
 border:1px solid rgba(255,157,92,.30);animation:popIn .55s var(--ease) both;
}
.streak-flame{font-size:2.1rem;animation:flameFlicker 1.6s ease-in-out infinite;filter:drop-shadow(0 0 10px rgba(255,157,92,.55))}
.streak-days{font-size:1.6rem;font-weight:950;color:#ffcf9a;letter-spacing:-.03em;line-height:1}
.streak-label{font-size:.78rem;color:var(--muted);font-weight:700;margin-top:2px}

.level-card{
 padding:20px;border-radius:22px;
 background:linear-gradient(135deg,rgba(86,240,159,.10),rgba(178,141,255,.06));
 border:1px solid rgba(86,240,159,.20);position:relative;overflow:hidden;
 animation:fadeInUp .5s var(--ease) both;transition:transform .25s var(--ease);
}
.level-card:hover{transform:translateY(-3px)}

.water-alert{
 padding:14px 16px;border-radius:16px;border:1px solid rgba(106,184,255,.28);
 background:rgba(106,184,255,.08);animation:fadeInUp .4s var(--ease) both;
}
.lock{padding:22px;border-radius:20px;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.03);animation:fadeInUp .4s var(--ease) both}

div[data-testid="stMetric"]{
 background:rgba(255,255,255,.028);border:1px solid var(--line);padding:14px 16px;border-radius:18px;
 transition:transform .25s var(--ease),border-color .25s var(--ease);
}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:rgba(86,240,159,.28)}
div[data-testid="stMetricValue"]{font-weight:900}

.stButton>button{
 border-radius:16px!important;font-weight:850!important;font-size:1.02rem!important;
 padding:.6rem 1.1rem!important;
 transition:transform .18s var(--ease),box-shadow .18s var(--ease),filter .18s var(--ease)!important;
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(86,240,159,.18);filter:brightness(1.05)}
.stButton>button:active{transform:translateY(0) scale(.98)}
button[kind="primary"]{
 background:linear-gradient(90deg,var(--green2),var(--green))!important;
 background-size:180% auto!important;
}
button[kind="primary"]:hover{background-position:right center!important}

div[data-testid="stProgress"] > div > div{
 background:linear-gradient(90deg,var(--green2),var(--green),#8bffce)!important;
 background-size:200% auto!important;animation:gradientShift 3s linear infinite;
 border-radius:99px!important;
}
div[data-testid="stProgress"]{border-radius:99px;overflow:hidden}

div[data-testid="stVerticalBlockBorderWrapper"]{
 transition:transform .25s var(--ease),box-shadow .25s var(--ease);
 border-radius:18px!important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-2px)}

.stTabs [data-baseweb="tab-list"]{gap:4px}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0!important;transition:all .2s var(--ease);font-weight:700}
.stTabs [aria-selected="true"]{background:rgba(86,240,159,.10)!important}

@media (max-width:768px){
 .hero{padding:24px 20px;border-radius:22px}
 .hero-title{font-size:2.1rem}
 .hero-sub{font-size:.92rem}
 .block-container{padding-top:.6rem}
 .stat-grid{grid-template-columns:repeat(2,1fr);gap:10px}
 .stat-card .value{font-size:1.35rem}
 .streak-card{padding:12px 16px}
}

footer{visibility:hidden}
#MainMenu{visibility:hidden}

.unlock-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:8px}
.unlock-card{
 border-radius:16px;padding:14px;text-align:center;border:1px solid var(--line);
 background:rgba(255,255,255,.02);transition:all .3s var(--ease);animation:popIn .45s var(--ease) both;
}
.unlock-card.open{background:linear-gradient(160deg,rgba(86,240,159,.14),rgba(86,240,159,.02));border-color:rgba(86,240,159,.35)}
.unlock-card.open:hover{transform:translateY(-4px) scale(1.02);box-shadow:0 14px 34px rgba(86,240,159,.14)}
.unlock-card.closed{opacity:.55;filter:grayscale(.35)}
.unlock-card .uicon{font-size:1.8rem;display:block;margin-bottom:6px}
.unlock-card .uname{font-size:.86rem;font-weight:800;color:white}
.unlock-card .ureq{font-size:.72rem;color:var(--muted);margin-top:3px}

.winner-banner{
 padding:20px 24px;border-radius:20px;text-align:center;margin:10px 0;
 background:linear-gradient(120deg,rgba(255,209,102,.16),rgba(86,240,159,.08));
 border:1px solid rgba(255,209,102,.35);animation:popIn .5s var(--ease) both;
}
.winner-banner .wtitle{font-size:1.5rem;font-weight:950;color:#ffe4a3}

.avatar-ring{
 width:86px;height:86px;border-radius:99px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,var(--green2),var(--blue));font-weight:950;font-size:1.7rem;color:#04140c;
 box-shadow:0 8px 26px rgba(86,240,159,.25);margin:0 auto 10px;border:3px solid rgba(255,255,255,.12);
}
.avatar-photo{
 width:86px;height:86px;border-radius:99px;object-fit:cover;display:block;margin:0 auto 10px;
 border:3px solid rgba(86,240,159,.35);box-shadow:0 8px 26px rgba(0,0,0,.3);
}

.rank-row{
 display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:14px;margin-bottom:6px;
 background:rgba(255,255,255,.025);border:1px solid var(--line);transition:transform .2s var(--ease);
}
.rank-row:hover{transform:translateX(3px)}
.rank-row.top1{background:linear-gradient(90deg,rgba(255,209,102,.14),transparent);border-color:rgba(255,209,102,.3)}
.rank-row.top2{background:linear-gradient(90deg,rgba(200,210,220,.12),transparent);border-color:rgba(200,210,220,.25)}
.rank-row.top3{background:linear-gradient(90deg,rgba(255,157,92,.12),transparent);border-color:rgba(255,157,92,.25)}
.rank-pos{font-weight:950;font-size:1.05rem;min-width:28px}
.rank-name{flex:1;font-weight:700}
.rank-val{font-weight:900;color:var(--green)}

.side-profile{
 padding:14px;border-radius:16px;background:rgba(86,240,159,.06);
 border:1px solid rgba(86,240,159,.18);text-align:center;margin-bottom:10px;
 animation:fadeInUp .4s var(--ease) both;
}
.side-avatar{width:52px;height:52px;border-radius:99px;margin:0 auto 8px;object-fit:cover;border:2px solid rgba(86,240,159,.4)}
.side-avatar-fallback{
 width:52px;height:52px;border-radius:99px;margin:0 auto 8px;
 background:linear-gradient(135deg,var(--green2),var(--blue));
 display:flex;align-items:center;justify-content:center;font-weight:900;color:#04140c;
}
.brand-row{display:flex;align-items:center;gap:10px;margin-bottom:2px}
.brand-icon{
 width:36px;height:36px;border-radius:11px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,var(--green2),var(--blue));font-size:1.15rem;
 animation:floatSoft 3.6s ease-in-out infinite;
}

.glass-btn{
 border-radius:16px;padding:14px 8px;text-align:center;border:1px solid rgba(106,184,255,.25);
 background:rgba(106,184,255,.06);font-weight:800;font-size:.85rem;
}

.community-card{
 border-radius:18px;padding:14px;border:1px solid var(--line);background:rgba(255,255,255,.025);
 text-align:center;transition:transform .25s var(--ease);
}
.community-card:hover{transform:translateY(-4px)}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DB
# ============================================================

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}

def ensure_col(con, table, definition):
    name = definition.split()[0]
    if name not in cols(con, table):
        con.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS profiles(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL, age INTEGER NOT NULL, sex_energy TEXT NOT NULL,
      height_cm REAL NOT NULL, weight_kg REAL NOT NULL,
      activity TEXT NOT NULL, goal TEXT NOT NULL,
      favorite_foods TEXT DEFAULT '', favorite_fruits TEXT DEFAULT '',
      favorite_vegetables TEXT DEFAULT '', avoid_foods TEXT DEFAULT '',
      allergies TEXT DEFAULT '', special_state TEXT DEFAULT 'Ninguno',
      photo_path TEXT DEFAULT '', created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS weight_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      log_date TEXT NOT NULL, weight_kg REAL NOT NULL, waist_cm REAL, note TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS measurements(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      log_date TEXT NOT NULL, waist_cm REAL, hip_cm REAL, chest_cm REAL,
      arm_cm REAL, thigh_cm REAL, note TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS chat_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memories(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      memory TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS meal_diary(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      meal_date TEXT NOT NULL, meal_time TEXT NOT NULL, meal_type TEXT NOT NULL,
      title TEXT DEFAULT '', foods_json TEXT DEFAULT '[]',
      kcal REAL DEFAULT 0, protein REAL DEFAULT 0, carbs REAL DEFAULT 0,
      fat REAL DEFAULT 0, fiber REAL DEFAULT 0, sugars REAL DEFAULT 0,
      sodium_mg REAL DEFAULT 0, sat_fat REAL DEFAULT 0,
      image_path TEXT DEFAULT '', note TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS hydration(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      log_date TEXT NOT NULL, logged_at TEXT NOT NULL, ml INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS healthy_goals(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      goal_date TEXT NOT NULL, goal_name TEXT NOT NULL, completed INTEGER DEFAULT 0,
      UNIQUE(profile_id, goal_date, goal_name)
    );
    CREATE TABLE IF NOT EXISTS point_events(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      event_date TEXT NOT NULL, points INTEGER NOT NULL, reason TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS favorites(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      title TEXT NOT NULL, recipe TEXT NOT NULL, category TEXT DEFAULT '', created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS recipe_ratings(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      favorite_id INTEGER, recipe_title TEXT NOT NULL, rating INTEGER NOT NULL,
      comment TEXT DEFAULT '', created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS weekly_plans(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      week_start TEXT NOT NULL, plan_json TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS plate_tests(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER,
      created_at TEXT NOT NULL, predicted_foods TEXT, actual_foods TEXT,
      correct INTEGER, avg_conf REAL, estimated_kcal REAL
    );
    CREATE TABLE IF NOT EXISTS pushup_challenges(
      id INTEGER PRIMARY KEY AUTOINCREMENT, challenger_id INTEGER NOT NULL,
      opponent_id INTEGER NOT NULL, created_at TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending', duration_seconds INTEGER NOT NULL DEFAULT 60
    );
    CREATE TABLE IF NOT EXISTS pushup_attempts(
      id INTEGER PRIMARY KEY AUTOINCREMENT, challenge_id INTEGER NOT NULL,
      profile_id INTEGER NOT NULL, reps INTEGER NOT NULL, duration_seconds REAL NOT NULL,
      video_path TEXT DEFAULT '', created_at TEXT NOT NULL,
      UNIQUE(challenge_id, profile_id)
    );
    CREATE TABLE IF NOT EXISTS quiz_results(
      id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
      quiz_date TEXT NOT NULL, level TEXT DEFAULT 'Básico', correct INTEGER NOT NULL, total INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS direct_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT, from_id INTEGER NOT NULL, to_id INTEGER NOT NULL,
      content TEXT NOT NULL, created_at TEXT NOT NULL
    );
    """)
    ensure_col(con, "profiles", "pin_hash TEXT DEFAULT ''")
    ensure_col(con, "profiles", "water_goal_ml INTEGER DEFAULT 2000")
    ensure_col(con, "plate_tests", "app_version TEXT DEFAULT ''")
    ensure_col(con, "quiz_results", "level TEXT DEFAULT 'Básico'")
    con.commit()
    con.close()

init_db()

# ============================================================
# SEGURIDAD
# ============================================================

def hash_pin(pin, stored=None):
    if stored and "$" in stored:
        salt_hex, _ = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    else:
        salt = pysecrets.token_bytes(16)
        salt_hex = salt.hex()
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 180_000).hex()
    return f"{salt_hex}${digest}"

def verify_pin(pin, stored):
    if not stored:
        return True
    try:
        return hash_pin(pin, stored) == stored
    except Exception:
        return False

# ============================================================
# PERFILES
# ============================================================

ACTIVITY_FACTORS = {"Sedentario":1.20,"Ligero":1.375,"Moderado":1.55,"Alto":1.725}

def list_profiles():
    con=db()
    rows=con.execute("SELECT * FROM profiles ORDER BY name COLLATE NOCASE").fetchall()
    con.close()
    return [dict(r) for r in rows]

def get_profile(pid):
    if not pid: return None
    con=db()
    r=con.execute("SELECT * FROM profiles WHERE id=?",(pid,)).fetchone()
    con.close()
    return dict(r) if r else None

def create_profile(d):
    con=db()
    cur=con.execute("""
      INSERT INTO profiles(
        name,age,sex_energy,height_cm,weight_kg,activity,goal,
        favorite_foods,favorite_fruits,favorite_vegetables,avoid_foods,
        allergies,special_state,photo_path,created_at,pin_hash,water_goal_ml
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        d["name"],d["age"],d["sex_energy"],d["height_cm"],d["weight_kg"],
        d["activity"],d["goal"],d["favorite_foods"],d["favorite_fruits"],
        d["favorite_vegetables"],d["avoid_foods"],d["allergies"],
        d["special_state"],d["photo_path"],datetime.now().isoformat(timespec="seconds"),
        d["pin_hash"],d["water_goal_ml"]
    ))
    pid=cur.lastrowid
    con.execute("INSERT INTO weight_logs(profile_id,log_date,weight_kg,note) VALUES(?,?,?,?)",
                (pid,str(date.today()),d["weight_kg"],"Peso inicial"))
    con.commit();con.close()
    sync_community()
    return pid

def update_profile(pid,d):
    con=db()
    con.execute("""
      UPDATE profiles SET
       name=?,age=?,sex_energy=?,height_cm=?,weight_kg=?,activity=?,goal=?,
       favorite_foods=?,favorite_fruits=?,favorite_vegetables=?,avoid_foods=?,
       allergies=?,special_state=?,photo_path=?,water_goal_ml=?
      WHERE id=?
    """,(
        d["name"],d["age"],d["sex_energy"],d["height_cm"],d["weight_kg"],
        d["activity"],d["goal"],d["favorite_foods"],d["favorite_fruits"],
        d["favorite_vegetables"],d["avoid_foods"],d["allergies"],
        d["special_state"],d["photo_path"],d["water_goal_ml"],pid
    ))
    con.commit();con.close()
    sync_community()

def delete_profile(pid):
    con=db()
    for t in ["weight_logs","measurements","chat_messages","memories","meal_diary","hydration",
              "healthy_goals","point_events","favorites","recipe_ratings","weekly_plans","plate_tests","quiz_results"]:
        con.execute(f"DELETE FROM {t} WHERE profile_id=?",(pid,))
    con.execute("DELETE FROM pushup_attempts WHERE profile_id=?",(pid,))
    con.execute("DELETE FROM pushup_challenges WHERE challenger_id=? OR opponent_id=?",(pid,pid))
    con.execute("DELETE FROM direct_messages WHERE from_id=? OR to_id=?",(pid,pid))
    con.execute("DELETE FROM profiles WHERE id=?",(pid,))
    con.commit();con.close()
    sync_community()

# ============================================================
# COMUNIDAD (archivo público sin PIN, para "ver otros perfiles")
# ============================================================

def sync_community():
    """Guarda un archivo con nombre + foto de todos los perfiles (nunca el PIN)."""
    try:
        data=[]
        for p in list_profiles():
            data.append({
                "id": p["id"], "name": p["name"], "goal": p["goal"],
                "photo_path": p.get("photo_path",""),
            })
        COMMUNITY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def read_community():
    if COMMUNITY_PATH.exists():
        try:
            return json.loads(COMMUNITY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

# ============================================================
# ARCHIVOS
# ============================================================

def compact_jpeg(raw,max_side=1400,quality=84):
    img=Image.open(io.BytesIO(raw))
    img=ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    out=io.BytesIO()
    img.save(out,"JPEG",quality=quality,optimize=True)
    return out.getvalue()

def save_jpeg(raw,folder,prefix,max_side=1000):
    data=compact_jpeg(raw,max_side=max_side,quality=86)
    path=folder/f"{prefix}_{int(time.time()*1000)}.jpg"
    path.write_bytes(data)
    return str(path)

def data_url(jpeg):
    return "data:image/jpeg;base64,"+base64.b64encode(jpeg).decode()

# ============================================================
# PUNTOS / NIVELES / RACHA
# ============================================================

LEVELS=[
    (0,"Semilla KSC","🌱"),(100,"Explorador KSC","🧭"),
    (250,"NutriRanger","🥗"),(500,"Maestro KSC","🏅"),
    (900,"Leyenda KSC","👑"),(1500,"Elite KSC","💎")
]

def add_points(pid,points,reason):
    con=db()
    con.execute("INSERT INTO point_events(profile_id,event_date,points,reason) VALUES(?,?,?,?)",
                (pid,str(date.today()),int(points),reason))
    con.commit();con.close()

def total_points(pid):
    con=db();r=con.execute("SELECT COALESCE(SUM(points),0) p FROM point_events WHERE profile_id=?",(pid,)).fetchone();con.close()
    return int(r["p"])

def level_info(pid):
    pts=total_points(pid);cur=LEVELS[0];nxt=None
    for l in LEVELS:
        if pts>=l[0]: cur=l
        elif nxt is None: nxt=l;break
    return pts,cur,nxt

def streak(pid):
    con=db();rows=con.execute("SELECT DISTINCT event_date FROM point_events WHERE profile_id=? ORDER BY event_date DESC",(pid,)).fetchall();con.close()
    dates={date.fromisoformat(r["event_date"]) for r in rows}
    if not dates:return 0
    d=date.today()
    if d not in dates and d-timedelta(days=1) in dates:d-=timedelta(days=1)
    n=0
    while d in dates:n+=1;d-=timedelta(days=1)
    return n

def leaderboard():
    con=db()
    df=pd.read_sql_query("""
      SELECT p.name,COALESCE(SUM(e.points),0) points
      FROM profiles p LEFT JOIN point_events e ON e.profile_id=p.id
      GROUP BY p.id,p.name ORDER BY points DESC,p.name
    """,con);con.close()
    return df

# ============================================================
# AGUA (vasos/botellas en vez de ml crudos) / RETOS
# ============================================================

DEFAULT_GOALS=[
    "Registrar mis comidas","Incluir una fruta","Incluir verduras",
    "Tomar agua durante el día","Elegir una bebida sin azúcar añadida"
]

WATER_UNITS = [
    ("½ vaso", 125, "🥃"),
    ("1 vaso", 250, "🥛"),
    ("2 vasos", 500, "🥛🥛"),
    ("1 botella", 750, "🧴"),
]

def water_today(pid):
    con=db();r=con.execute("SELECT COALESCE(SUM(ml),0) ml FROM hydration WHERE profile_id=? AND log_date=?",(pid,str(date.today()))).fetchone();con.close()
    return int(r["ml"])

def log_water(pid,ml):
    con=db()
    con.execute("INSERT INTO hydration(profile_id,log_date,logged_at,ml) VALUES(?,?,?,?)",
                (pid,str(date.today()),datetime.now().isoformat(timespec="seconds"),int(ml)))
    con.commit();con.close()
    add_points(pid,3,f"Agua +{ml} ml")

def daily_goals(pid):
    con=db()
    for g in DEFAULT_GOALS:
        con.execute("INSERT OR IGNORE INTO healthy_goals(profile_id,goal_date,goal_name,completed) VALUES(?,?,?,0)",
                    (pid,str(date.today()),g))
    con.commit()
    rows=con.execute("SELECT * FROM healthy_goals WHERE profile_id=? AND goal_date=? ORDER BY id",
                     (pid,str(date.today()))).fetchall()
    con.close()
    return [dict(r) for r in rows]

def complete_goal(gid,pid):
    con=db();r=con.execute("SELECT completed FROM healthy_goals WHERE id=?",(gid,)).fetchone()
    if r and not r["completed"]:
        con.execute("UPDATE healthy_goals SET completed=1 WHERE id=?",(gid,))
        con.commit();con.close();add_points(pid,10,"Reto saludable completado");return
    con.close()

# ============================================================
# ENERGÍA / PERFIL
# ============================================================

def energy_estimate(p):
    if not p:return {"enabled":False,"reason":"Sin perfil."}
    if int(p["age"])<18:
        return {"enabled":False,"reason":"En menores de 18 años IA KSC no fija déficit, superávit ni metas calóricas para cambiar de peso."}
    if p.get("special_state") in ("Embarazo","Lactancia"):
        return {"enabled":False,"reason":"En embarazo o lactancia no se fija una meta calórica personalizada."}
    sex=p.get("sex_energy")
    if sex not in ("Masculino","Femenino"):
        return {"enabled":False,"reason":"Selecciona la variable fisiológica usada por la ecuación para estimar energía."}
    w=float(p["weight_kg"]);h=float(p["height_cm"]);a=int(p["age"])
    bmr=10*w+6.25*h-5*a+(5 if sex=="Masculino" else -161)
    maintenance=bmr*ACTIVITY_FACTORS.get(p.get("activity"),1.375)
    goal=p.get("goal")
    if goal=="Perder peso":low,high=maintenance-400,maintenance-250
    elif goal in ("Ganar peso","Ganar masa muscular"):low,high=maintenance+150,maintenance+300
    else:low,high=maintenance-100,maintenance+100
    low=max(1200,low);high=max(low,high)
    bmi=w/((h/100)**2)
    return {"enabled":True,"maintenance":round(maintenance),"target_low":round(low),"target_high":round(high),"bmi":round(bmi,1)}

def memories(pid,limit=12):
    con=db();rows=con.execute("SELECT memory FROM memories WHERE profile_id=? ORDER BY id DESC LIMIT ?",(pid,limit)).fetchall();con.close()
    return [r["memory"] for r in rows]

def maybe_memory(pid,text):
    low=text.lower()
    keys=["me gusta ","me encanta ","prefiero ","no me gusta ","no como ","evito ","soy alérg","soy alerg"]
    if any(k in low for k in keys) and len(text.strip())<=300:
        con=db()
        if not con.execute("SELECT 1 FROM memories WHERE profile_id=? AND memory=?",(pid,text.strip())).fetchone():
            con.execute("INSERT INTO memories(profile_id,memory,created_at) VALUES(?,?,?)",
                        (pid,text.strip(),datetime.now().isoformat(timespec="seconds")));con.commit()
        con.close()

def profile_context(p):
    e=energy_estimate(p)
    energy=(f"Mantenimiento aprox. {e['maintenance']} kcal/día; rango orientativo {e['target_low']}–{e['target_high']} kcal/día."
            if e.get("enabled") else "No dar meta calórica exacta: "+e.get("reason",""))
    return f"""
PERFIL ACTIVO:
Nombre: {p['name']}
Edad: {p['age']}
Talla: {p['height_cm']} cm
Peso: {p['weight_kg']} kg
Actividad: {p['activity']}
Objetivo: {p['goal']}
Comidas favoritas: {p.get('favorite_foods','')}
Frutas favoritas: {p.get('favorite_fruits','')}
Verduras favoritas: {p.get('favorite_vegetables','')}
Evita/no le gustan: {p.get('avoid_foods','')}
Alergias/restricciones: {p.get('allergies','')}
Estado especial: {p.get('special_state','Ninguno')}
Meta de agua: {p.get('water_goal_ml',2000)} ml
Energía: {energy}
Memoria: {'; '.join(memories(p['id'])) or 'sin recuerdos adicionales'}
"""

# ============================================================
# IA
# ============================================================

def secret(name,default=""):
    try:
        v=st.secrets.get(name,"")
        if v:return str(v)
    except Exception:pass
    return os.getenv(name,default)

def ai_key():return secret("GROQ_API_KEY","")
def usda_key():return secret("USDA_API_KEY","DEMO_KEY")

@st.cache_resource
def ai_client(key):
    return Groq(api_key=key,timeout=60.0,max_retries=2)

def system_prompt(p):
    return f"""
Eres IA KSC, asistente nutricional educativo de NutriVision.

Si preguntan qué es IA KSC o quién la diseñó, responde:
"IA KSC es un asistente nutricional educativo diseñado por los alumnos César Zapata, Alex Timaná García, Atarama Portocarrero y André Requena."
No menciones al proveedor técnico salvo que pregunten expresamente por la infraestructura.

SOLO HABLAS DE: alimentación, nutrición general, calorías, platos, porciones, recetas,
jugos, postres, frutas, verduras, etiquetas, códigos de barra, compras, preparación y hábitos alimentarios.
Si cambian de tema, responde brevemente que IA KSC se especializa en alimentación.

Usa siempre el perfil. Respeta alergias, gustos y alimentos evitados.
Puedes crear recetas, planes, alternativas y listas de compras.
Calorías siempre aproximadas cuando no exista peso real.
Responde completo: no cortes tu respuesta a la mitad, termina siempre la idea.

No diagnostiques. No digas que eres médico/nutricionista. No prescribas fármacos/suplementos.
No recomiendes vómitos, laxantes, deshidratación, ayunos prolongados ni dietas extremas.
Si el perfil es menor de 18 años, no des déficit, superávit, objetivo calórico para cambiar de peso ni dieta restrictiva.
En embarazo/lactancia no fijes metas calóricas personalizadas.
Nunca uses una foto del usuario para inferir salud, grasa corporal o peso.

{profile_context(p)}
"""

def get_chat(pid,limit=24):
    con=db();rows=con.execute("SELECT role,content FROM chat_messages WHERE profile_id=? ORDER BY id DESC LIMIT ?",(pid,limit)).fetchall();con.close()
    return [dict(r) for r in reversed(rows)]

def add_chat(pid,role,content):
    con=db();con.execute("INSERT INTO chat_messages(profile_id,role,content,created_at) VALUES(?,?,?,?)",
                         (pid,role,content,datetime.now().isoformat(timespec="seconds")));con.commit();con.close()

def ksc_chat(p,text):
    if not ai_key():raise RuntimeError("Falta GROQ_API_KEY.")
    msgs=[{"role":"system","content":system_prompt(p)}]+get_chat(p["id"],16)+[{"role":"user","content":text}]
    try:
        r=ai_client(ai_key()).chat.completions.create(
            model=AI_MODEL,messages=msgs,temperature=.55,top_p=.85,
            max_completion_tokens=2200,reasoning_effort="none",stream=False
        )
        ans=(r.choices[0].message.content or "").strip()
        maybe_memory(p["id"],text)
        return ans
    except AuthenticationError as e:raise RuntimeError("La clave de IA no es válida.") from e
    except RateLimitError as e:raise RuntimeError("Límite gratuito temporal alcanzado.") from e
    except APIConnectionError as e:raise RuntimeError("No se pudo conectar con IA KSC.") from e
    except BadRequestError as e:raise RuntimeError(f"No se pudo procesar: {e}") from e

def ai_json(prompt,jpeg=None,max_tokens=1800):
    if not ai_key():raise RuntimeError("Falta GROQ_API_KEY.")
    content=[{"type":"text","text":prompt}]
    if jpeg:content.append({"type":"image_url","image_url":{"url":data_url(jpeg)}})
    r=ai_client(ai_key()).chat.completions.create(
        model=AI_MODEL,messages=[{"role":"user","content":content}],
        response_format={"type":"json_object"},temperature=.2,max_completion_tokens=max_tokens,
        reasoning_effort="none",stream=False
    )
    return json.loads(r.choices[0].message.content or "{}")

VISION_PROMPT="""
Analiza únicamente los alimentos y bebidas visibles.
Devuelve SOLO JSON:
{"summary":"...","foods":[{"name_es":"...","usda_query":"short generic English USDA query",
"estimated_grams":120,"confidence":90,"preparation":"..."}],"limitations":["..."]}
Máximo 6 alimentos. Los gramos son solo estimación visual. Si no hay comida foods=[].
"""
LABEL_PROMPT="""
Lee la tabla nutricional visible. Devuelve SOLO JSON:
{"product_name":"","basis":"100 g|100 ml|por porción|desconocido","serving_size":"",
"kcal":null,"protein_g":null,"carbs_g":null,"fat_g":null,"fiber_g":null,
"sugars_g":null,"sodium_mg":null,"saturated_fat_g":null,"trans_fat_g":null,
"ingredients":"","confidence":0,"warnings":[]}
No inventes valores ilegibles.
"""
FRIDGE_PROMPT="""
Identifica alimentos/ingredientes visibles, incluso si la imagen no es perfecta o hay poca luz.
Si de verdad no distingues nada, igual devuelve tu mejor intento con confianza baja.
Devuelve SOLO JSON:
{"ingredients":["..."],"notes":["..."]}. No inventes ingredientes que claramente no podrían estar ahí.
"""
BARCODE_READ_PROMPT="""
Mira la foto de un código de barras / empaque de producto. Devuelve SOLO JSON:
{"barcode_digits":"solo los números si son legibles o null","product_guess":"nombre visible del producto o null"}
"""

@st.cache_data(ttl=1800,show_spinner=False)
def detect_foods(key,jpeg):
    d=ai_json(VISION_PROMPT,jpeg)
    foods=[]
    for f in d.get("foods",[])[:6]:
        if not isinstance(f,dict):continue
        name=str(f.get("name_es","")).strip();query=str(f.get("usda_query","")).strip()
        if not name or not query:continue
        try:g=int(float(f.get("estimated_grams",100)))
        except:g=100
        try:c=float(f.get("confidence",0))
        except:c=0
        foods.append({"name_es":name[:90],"usda_query":query[:110],"estimated_grams":max(1,min(1500,g)),
                      "confidence":max(0,min(100,c)),"preparation":str(f.get("preparation",""))[:70]})
    return {"summary":str(d.get("summary",""))[:300],"foods":foods,
            "limitations":[str(x)[:180] for x in d.get("limitations",[])[:5]]}

# ============================================================
# USDA
# ============================================================

def num(v):
    try:return float(v) if v is not None else 0.0
    except:return 0.0

def nutrient_value(lst,contains,unit=None):
    for n in lst or []:
        name=str(n.get("nutrientName","")).lower();u=str(n.get("unitName","")).upper()
        if unit and u!=unit.upper():continue
        if any(x.lower() in name for x in contains):return num(n.get("value"))
    return 0.0

def parse_usda(food):
    ns=food.get("foodNutrients",[]) or []
    return {"description":food.get("description",""),"per100":{
        "kcal":nutrient_value(ns,["energy"],"KCAL"),"protein":nutrient_value(ns,["protein"]),
        "carbs":nutrient_value(ns,["carbohydrate"]),"fat":nutrient_value(ns,["total lipid"]),
        "fiber":nutrient_value(ns,["fiber"]),"sugars":nutrient_value(ns,["total sugars","sugars"]),
        "sodium_mg":nutrient_value(ns,["sodium"]),"sat_fat":nutrient_value(ns,["total saturated"])
    }}

@st.cache_data(ttl=86400,show_spinner=False)
def usda_search(query,key):
    try:
        r=requests.post(f"{USDA_BASE}/foods/search",params={"api_key":key},
                        json={"query":query,"pageSize":5,"dataType":["Foundation","SR Legacy","Survey (FNDDS)"]},timeout=15)
        r.raise_for_status();foods=r.json().get("foods",[])
        return parse_usda(foods[0]) if foods else None
    except:return None

def portion(per100,g):
    return {k:num(v)*g/100 for k,v in per100.items()}

NKEYS=["kcal","protein","carbs","fat","fiber","sugars","sodium_mg","sat_fat"]

def total_nutrition(items):
    out={k:0.0 for k in NKEYS}
    for item in items:
        for k in NKEYS:out[k]+=num(item["portion"].get(k))
    return out

def show_metrics(v):
    a,b,c,d=st.columns(4)
    a.metric("🔥 Energía",f"{v['kcal']:.0f} kcal");b.metric("💪 Proteína",f"{v['protein']:.1f} g")
    c.metric("🍚 Carbohidratos",f"{v['carbs']:.1f} g");d.metric("🥑 Grasas",f"{v['fat']:.1f} g")
    e,f,g,h=st.columns(4)
    e.metric("🌾 Fibra",f"{v['fiber']:.1f} g");f.metric("🍬 Azúcares",f"{v['sugars']:.1f} g")
    g.metric("🧂 Sodio",f"{v['sodium_mg']:.0f} mg");h.metric("🧈 Saturadas",f"{v['sat_fat']:.1f} g")

def enrich(result,prefix):
    calc=[]
    for i,f in enumerate(result.get("foods",[])):
        with st.container(border=True):
            a,b,c=st.columns([1.1,.55,1.1])
            a.markdown(f"### {f['name_es']}");a.caption(f"{f['preparation']} · confianza {f['confidence']:.0f}%")
            g=b.number_input("Cantidad (g)",1,1500,int(f["estimated_grams"]),5,key=f"{prefix}_g_{i}")
            q=c.text_input("Búsqueda nutricional",f["usda_query"],key=f"{prefix}_q_{i}")
            ref=usda_search(q,usda_key())
            if ref:
                p=portion(ref["per100"],g)
                calc.append({"name":f["name_es"],"grams":g,"confidence":f["confidence"],"portion":p})
                st.caption("Referencia: "+ref["description"]);show_metrics(p)
            else:st.info("No encontré referencia. Puedes editar la búsqueda.")
    return calc

# ============================================================
# DIARIO
# ============================================================

def add_meal(pid,meal_type,title,calc,total,image_path,note):
    con=db()
    con.execute("""
      INSERT INTO meal_diary(profile_id,meal_date,meal_time,meal_type,title,foods_json,
       kcal,protein,carbs,fat,fiber,sugars,sodium_mg,sat_fat,image_path,note)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(pid,str(date.today()),datetime.now().strftime("%H:%M"),meal_type,title,
         json.dumps(calc,ensure_ascii=False),total["kcal"],total["protein"],total["carbs"],
         total["fat"],total["fiber"],total["sugars"],total["sodium_mg"],total["sat_fat"],image_path,note))
    con.commit();con.close();add_points(pid,15,"Comida registrada")

def meals_between(pid,start,end):
    con=db();rows=con.execute("SELECT * FROM meal_diary WHERE profile_id=? AND meal_date BETWEEN ? AND ? ORDER BY meal_date,meal_time",
                              (pid,str(start),str(end))).fetchall();con.close()
    return [dict(r) for r in rows]

def day_totals(pid,d=None):
    d=d or date.today();meals=meals_between(pid,d,d);tot={k:0.0 for k in NKEYS}
    for m in meals:
        tot["kcal"]+=num(m["kcal"]);tot["protein"]+=num(m["protein"]);tot["carbs"]+=num(m["carbs"])
        tot["fat"]+=num(m["fat"]);tot["fiber"]+=num(m["fiber"]);tot["sugars"]+=num(m["sugars"])
        tot["sodium_mg"]+=num(m["sodium_mg"]);tot["sat_fat"]+=num(m["sat_fat"])
    return tot,meals

# ============================================================
# FAVORITOS / PLANES
# ============================================================

def save_favorite(pid,title,recipe,category):
    con=db();con.execute("INSERT INTO favorites(profile_id,title,recipe,category,created_at) VALUES(?,?,?,?,?)",
                         (pid,title,recipe,category,datetime.now().isoformat(timespec="seconds")));con.commit();con.close()
    add_points(pid,5,"Receta favorita")

def favorites(pid):
    con=db();rows=con.execute("SELECT * FROM favorites WHERE profile_id=? ORDER BY id DESC",(pid,)).fetchall();con.close()
    return [dict(r) for r in rows]

def save_rating(pid,fid,title,rating,comment):
    con=db();con.execute("""INSERT INTO recipe_ratings(profile_id,favorite_id,recipe_title,rating,comment,created_at)
                            VALUES(?,?,?,?,?,?)""",(pid,fid,title,rating,comment,datetime.now().isoformat(timespec="seconds")))
    con.commit();con.close();add_points(pid,3,"Receta calificada")

PLAN_PROMPT="""
Crea un plan semanal de alimentación. Devuelve SOLO JSON:
{"days":[{"day":"Lunes","breakfast":"...","lunch":"...","dinner":"...","snack":"..."}],
"shopping_list":["ingrediente y cantidad aproximada"],"notes":["..."]}
Debe incluir 7 días. Respeta alergias y gustos. En menores de 18 no hagas una dieta para bajar/subir peso.
"""

def save_plan(pid,week_start,plan):
    con=db();con.execute("INSERT INTO weekly_plans(profile_id,week_start,plan_json,created_at) VALUES(?,?,?,?)",
                         (pid,str(week_start),json.dumps(plan,ensure_ascii=False),datetime.now().isoformat(timespec="seconds")))
    con.commit();con.close();add_points(pid,20,"Plan semanal")

def latest_plan(pid):
    con=db();r=con.execute("SELECT * FROM weekly_plans WHERE profile_id=? ORDER BY id DESC LIMIT 1",(pid,)).fetchone();con.close()
    if not r:return None
    d=dict(r)
    try:d["plan"]=json.loads(d["plan_json"])
    except:d["plan"]={}
    return d

# ============================================================
# PESO / MEDIDAS / EUREKA
# ============================================================

def add_weight(pid,d,w,waist,note):
    con=db();con.execute("INSERT INTO weight_logs(profile_id,log_date,weight_kg,waist_cm,note) VALUES(?,?,?,?,?)",
                         (pid,str(d),w,waist,note));con.execute("UPDATE profiles SET weight_kg=? WHERE id=?",(w,pid))
    con.commit();con.close();add_points(pid,5,"Peso registrado")

def weights(pid):
    con=db();rows=con.execute("SELECT * FROM weight_logs WHERE profile_id=? ORDER BY log_date,id",(pid,)).fetchall();con.close()
    return [dict(r) for r in rows]

def add_measure(pid,d,waist,hip,chest,arm,thigh,note):
    con=db();con.execute("""INSERT INTO measurements(profile_id,log_date,waist_cm,hip_cm,chest_cm,arm_cm,thigh_cm,note)
                            VALUES(?,?,?,?,?,?,?,?)""",(pid,str(d),waist,hip,chest,arm,thigh,note))
    con.commit();con.close();add_points(pid,5,"Medidas registradas")

def measures(pid):
    con=db();rows=con.execute("SELECT * FROM measurements WHERE profile_id=? ORDER BY log_date,id",(pid,)).fetchall();con.close()
    return [dict(r) for r in rows]

def save_plate_test(pid,pred,actual,correct,conf,kcal):
    con=db();con.execute("""INSERT INTO plate_tests(profile_id,created_at,predicted_foods,actual_foods,correct,avg_conf,estimated_kcal,app_version)
                            VALUES(?,?,?,?,?,?,?,?)""",
                         (pid,datetime.now().isoformat(timespec="seconds"),pred,actual,1 if correct else 0,conf,kcal,APP_VERSION))
    con.commit();con.close()

# ============================================================
# CÓDIGO DE BARRAS (Open Food Facts, gratis, sin clave + caché local)
# ============================================================

def _load_barcode_cache():
    if BARCODE_CACHE_PATH.exists():
        try:
            return json.loads(BARCODE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_barcode_cache(cache):
    try:
        BARCODE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def lookup_barcode(code):
    code = re.sub(r"\D", "", code or "")
    if not code:
        return None, "Código vacío."
    cache = _load_barcode_cache()
    if code in cache:
        return cache[code], None
    try:
        r = requests.get(f"{OFF_BASE}/{code}.json", timeout=15)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != 1:
            return None, "No encontré ese código en la base de datos pública."
        prod = j.get("product", {})
        n = prod.get("nutriments", {})
        info = {
            "code": code,
            "name": prod.get("product_name") or prod.get("product_name_es") or "Producto sin nombre",
            "brand": prod.get("brands", ""),
            "image": prod.get("image_front_url", ""),
            "ingredients": prod.get("ingredients_text_es") or prod.get("ingredients_text") or "",
            "kcal_100g": n.get("energy-kcal_100g"),
            "protein_100g": n.get("proteins_100g"),
            "carbs_100g": n.get("carbohydrates_100g"),
            "fat_100g": n.get("fat_100g"),
            "sugars_100g": n.get("sugars_100g"),
            "fiber_100g": n.get("fiber_100g"),
            "sodium_100g": n.get("sodium_100g"),
            "nutriscore": (prod.get("nutriscore_grade") or "").upper(),
            "allergens": prod.get("allergens", ""),
            "nova_group": prod.get("nova_group"),
        }
        cache[code] = info
        _save_barcode_cache(cache)
        return info, None
    except Exception as e:
        return None, f"No pude consultar la base pública ({e}). Intenta con el número manual."

# ============================================================
# PUSH-UP CHALLENGES
# ============================================================

def create_challenge(challenger,opponent):
    con=db();cur=con.execute("""INSERT INTO pushup_challenges(challenger_id,opponent_id,created_at,status,duration_seconds)
                                VALUES(?,?,?,?,60)""",(challenger,opponent,datetime.now().isoformat(timespec="seconds"),"pending"))
    cid=cur.lastrowid;con.commit();con.close();add_points(challenger,5,"Reto push-up enviado");return cid

def challenges(pid):
    con=db();rows=con.execute("""
      SELECT c.*,p1.name challenger_name,p2.name opponent_name
      FROM pushup_challenges c JOIN profiles p1 ON p1.id=c.challenger_id JOIN profiles p2 ON p2.id=c.opponent_id
      WHERE c.challenger_id=? OR c.opponent_id=? ORDER BY c.id DESC
    """,(pid,pid)).fetchall();con.close()
    return [dict(r) for r in rows]

def accept_challenge(cid,pid):
    con=db();con.execute("UPDATE pushup_challenges SET status='active' WHERE id=?",(cid,));con.commit();con.close()
    add_points(pid,5,"Reto push-up aceptado")

def attempts(cid):
    con=db();rows=con.execute("""
      SELECT a.*,p.name FROM pushup_attempts a JOIN profiles p ON p.id=a.profile_id
      WHERE a.challenge_id=? ORDER BY a.id
    """,(cid,)).fetchall();con.close()
    return [dict(r) for r in rows]

def save_attempt(cid,pid,reps,duration,video_path=""):
    con=db();con.execute("""INSERT OR REPLACE INTO pushup_attempts(challenge_id,profile_id,reps,duration_seconds,video_path,created_at)
                            VALUES(?,?,?,?,?,?)""",
                         (cid,pid,int(reps),float(duration),video_path,datetime.now().isoformat(timespec="seconds")))
    count=con.execute("SELECT COUNT(*) c FROM pushup_attempts WHERE challenge_id=?",(cid,)).fetchone()["c"]
    if count>=2:con.execute("UPDATE pushup_challenges SET status='completed' WHERE id=?",(cid,))
    con.commit();con.close();add_points(pid,max(10,int(reps)),f"Push-ups: {reps}")

def pushup_ranking():
    con=db();df=pd.read_sql_query("""
      SELECT p.name,COALESCE(MAX(a.reps),0) mejor_marca,COUNT(a.id) intentos,COALESCE(SUM(a.reps),0) total
      FROM profiles p LEFT JOIN pushup_attempts a ON a.profile_id=p.id
      GROUP BY p.id,p.name ORDER BY mejor_marca DESC,total DESC
    """,con);con.close();return df

def angle(a,b,c):
    ba=(a[0]-b[0],a[1]-b[1]);bc=(c[0]-b[0],c[1]-b[1])
    den=math.hypot(*ba)*math.hypot(*bc)
    if den==0:return 180
    cosv=max(-1,min(1,(ba[0]*bc[0]+ba[1]*bc[1])/den))
    return math.degrees(math.acos(cosv))

def ensure_pose_model():
    if POSE_MODEL_PATH.exists() and POSE_MODEL_PATH.stat().st_size>100000:return True,""
    try:
        r=requests.get(POSE_MODEL_URL,timeout=60);r.raise_for_status();POSE_MODEL_PATH.write_bytes(r.content);return True,""
    except Exception as e:return False,str(e)

POSE_SKELETON_EDGES = [
    (11,13),(13,15),(12,14),(14,16),(11,12),(23,24),
    (11,23),(12,24),(23,25),(25,27),(24,26),(26,28),
]

def render_pushup_camera(cid,pid):
    try:
        import av, cv2, mediapipe as mp
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        from aiortc.contrib.media import MediaRecorder
    except Exception:
        st.error("Para activar la cámara Push-Up con esqueleto en vivo instala las librerías opcionales.")
        st.code("pip install streamlit-webrtc mediapipe av opencv-python-headless",language="powershell")
        st.info("Mientras tanto puedes registrar un intento manual abajo para probar el sistema de retos.")
        reps=st.number_input("Repeticiones verificadas manualmente",0,300,0,1,key=f"manual_reps_{cid}_{pid}")
        if st.button("Guardar intento manual",key=f"manual_save_{cid}_{pid}"):
            save_attempt(cid,pid,reps,60,"");st.rerun()
        return

    ok,err=ensure_pose_model()
    if not ok:
        st.error("No pude descargar el modelo de pose: "+err);return

    key=f"pushup_{cid}_{pid}"
    if st.session_state.get("push_key")!=key:
        st.session_state["push_key"]=key
        st.session_state["push_state"]={"lock":threading.Lock(),"reps":0,"stage":"up","start":None,"elapsed":0.0,"last_ts":0,"landmarker":None}
        st.session_state["push_video"]=str(PUSHUP_DIR/f"{key}_{int(time.time())}.mp4")
    shared=st.session_state["push_state"];video_path=st.session_state["push_video"]

    def get_landmarker():
        with shared["lock"]:
            if shared["landmarker"] is not None:return shared["landmarker"]
            Options=mp.tasks.vision.PoseLandmarkerOptions
            shared["landmarker"]=mp.tasks.vision.PoseLandmarker.create_from_options(
                Options(base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,
                        min_pose_detection_confidence=.55,min_pose_presence_confidence=.55,min_tracking_confidence=.55)
            )
            return shared["landmarker"]

    def draw_skeleton_and_ring(img, lm, elbow_angle, reps, elapsed, stage):
        hpx, wpx = img.shape[0], img.shape[1]
        pts = [(int(p.x*wpx), int(p.y*hpx)) for p in lm]
        for a_idx, b_idx in POSE_SKELETON_EDGES:
            cv2.line(img, pts[a_idx], pts[b_idx], (86, 240, 159), 3, cv2.LINE_AA)
        for idx in set([i for e in POSE_SKELETON_EDGES for i in e]):
            cv2.circle(img, pts[idx], 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[idx], 6, (34, 201, 138), 2, cv2.LINE_AA)
        # anillo circular de progreso (verde) segun el angulo del codo dentro del rango de una repeticion
        cx, cy, r = 70, 70, 46
        color_ring = (159, 240, 86) if stage == "down" else (255, 209, 102)
        prog = max(0.0, min(1.0, (160 - elbow_angle) / (160 - 85))) if stage != "up" else 0.0
        cv2.circle(img, (cx, cy), r, (40, 40, 40), 8, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, int(360*prog), color_ring, 8, cv2.LINE_AA)
        cv2.putText(img, str(reps), (cx-18, cy+12), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)
        cv2.putText(img, f"TIME {max(0,60-elapsed):.0f}s", (20, hpx-20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)

    def callback(frame):
        img=frame.to_ndarray(format="bgr24");rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        with shared["lock"]:
            if shared["start"] is None:shared["start"]=time.monotonic()
            shared["elapsed"]=time.monotonic()-shared["start"]
            ts=max(shared["last_ts"]+1,int(time.monotonic()*1000));shared["last_ts"]=ts
        try:
            lmres=get_landmarker().detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ts)
            if lmres.pose_landmarks:
                lm=lmres.pose_landmarks[0]
                use_left=min(lm[11].visibility,lm[13].visibility,lm[15].visibility)>=min(lm[12].visibility,lm[14].visibility,lm[16].visibility)
                s,e,w,h,a=(lm[11],lm[13],lm[15],lm[23],lm[27]) if use_left else (lm[12],lm[14],lm[16],lm[24],lm[28])
                elbow=angle((s.x,s.y),(e.x,e.y),(w.x,w.y));body=angle((s.x,s.y),(h.x,h.y),(a.x,a.y))
                with shared["lock"]:
                    if shared["elapsed"]<60 and body>145:
                        if elbow<95 and shared["stage"]=="up":shared["stage"]="down"
                        elif elbow>155 and shared["stage"]=="down":shared["reps"]+=1;shared["stage"]="up"
                    reps=shared["reps"];elapsed=shared["elapsed"];stage=shared["stage"]
                draw_skeleton_and_ring(img, lm, elbow, reps, elapsed, stage)
        except Exception:pass
        return av.VideoFrame.from_ndarray(img,format="bgr24")

    def recorder():
        return MediaRecorder(video_path)

    st.info("Coloca la cámara de lado y muestra hombros, codos, caderas y tobillos. El anillo verde se llena en cada repetición. El intento dura 60 s.")
    webrtc_streamer(
        key=key,mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video":True,"audio":False},
        video_frame_callback=callback,in_recorder_factory=recorder,async_processing=True
    )
    with shared["lock"]:reps=shared["reps"];elapsed=shared["elapsed"]
    st.metric("Push-ups detectados",reps)
    st.metric("Tiempo",f"{min(elapsed,60):.1f}/60 s")
    st.warning("Es una estimación por visión artificial, no un árbitro perfecto. Detente si sientes dolor o mareo.")
    if st.button("Guardar intento",type="primary",key=f"save_push_{cid}_{pid}"):
        if elapsed<45:st.error("Haz una sesión cercana al minuto antes de guardarla.")
        else:
            save_attempt(cid,pid,reps,min(elapsed,60),video_path if Path(video_path).exists() else "")
            st.rerun()

# ============================================================
# VOZ REALISTA (gratis): edge-tts para hablar + Whisper (Groq) para
# escuchar. Todo automático: grabas -> se transcribe -> se envía sola
# -> IA KSC responde -> se reproduce con voz natural, sin copiar/pegar.
# ============================================================

VOICES = {
    "🇵🇪 Camila (Perú, mujer)": "es-PE-CamilaNeural",
    "🇵🇪 Alex (Perú, hombre)": "es-PE-AlexNeural",
    "🇲🇽 Dalia (México, mujer)": "es-MX-DaliaNeural",
    "🇲🇽 Jorge (México, hombre)": "es-MX-JorgeNeural",
    "🇪🇸 Elvira (España, mujer)": "es-ES-ElviraNeural",
    "🇪🇸 Álvaro (España, hombre)": "es-ES-AlvaroNeural",
    "🇦🇷 Elena (Argentina, mujer)": "es-AR-ElenaNeural",
    "🇦🇷 Tomás (Argentina, hombre)": "es-AR-TomasNeural",
}
DEFAULT_VOICE_LABEL = "🇵🇪 Camila (Perú, mujer)"

@st.cache_data(ttl=3600, show_spinner=False, max_entries=60)
def synth_speech(text, voice_id):
    """Genera audio MP3 con una voz neuronal realista y gratuita (Microsoft Edge TTS,
    sin API key). Cacheado para no regenerar el mismo texto dos veces."""
    if not EDGE_TTS_OK:
        return None
    text = (text or "").strip()
    if not text:
        return None
    text = text[:1800]  # evita audios eternos

    async def _run():
        communicate = edge_tts.Communicate(text, voice_id, rate="+4%")
        audio = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return audio

    try:
        return asyncio.run(_run())
    except Exception:
        return None

def current_voice_id():
    label = st.session_state.get("ksc_voice_label", DEFAULT_VOICE_LABEL)
    return VOICES.get(label, VOICES[DEFAULT_VOICE_LABEL])

def voice_picker(location=st.sidebar):
    label = location.selectbox(
        "🔊 Voz de IA KSC",
        list(VOICES.keys()),
        index=list(VOICES.keys()).index(st.session_state.get("ksc_voice_label", DEFAULT_VOICE_LABEL)),
        key="ksc_voice_label",
    )
    return VOICES[label]

def speak_reply(text, key="tts", autoplay=True):
    """Reproduce 'text' con voz neuronal realista. autoplay=True suena solo, sin botón."""
    if not EDGE_TTS_OK:
        st.caption("🔇 Voz no disponible: falta instalar 'edge-tts' (agrégalo a requirements.txt).")
        return
    audio = synth_speech(text, current_voice_id())
    if audio:
        st.audio(audio, format="audio/mp3", autoplay=autoplay)
    else:
        st.caption("No pude generar el audio ahora mismo (revisa tu conexión).")

def transcribe_audio(audio_bytes):
    """Transcribe voz a texto con Whisper vía Groq (gratis dentro de tu cuota de GROQ_API_KEY)."""
    if not ai_key():
        return None, "Falta GROQ_API_KEY."
    try:
        client = ai_client(ai_key())
        result = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3-turbo",
            language="es",
            response_format="text",
        )
        text = str(result).strip()
        return (text or None), None
    except Exception as e:
        return None, f"No pude transcribir el audio: {e}"

# ============================================================
# UI
# ============================================================

def hero(p=None):
    extra=""
    if p:
        pts,lvl,_=level_info(p["id"]);r=streak(p["id"])
        extra=(
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;position:relative;z-index:1">'
            f'<span class="chip">👋 Hola, {p["name"]}</span>'
            f'<span class="chip">{lvl[2]} {lvl[1]}</span>'
            f'<span class="chip">🏆 {pts} pts</span>'
            f'<span class="chip">🔥 {r} día{"s" if r!=1 else ""} de racha</span>'
            '</div>'
        )
    st.markdown(
        '<div class="hero">'
        '<span class="badge">● EUREKA 2026 · NUTRIVISION</span>'
        '<div class="hero-title">IA <span class="green">KSC</span></div>'
        '<div class="hero-sub">Tu diario nutricional completo: comidas, código de barras, chat por voz, retos físicos, comunidad y progreso — todo en un solo lugar.</div>'
        f'{extra}'
        '</div>',
        unsafe_allow_html=True
    )

def section(k,t,s):
    st.markdown(f'<div class="kicker">{k.upper()}</div><div style="font-size:2.05rem;font-weight:950;color:white">{t}</div><div class="note">{s}</div><br>',unsafe_allow_html=True)

def profile_photo_b64(path):
    try:
        raw=Path(path).read_bytes()
        return "data:image/jpeg;base64,"+base64.b64encode(raw).decode()
    except Exception:
        return None

def profile_selector():
    ps=list_profiles()
    if not ps:return None
    mapping={f"{p['name']} · {p['age']} años":p["id"] for p in ps}
    names=list(mapping)
    cur=st.session_state.get("pid")
    idx=next((i for i,n in enumerate(names) if mapping[n]==cur),0)
    label=st.sidebar.selectbox("Perfil activo",names,index=idx)
    pid=mapping[label];st.session_state["pid"]=pid;p=get_profile(pid)
    key=f"unlocked_{pid}"
    if not p.get("pin_hash"):st.session_state[key]=True
    if not st.session_state.get(key):
        st.sidebar.markdown(f'<div class="side-profile"><div class="side-avatar-fallback">🔒</div><b>{p["name"]}</b><div class="note" style="margin-top:2px">Perfil bloqueado</div></div>',unsafe_allow_html=True)
        pin=st.sidebar.text_input("PIN",type="password",max_chars=4,key=f"pin_{pid}")
        if st.sidebar.button("🔓 Desbloquear",use_container_width=True,type="primary"):
            if verify_pin(pin,p["pin_hash"]):st.session_state[key]=True;st.rerun()
            else:st.sidebar.error("PIN incorrecto.")
        return None
    photo=p.get("photo_path")
    b64=profile_photo_b64(photo) if photo and Path(photo).exists() else None
    avatar_html=f'<img src="{b64}" class="side-avatar">' if b64 else f'<div class="side-avatar-fallback">{p["name"][:1].upper()}</div>'
    r=streak(p["id"])
    st.sidebar.markdown(f"""
    <div class="side-profile">
      {avatar_html}
      <b>{p['name']}</b>
      <div style="margin-top:6px"><span class="chip">{p['goal']}</span></div>
      <div class="note" style="margin-top:4px">🔥 {r} día{'s' if r!=1 else ''} de racha</div>
    </div>""",unsafe_allow_html=True)
    if st.sidebar.button("🔒 Bloquear",use_container_width=True):st.session_state[key]=False;st.rerun()
    return p

def need_profile(p):
    if not p:
        st.markdown('<div class="lock"><h3>🔒 Desbloquea un perfil</h3><div class="note">Selecciona o crea un perfil en la barra lateral para continuar.</div></div>',unsafe_allow_html=True)
        st.stop()

# ============================================================
# SIDEBAR — MENÚ REORDENADO Y RENOMBRADO
# ============================================================

MENU = [
    "🏠 Inicio",
    "👤 Mi perfil",
    "🌐 Comunidad",
    "📷 Diario de comidas",
    "🔎 Escáner de código de barras",
    "🧾 Escáner de etiqueta",
    "⚖️ Comparar platos",
    "💬 Chat por voz con IA KSC",
    "🍳 Cocina inteligente",
    "📅 Plan semanal",
    "💧 Agua & hábitos",
    "🏆 Recompensas KSC",
    "💪 Arena de Push-Ups",
    "📈 Mi progreso",
    "🎓 Academia KSC (quiz)",
    "🧪 Eureka Lab",
    "⚙️ Configuración",
]

with st.sidebar:
    st.markdown('<div class="brand-row"><div class="brand-icon">🥗</div><div style="font-size:1.3rem;font-weight:950;color:white;letter-spacing:-.03em">IA KSC</div></div>',unsafe_allow_html=True)
    st.caption(f"NutriVision · V{APP_VERSION}")
    st.markdown("---")
    page=st.radio("Menú",MENU,label_visibility="collapsed")
    st.markdown("---")
    profile=profile_selector()
    st.markdown("---")
    voice_picker()

hero(profile)
sync_community()

# ============================================================
# INICIO
# ============================================================

if page=="🏠 Inicio":
    section("HOY","Panel principal","Calorías, agua, puntos, racha y qué comer ahora — todo de un vistazo.")
    if not profile:
        st.info("Crea o desbloquea un perfil en la barra lateral para comenzar.")
        st.markdown("### ⚡ Empieza rápido")
        c1,c2,c3=st.columns(3)
        c1.markdown('<div class="mini"><div class="kicker">PASO 1</div><div class="big">👤 Crea tu perfil</div><div class="note">Ve a "Mi perfil" y regístrate con foto, gustos y PIN.</div></div>',unsafe_allow_html=True)
        c2.markdown('<div class="mini"><div class="kicker">PASO 2</div><div class="big">📷 Registra tu comida</div><div class="note">Sube una foto y deja que IA KSC calcule los nutrientes.</div></div>',unsafe_allow_html=True)
        c3.markdown('<div class="mini"><div class="kicker">PASO 3</div><div class="big">🏆 Gana puntos</div><div class="note">Cada acción saludable suma puntos y sube tu nivel.</div></div>',unsafe_allow_html=True)
    else:
        total,meals=day_totals(profile["id"]);water=water_today(profile["id"]);goal=int(profile.get("water_goal_ml") or 2000)
        pts,lvl,nxt=level_info(profile["id"]);racha=streak(profile["id"])
        if water==0:st.markdown('<div class="water-alert"><b>💧 Falta registrar agua hoy.</b> Ve a "Agua & hábitos" para anotar tus vasos.</div>',unsafe_allow_html=True)

        wpct=min(100,round(water/goal*100)) if goal else 0
        st.markdown(textwrap.dedent(f"""
        <div class="stat-grid">
          <div class="stat-card accent-orange" style="animation-delay:.02s">
            <span class="icon">🔥</span>
            <div class="label">Calorías hoy</div>
            <div class="value">{total['kcal']:.0f}</div>
            <div class="sub">kcal registradas</div>
          </div>
          <div class="stat-card accent-blue" style="animation-delay:.08s">
            <span class="icon">💧</span>
            <div class="label">Hidratación</div>
            <div class="value">{water} <span style="font-size:.95rem;color:var(--muted);font-weight:700">/ {goal} ml</span></div>
            <div class="sub">{wpct}% de tu meta</div>
          </div>
          <div class="stat-card accent-purple" style="animation-delay:.14s">
            <span class="icon">🏆</span>
            <div class="label">Puntos KSC</div>
            <div class="value">{pts}</div>
            <div class="sub">Nivel {lvl[2]} {lvl[1]}</div>
          </div>
          <div class="stat-card accent-green" style="animation-delay:.2s">
            <div class="streak-card" style="border:none;background:none;padding:0;gap:10px">
              <span class="streak-flame">🔥</span>
              <div>
                <div class="streak-days">{racha}</div>
                <div class="streak-label">día{'s' if racha!=1 else ''} de racha</div>
              </div>
            </div>
          </div>
        </div>
        """),unsafe_allow_html=True)
        show_metrics(total)
        st.markdown("### 🤖 ¿Qué puedo comer ahora?")
        hour=datetime.now().hour
        moment="desayuno" if hour<10 else "media mañana" if hour<12 else "almuerzo" if hour<16 else "merienda" if hour<19 else "cena"
        if st.button(f"Recomiéndame {moment}",type="primary",use_container_width=True):
            prompt=f"Es {moment}. Hoy llevo {total['kcal']:.0f} kcal, {total['protein']:.1f} g proteína, {total['fiber']:.1f} g fibra y {water} ml de agua. Dame 3 opciones para mi perfil y mis gustos."
            try:
                ans=ksc_chat(profile,prompt)
                st.markdown(ans)
                speak_reply(ans, key="home_tts", autoplay=False)
            except RuntimeError as e:st.error(str(e))
        if meals:st.dataframe(pd.DataFrame(meals)[["meal_time","meal_type","title","kcal","protein","fiber"]],hide_index=True,use_container_width=True)

# ============================================================
# MI PERFIL
# ============================================================

elif page=="👤 Mi perfil":
    section("USUARIOS","Mi perfil","Nombre, foto, talla, peso, gustos, alergias, objetivo y agua. Se guarda para siempre en la base local.")
    t1,t2=st.tabs(["➕ Crear perfil nuevo","✏️ Editar mi perfil"])
    with t1:
        with st.form("newp"):
            c1,c2=st.columns(2)
            with c1:
                name=st.text_input("Nombre");age=st.number_input("Edad",10,100,16,1)
                sex=st.selectbox("Variable fisiológica",["Prefiero no indicar","Masculino","Femenino"])
                height=st.number_input("Talla cm",120.,230.,165.,.5);weight=st.number_input("Peso kg",30.,250.,60.,.1)
                activity=st.selectbox("Actividad",list(ACTIVITY_FACTORS),index=1)
                goal=st.selectbox("Objetivo",["Mantener","Ganar masa muscular","Ganar peso","Perder peso"])
                water_goal=st.number_input("Meta de agua ml/día",500,5000,2000,100)
            with c2:
                photo=st.file_uploader("Foto",type=["jpg","jpeg","png"],key="np_photo")
                fav=st.text_area("Comidas favoritas");fr=st.text_area("Frutas favoritas");veg=st.text_area("Verduras favoritas")
                avoid=st.text_area("No me gusta / evito");allerg=st.text_area("Alergias/restricciones")
                special=st.selectbox("Estado especial",["Ninguno","Embarazo","Lactancia"])
                pin=st.text_input("PIN 4 dígitos",type="password",max_chars=4);pin2=st.text_input("Repite PIN",type="password",max_chars=4)
            save=st.form_submit_button("Crear perfil",type="primary",use_container_width=True)
        if save:
            if not name.strip():st.error("Escribe nombre.")
            elif not re.fullmatch(r"\d{4}",pin or "") or pin!=pin2:st.error("PIN inválido o no coincide.")
            else:
                pp=save_jpeg(photo.getvalue(),PROFILE_DIR,"profile",700) if photo else ""
                pid=create_profile({"name":name.strip(),"age":int(age),"sex_energy":sex,"height_cm":height,"weight_kg":weight,
                    "activity":activity,"goal":goal,"favorite_foods":fav,"favorite_fruits":fr,"favorite_vegetables":veg,
                    "avoid_foods":avoid,"allergies":allerg,"special_state":special,"photo_path":pp,
                    "pin_hash":hash_pin(pin),"water_goal_ml":int(water_goal)})
                add_points(pid,20,"Perfil creado");st.session_state["pid"]=pid;st.session_state[f"unlocked_{pid}"]=True
                st.success("Perfil creado y guardado permanentemente. 🎉");st.rerun()

    with t2:
        if not profile:st.info("Desbloquea un perfil.")
        else:
            b64=profile_photo_b64(profile["photo_path"]) if profile.get("photo_path") and Path(profile["photo_path"]).exists() else None
            if b64:st.markdown(f'<img src="{b64}" class="avatar-photo" style="width:120px;height:120px">',unsafe_allow_html=True)
            else:st.markdown(f'<div class="avatar-ring" style="width:120px;height:120px;font-size:2.2rem">{profile["name"][:1].upper()}</div>',unsafe_allow_html=True)
            with st.form("editp"):
                c1,c2=st.columns(2)
                with c1:
                    name=st.text_input("Nombre",profile["name"]);age=st.number_input("Edad",10,100,int(profile["age"]),1)
                    opts=["Prefiero no indicar","Masculino","Femenino"];sex=st.selectbox("Variable fisiológica",opts,index=opts.index(profile["sex_energy"]) if profile["sex_energy"] in opts else 0)
                    height=st.number_input("Talla",120.,230.,float(profile["height_cm"]),.5);weight=st.number_input("Peso",30.,250.,float(profile["weight_kg"]),.1)
                    acts=list(ACTIVITY_FACTORS);activity=st.selectbox("Actividad",acts,index=acts.index(profile["activity"]))
                    goals=["Mantener","Ganar masa muscular","Ganar peso","Perder peso"];goal=st.selectbox("Objetivo",goals,index=goals.index(profile["goal"]))
                    water_goal=st.number_input("Meta agua",500,5000,int(profile.get("water_goal_ml") or 2000),100)
                with c2:
                    photo=st.file_uploader("Cambiar foto",type=["jpg","jpeg","png"],key="ep_photo")
                    fav=st.text_area("Comidas favoritas",profile.get("favorite_foods",""));fr=st.text_area("Frutas",profile.get("favorite_fruits",""))
                    veg=st.text_area("Verduras",profile.get("favorite_vegetables",""));avoid=st.text_area("Evito",profile.get("avoid_foods",""))
                    allerg=st.text_area("Alergias",profile.get("allergies",""));sp=["Ninguno","Embarazo","Lactancia"]
                    special=st.selectbox("Estado especial",sp,index=sp.index(profile.get("special_state","Ninguno")))
                save=st.form_submit_button("Guardar cambios",type="primary",use_container_width=True)
            if save:
                pp=profile.get("photo_path","")
                if photo:pp=save_jpeg(photo.getvalue(),PROFILE_DIR,f"profile_{profile['id']}",700)
                update_profile(profile["id"],{"name":name,"age":int(age),"sex_energy":sex,"height_cm":height,"weight_kg":weight,
                    "activity":activity,"goal":goal,"favorite_foods":fav,"favorite_fruits":fr,"favorite_vegetables":veg,
                    "avoid_foods":avoid,"allergies":allerg,"special_state":special,"photo_path":pp,"water_goal_ml":int(water_goal)})
                st.success("Guardado.");st.rerun()
            st.markdown("---")
            with st.expander("🗑️ Eliminar este perfil (irreversible)"):
                st.warning("Esto borra el perfil y todo su historial de forma permanente.")
                confirm=st.text_input("Escribe ELIMINAR para confirmar",key="del_confirm")
                if st.button("Eliminar perfil definitivamente") and confirm=="ELIMINAR":
                    delete_profile(profile["id"]);st.session_state.pop("pid",None);st.rerun()

# ============================================================
# COMUNIDAD (ver otros perfiles, retar, mensajes locales)
# ============================================================

elif page=="🌐 Comunidad":
    need_profile(profile)
    section("COMUNIDAD","Otros usuarios de IA KSC","Perfiles creados en esta app (sin contraseñas). Puedes retarlos o escribirles.")
    st.caption("Esto muestra los perfiles guardados en este dispositivo/servidor. Para una red entre distintos dispositivos se necesitaría un servidor compartido; aquí todos comparten la misma base de datos local.")
    community=[c for c in read_community() if c["id"]!=profile["id"]]
    if not community:
        st.info("Todavía no hay otros perfiles. Invita a alguien a crear el suyo.")
    else:
        cols=st.columns(3)
        for i,c in enumerate(community):
            with cols[i%3]:
                b64=profile_photo_b64(c["photo_path"]) if c.get("photo_path") and Path(c["photo_path"]).exists() else None
                img_html=f'<img src="{b64}" style="width:64px;height:64px;border-radius:99px;object-fit:cover;margin:0 auto 8px;display:block;border:2px solid rgba(86,240,159,.35)">' if b64 else f'<div class="avatar-ring" style="width:64px;height:64px;font-size:1.3rem">{c["name"][:1].upper()}</div>'
                st.markdown(f'<div class="community-card">{img_html}<b>{c["name"]}</b><div class="note">{c["goal"]}</div></div>',unsafe_allow_html=True)
                b1,b2=st.columns(2)
                if b1.button("⚔️ Retar",key=f"chal_{c['id']}",use_container_width=True):
                    create_challenge(profile["id"],c["id"]);st.success("Reto de push-ups enviado")
                if b2.button("💬 Ver chat",key=f"openmsg_{c['id']}",use_container_width=True):
                    st.session_state["dm_target"]=c["id"]

    target=st.session_state.get("dm_target")
    if target:
        tp=get_profile(target)
        if tp:
            st.markdown(f"### 💬 Mensajes con {tp['name']}")
            con=db()
            rows=con.execute("""SELECT * FROM direct_messages WHERE (from_id=? AND to_id=?) OR (from_id=? AND to_id=?) ORDER BY id""",
                              (profile["id"],target,target,profile["id"])).fetchall()
            con.close()
            for r in rows:
                who = "Tú" if r["from_id"]==profile["id"] else tp["name"]
                align = "right" if r["from_id"]==profile["id"] else "left"
                st.markdown(f'<div style="text-align:{align};margin:4px 0"><span class="chip">{who}: {r["content"]}</span></div>',unsafe_allow_html=True)
            msg=st.text_input("Escribe un mensaje",key="dm_text")
            if st.button("Enviar",type="primary") and msg.strip():
                con=db();con.execute("INSERT INTO direct_messages(from_id,to_id,content,created_at) VALUES(?,?,?,?)",
                                     (profile["id"],target,msg.strip(),datetime.now().isoformat(timespec="seconds")))
                con.commit();con.close();st.rerun()

    st.markdown("### 📥 Retos pendientes de otros")
    cs=challenges(profile["id"])
    incoming=[c for c in cs if c["opponent_id"]==profile["id"] and c["status"]=="pending"]
    if not incoming:
        st.caption("No tienes retos pendientes.")
    for c in incoming:
        a,b=st.columns([.8,.2]);a.write(f"📥 {c['challenger_name']} te reta a push-ups")
        if b.button("Aceptar",key=f"acc_comm_{c['id']}"):accept_challenge(c["id"],profile["id"]);st.rerun()

# ============================================================
# DIARIO DE COMIDAS
# ============================================================

elif page=="📷 Diario de comidas":
    need_profile(profile);section("DIARIO","Diario fotográfico de comidas","Foto → IA KSC → nutrientes → guardar.")
    ta,tb=st.tabs(["📸 Nueva comida","🗓️ Historial"])
    with ta:
        src=st.radio("Fuente",["Subir foto","Cámara"],horizontal=True)
        f=st.file_uploader("Imagen",type=["jpg","jpeg","png"],key="mealfile") if src=="Subir foto" else st.camera_input("Toma una foto")
        if f:
            jpeg=compact_jpeg(f.getvalue());st.image(jpeg,width=380)
            if st.button("✨ Analizar",type="primary"):
                try:
                    st.session_state["mealres"]=detect_foods(ai_key(),jpeg);st.session_state["mealjpeg"]=jpeg
                except Exception as e:st.error(f"No pude analizar la foto: {e}")
            res=st.session_state.get("mealres")
            if res and res.get("foods"):
                st.write(res.get("summary",""));calc=enrich(res,"meal")
                if calc:
                    tot=total_nutrition(calc);st.markdown("## Total del plato");show_metrics(tot)
                    if profile.get("allergies"):st.warning("Alergias/restricciones del perfil: "+profile["allergies"])
                    if st.button("🤖 Analizar para mi perfil"):
                        try:
                            ans=ksc_chat(profile,f"Analiza este plato para mí: {[(x['name'],x['grams']) for x in calc]}. Totales {tot}.")
                            st.markdown(ans);speak_reply(ans,key="meal_tts",autoplay=False)
                        except Exception as e:st.error(str(e))
                    c1,c2=st.columns(2);mt=c1.selectbox("Momento",["Desayuno","Media mañana","Almuerzo","Merienda","Cena","Otro"]);title=c2.text_input("Nombre",res.get("summary","Mi comida")[:80])
                    note=st.text_input("Nota")
                    if st.button("Guardar en diario",type="primary",use_container_width=True):
                        ip=save_jpeg(st.session_state["mealjpeg"],MEAL_DIR,f"meal_{profile['id']}")
                        add_meal(profile["id"],mt,title,calc,tot,ip,note);st.success("+15 puntos guardados")
                    pred=", ".join(x["name"] for x in calc);actual=st.text_input("Realmente había",pred);correct=st.radio("¿Acertó?",["Sí","No"],horizontal=True)=="Sí"
                    if st.button("Guardar prueba Eureka"):
                        conf=sum(x["confidence"] for x in calc)/len(calc);save_plate_test(profile["id"],pred,actual,correct,conf,tot["kcal"]);st.success("Guardada")
            elif res is not None:
                st.info("No detecté alimentos claros en la foto. Prueba con más luz o más de cerca.")
    with tb:
        days=st.selectbox("Periodo",["7 días","30 días","90 días"]);n=int(days.split()[0]);ms=meals_between(profile["id"],date.today()-timedelta(days=n-1),date.today())
        if ms:
            df=pd.DataFrame(ms);st.dataframe(df[["meal_date","meal_time","meal_type","title","kcal","protein","fiber"]],hide_index=True,use_container_width=True)
            daily=df.groupby("meal_date",as_index=False)[["kcal","protein","fiber"]].sum();st.line_chart(daily.set_index("meal_date"))
        else:st.info("Sin registros.")

# ============================================================
# ESCÁNER DE CÓDIGO DE BARRAS
# ============================================================

elif page=="🔎 Escáner de código de barras":
    need_profile(profile)
    section("CÓDIGO DE BARRAS","Escáner de productos","Escribe el número o sube una foto legible; IA KSC lo interpreta con una base pública de productos.")
    t1,t2=st.tabs(["⌨️ Número manual","📷 Foto del código"])
    code=None
    with t1:
        manual=st.text_input("Número de código de barras (EAN/UPC)")
        if st.button("Buscar producto",type="primary") and manual.strip():
            code=manual.strip()
    with t2:
        pic=st.file_uploader("Foto del código de barras o empaque",type=["jpg","jpeg","png"],key="bcpic")
        if pic:
            jpeg=compact_jpeg(pic.getvalue());st.image(jpeg,width=340)
            if st.button("Leer código de la foto",type="primary"):
                try:
                    d=ai_json(BARCODE_READ_PROMPT,jpeg)
                    digits=d.get("barcode_digits")
                    if digits:
                        st.success(f"Código leído: {digits}");code=str(digits)
                    else:
                        st.warning("No pude leer números claros. Prueba con más luz/enfoque, o usa el número manual.")
                        if d.get("product_guess"):
                            st.caption("Nombre de producto visible: "+str(d["product_guess"]))
                except Exception as e:
                    st.error(f"No pude leer la foto: {e}")

    if code:
        info, err = lookup_barcode(code)
        if err:
            st.error(err)
        elif info:
            st.markdown(f"## {info['name']}")
            if info.get("brand"):st.caption("Marca: "+info["brand"])
            if info.get("image"):st.image(info["image"],width=200)
            vals={"kcal":num(info.get("kcal_100g")),"protein":num(info.get("protein_100g")),
                  "carbs":num(info.get("carbs_100g")),"fat":num(info.get("fat_100g")),
                  "fiber":num(info.get("fiber_100g")),"sugars":num(info.get("sugars_100g")),
                  "sodium_mg":num(info.get("sodium_100g"))*1000,"sat_fat":0.0}
            st.caption("Valores por 100 g/ml (fuente: Open Food Facts)")
            show_metrics(vals)
            if info.get("nutriscore"):st.markdown(f"**Nutri-Score:** {info['nutriscore']}")
            if info.get("nova_group"):st.markdown(f"**Grupo NOVA (procesamiento):** {info['nova_group']}")
            if info.get("allergens"):st.warning("Alérgenos declarados: "+info["allergens"])
            if info.get("ingredients"):st.write("**Ingredientes:**",info["ingredients"])
            if profile.get("allergies"):st.warning("Tu perfil declara alergias a: "+profile["allergies"]+" — verifica siempre la etiqueta original.")
            if st.button("🤖 ¿Me conviene este producto?",type="primary"):
                try:
                    ans=ksc_chat(profile,f"Analiza este producto escaneado para mi perfil: {json.dumps(info,ensure_ascii=False)}")
                    st.markdown(ans);speak_reply(ans,key="bc_tts",autoplay=False)
                except Exception as e:st.error(str(e))

# ============================================================
# ETIQUETA
# ============================================================

elif page=="🧾 Escáner de etiqueta":
    need_profile(profile);section("ETIQUETAS","Escáner nutricional","Fotografía la tabla; verifica manualmente los números.")
    f=st.file_uploader("Foto de etiqueta",type=["jpg","jpeg","png"],key="label")
    if f:
        jpeg=compact_jpeg(f.getvalue());st.image(jpeg,width=420)
        if st.button("Leer etiqueta",type="primary"):
            try:
                st.session_state["labeldata"]=ai_json(LABEL_PROMPT,jpeg)
            except Exception as e:
                st.error(f"No pude leer la etiqueta: {e}")
    d=st.session_state.get("labeldata")
    if d:
        st.markdown("## "+str(d.get("product_name") or "Producto"))
        vals={"kcal":num(d.get("kcal")),"protein":num(d.get("protein_g")),"carbs":num(d.get("carbs_g")),"fat":num(d.get("fat_g")),
              "fiber":num(d.get("fiber_g")),"sugars":num(d.get("sugars_g")),"sodium_mg":num(d.get("sodium_mg")),"sat_fat":num(d.get("saturated_fat_g"))}
        show_metrics(vals)
        if d.get("ingredients"):st.write("Ingredientes:",d["ingredients"])
        if profile.get("allergies"):st.warning("Tu perfil declara: "+profile["allergies"]+". Verifica siempre la etiqueta original.")
        for w in d.get("warnings",[]) or []:st.warning(str(w))

# ============================================================
# COMPARAR PLATOS
# ============================================================

elif page=="⚖️ Comparar platos":
    need_profile(profile);section("COMPARADOR","Dos platos frente a frente","Compara calorías, proteína, fibra y sodio.")
    a,b=st.columns(2);fa=a.file_uploader("Foto A",type=["jpg","jpeg","png"],key="ca");fb=b.file_uploader("Foto B",type=["jpg","jpeg","png"],key="cb")
    if fa and fb and st.button("Analizar ambos",type="primary",use_container_width=True):
        try:
            st.session_state["compare"]=(detect_foods(ai_key(),compact_jpeg(fa.getvalue())),detect_foods(ai_key(),compact_jpeg(fb.getvalue())))
        except Exception as e:
            st.error(f"No pude analizar las fotos: {e}")
    if "compare" in st.session_state:
        ra,rb=st.session_state["compare"];left,right=st.columns(2);totals=[]
        for col,res,prefix,label in [(left,ra,"A","PLATO A"),(right,rb,"B","PLATO B")]:
            with col:
                st.markdown("## "+label);calc=enrich(res,prefix)
                if calc:tot=total_nutrition(calc);totals.append(tot);show_metrics(tot)
        if len(totals)==2:
            st.dataframe(pd.DataFrame([
                {"Métrica":"kcal","A":totals[0]["kcal"],"B":totals[1]["kcal"]},
                {"Métrica":"Proteína g","A":totals[0]["protein"],"B":totals[1]["protein"]},
                {"Métrica":"Fibra g","A":totals[0]["fiber"],"B":totals[1]["fiber"]},
                {"Métrica":"Sodio mg","A":totals[0]["sodium_mg"],"B":totals[1]["sodium_mg"]},
            ]),hide_index=True,use_container_width=True)
            if st.button("🤖 ¿Cuál encaja mejor conmigo?"):
                try:
                    ans=ksc_chat(profile,f"Compara estos platos para mi perfil. A={totals[0]}, B={totals[1]}. Explica contexto y alternativa.")
                    st.markdown(ans);speak_reply(ans,key="cmp_tts",autoplay=False)
                except Exception as e:st.error(str(e))

# ============================================================
# CHAT POR VOZ
# ============================================================

elif page=="💬 Chat por voz con IA KSC":
    need_profile(profile)
    section("CHAT","Habla con IA KSC","Habla o escribe. Al terminar de grabar se envía solo, IA KSC responde y se escucha con voz realista — sin copiar ni pegar nada.")

    c1,c2 = st.columns([1,1])
    with c1:
        voice_picker(location=c1)
    with c2:
        voice_on = st.toggle("🔊 Responder con voz automáticamente", value=True)

    st.markdown("#### 🎤 Habla tu pregunta")
    audio_val = st.audio_input("Toca para grabar, y suelta cuando termines de hablar")

    auto_prompt = None
    if audio_val is not None:
        audio_bytes = audio_val.getvalue()
        ahash = hashlib.md5(audio_bytes).hexdigest()
        if st.session_state.get("last_audio_hash") != ahash:
            st.session_state["last_audio_hash"] = ahash
            with st.spinner("🎙️ Transcribiendo lo que dijiste..."):
                text, err = transcribe_audio(audio_bytes)
            if err:
                st.error(err)
            elif text:
                auto_prompt = text
            else:
                st.warning("No entendí lo que dijiste, intenta de nuevo más cerca del micrófono.")

    for m in get_chat(profile["id"],30):
        with st.chat_message("assistant" if m["role"]=="assistant" else "user"):st.markdown(m["content"])

    typed_prompt = st.chat_input("...o escribe tu pregunta sobre comida")
    prompt = auto_prompt or typed_prompt

    if prompt:
        with st.chat_message("user"):
            if auto_prompt:st.caption("🎤 dictado por voz")
            st.markdown(prompt)
        try:ans=ksc_chat(profile,prompt)
        except Exception as e:ans="No pude responder ahora mismo: "+str(e)
        add_chat(profile["id"],"user",prompt);add_chat(profile["id"],"assistant",ans)
        with st.chat_message("assistant"):
            st.markdown(ans)
            speak_reply(ans, key=f"chat_tts_{len(get_chat(profile['id']))}", autoplay=voice_on)
    st.markdown("### 🧠 Memoria alimentaria")
    for m in memories(profile["id"]):st.write("•",m)

# ============================================================
# COCINA
# ============================================================

elif page=="🍳 Cocina inteligente":
    need_profile(profile);section("CHEF KSC","Cocina inteligente","Recetas, refrigeradora, presupuesto, Perú, jugos, postres, favoritos y sustituciones.")
    tabs=st.tabs(["🍽️ Recetas","🧊 Tengo esto","💰 Presupuesto","🇵🇪 Perú","❤️ Favoritos"])
    with tabs[0]:
        cat=st.selectbox("Tipo",["Desayuno","Almuerzo","Cena","Snack","Jugo/Batido","Postre nutritivo"])
        ing=st.text_input("Ingrediente opcional");mins=st.selectbox("Tiempo",["10 min","20 min","30 min","45+ min"])
        if st.button("Crear receta",type="primary"):
            try:
                st.session_state["recipe"]=ksc_chat(profile,f"Crea un {cat}, usa {ing or 'lo que convenga'}, tiempo {mins}. Ingredientes con cantidades, pasos y calorías aproximadas.")
            except Exception as e:
                st.error(f"No pude crear la receta: {e}")
        if st.session_state.get("recipe"):
            st.markdown(st.session_state["recipe"])
            c1,c2=st.columns(2)
            if c1.button("❤️ Guardar"):save_favorite(profile["id"],cat+" KSC",st.session_state["recipe"],cat);st.success("Guardada")
            replacement=c2.text_input("Ingrediente a sustituir",key="sub")
            if c2.button("🔄 Sustituir") and replacement:
                try:
                    st.markdown(ksc_chat(profile,f"En esta receta sustituye {replacement}: {st.session_state['recipe']}"))
                except Exception as e:
                    st.error(str(e))
    with tabs[1]:
        st.caption("Escribe lo que tienes y/o sube una foto de tu refrigeradora o despensa. Ambas opciones son independientes: puedes usar solo texto, solo foto, o ambos.")
        text=st.text_area("Ingredientes que tienes (opcional si subes foto)")
        pic=st.file_uploader("Foto de refrigeradora/despensa (opcional)",type=["jpg","jpeg","png"],key="fridge")
        if pic is not None:
            st.image(compact_jpeg(pic.getvalue()),width=340)
            if st.button("🔍 Detectar ingredientes de la foto",type="primary"):
                try:
                    with st.spinner("Analizando tu foto..."):
                        d=ai_json(FRIDGE_PROMPT,compact_jpeg(pic.getvalue()))
                    raw_ing = d.get("ingredients",[]) if isinstance(d, dict) else []
                    if not isinstance(raw_ing, list):
                        raw_ing = [raw_ing]
                    clean_ing = [str(x).strip() for x in raw_ing if x is not None and str(x).strip()]
                    st.session_state["fridge"]=clean_ing
                    if not st.session_state["fridge"]:
                        st.warning("No pude identificar ingredientes claros en esta foto. Prueba con más luz o escribe manualmente abajo.")
                except Exception as e:
                    st.error(f"No pude analizar la foto: {e}")
        detected=[str(x) for x in st.session_state.get("fridge",[]) if x is not None]
        if detected:st.write("Detectados en la foto:",", ".join(detected))
        if st.button("🍳 Crear recetas con esto",type="primary",use_container_width=True):
            if not text.strip() and not detected:
                st.warning("Escribe algún ingrediente o sube una foto primero.")
            else:
                try:
                    st.markdown(ksc_chat(profile,f"Tengo {text or 'nada escrito'}; además detectaste {detected or 'nada en foto'}. Dame 3 recetas usando lo que tengo."))
                except Exception as e:
                    st.error(f"No pude generar recetas: {e}")
    with tabs[2]:
        budget=st.number_input("Presupuesto S/",5.,500.,25.,1.);days=st.number_input("Días",1,7,1,1)
        if st.button("Crear menú económico"):
            try:
                st.markdown(ksc_chat(profile,f"Tengo S/{budget:.2f} para {days} días. Crea menú económico en Perú; precios solo aproximados."))
            except Exception as e:
                st.error(str(e))
    with tabs[3]:
        dish=st.selectbox("Plato peruano",["Ceviche","Arroz con pollo","Lomo saltado","Ají de gallina","Causa","Seco de chavelo","Menestra con arroz","Pollo a la brasa","Papa a la huancaína"])
        if st.button("Analizar / adaptar"):
            try:
                st.markdown(ksc_chat(profile,f"Analiza {dish} para mi perfil y dame una versión alternativa si conviene, conservando identidad del plato."))
            except Exception as e:
                st.error(str(e))
    with tabs[4]:
        favs=favorites(profile["id"])
        if not favs:st.info("Aún no guardaste recetas favoritas.")
        for f in favs:
            with st.expander("❤️ "+f["title"]):
                st.markdown(f["recipe"]);rating=st.slider("Puntuación",1,5,4,key=f"rt_{f['id']}");comment=st.text_input("Comentario",key=f"cm_{f['id']}")
                if st.button("Guardar valoración",key=f"sv_{f['id']}"):save_rating(profile["id"],f["id"],f["title"],rating,comment);st.success("Listo")

# ============================================================
# PLAN
# ============================================================

elif page=="📅 Plan semanal":
    need_profile(profile);section("PLAN","Semana + lista de compras","7 días, preferencias y exportación.")
    week=st.date_input("Inicio de semana",date.today()-timedelta(days=date.today().weekday()))
    if st.button("Generar plan",type="primary",use_container_width=True):
        try:
            st.session_state["plan"]=ai_json(PLAN_PROMPT+"\n"+profile_context(profile),max_tokens=3000)
        except Exception as e:
            st.error(f"No pude generar el plan: {e}")
    plan=st.session_state.get("plan") or (latest_plan(profile["id"]) or {}).get("plan")
    if plan:
        days=plan.get("days",[]);st.dataframe(pd.DataFrame(days),hide_index=True,use_container_width=True)
        st.markdown("### 🛒 Lista de compras")
        shopping = [str(x) for x in (plan.get("shopping_list",[]) or []) if x is not None]
        for x in shopping:st.write("•",x)
        c1,c2=st.columns(2)
        if c1.button("Guardar plan"):save_plan(profile["id"],week,plan);st.success("Guardado")
        html="<html><body><h1>Plan semanal IA KSC</h1>"+pd.DataFrame(days).to_html(index=False)+"<h2>Compras</h2><ul>"+"".join(f"<li>{x}</li>" for x in shopping)+"</ul></body></html>"
        c2.download_button("🖨️ Exportar HTML",html.encode(),file_name="plan_IA_KSC.html",mime="text/html",use_container_width=True)

# ============================================================
# AGUA & HÁBITOS
# ============================================================

elif page=="💧 Agua & hábitos":
    need_profile(profile);section("HÁBITOS","Agua y retos diarios","Registra tu hidratación en vasos y completa hábitos saludables.")
    water=water_today(profile["id"]);goal=int(profile.get("water_goal_ml") or 2000);pct=min(100,round(water/goal*100) if goal else 0)
    st.markdown(f'<div class="water-alert" style="display:flex;justify-content:space-between;align-items:center"><div><div class="kicker">HIDRATACIÓN DE HOY</div><div class="big">💧 {water} <span style="color:var(--muted);font-size:1rem">/ {goal} ml</span></div></div><div style="font-size:1.6rem;font-weight:950;color:var(--blue)">{pct}%</div></div>',unsafe_allow_html=True)
    st.progress(min(1.,water/goal if goal else 0))
    st.markdown("#### Toma agua y regístrala")
    cc=st.columns(len(WATER_UNITS))
    for col,(label,ml,icon) in zip(cc,WATER_UNITS):
        if col.button(f"{icon}\n{label}",use_container_width=True,key=f"w_{ml}"):log_water(profile["id"],ml);st.rerun()
    with st.expander("✏️ Cantidad personalizada"):
        custom_ml=st.number_input("ml",50,3000,250,50)
        if st.button("Registrar cantidad personalizada"):log_water(profile["id"],custom_ml);st.rerun()
    st.markdown("### 🎯 Retos de hoy")
    goals=daily_goals(profile["id"]);done_n=sum(1 for g in goals if g["completed"])
    st.progress(done_n/len(goals) if goals else 0);st.caption(f"{done_n}/{len(goals)} retos completados hoy")
    for g in goals:
        with st.container(border=True):
            a,b=st.columns([.8,.2]);a.write(("✅ " if g["completed"] else "⬜ ")+g["goal_name"])
            if not g["completed"] and b.button("Completar",key=f"goal_{g['id']}"):complete_goal(g["id"],profile["id"]);st.rerun()

# ============================================================
# RECOMPENSAS KSC (juego)
# ============================================================

elif page=="🏆 Recompensas KSC":
    need_profile(profile);section("JUEGO","Puntos, niveles y desbloqueos","Completa hábitos, recetas, quizzes y retos para subir de nivel.")
    pts,lvl,nxt=level_info(profile["id"]);r=streak(profile["id"])
    lc,rc=st.columns([2,1])
    with lc:
        st.markdown(f'<div class="level-card"><div class="kicker">NIVEL ACTUAL</div><div class="big" style="font-size:2rem">{lvl[2]} {lvl[1]}</div><div class="note">{pts} puntos acumulados</div></div>',unsafe_allow_html=True)
    with rc:
        st.markdown(f'<div class="streak-card"><span class="streak-flame">🔥</span><div><div class="streak-days">{r}</div><div class="streak-label">día{"s" if r!=1 else ""} seguidos</div></div></div>',unsafe_allow_html=True)
    if nxt:st.progress((pts-lvl[0])/(nxt[0]-lvl[0]));st.caption(f"Faltan {nxt[0]-pts} puntos para {nxt[2]} {nxt[1]}")
    st.markdown("### 🔓 Desbloqueos")
    cards=""
    for need,name in [(100,"🧃 Laboratorio de batidos"),(250,"🍰 Chef de postres"),(500,"📅 Planificador Maestro"),(900,"👑 Leyenda"),(1500,"💎 Elite")]:
        icon,label=name.split(" ",1);opened=pts>=need
        cards+=f'<div class="unlock-card {"open" if opened else "closed"}"><span class="uicon">{icon if opened else "🔒"}</span><div class="uname">{label}</div><div class="ureq">{need} puntos</div></div>'
    st.markdown(f'<div class="unlock-grid">{cards}</div>',unsafe_allow_html=True)
    st.markdown("### 💡 Cómo ganar puntos")
    st.markdown(textwrap.dedent("""
    - 📷 Registrar una comida: **+15**
    - 💧 Registrar agua: **+3**
    - ✅ Completar un reto diario: **+10**
    - ⚖️ Registrar peso: **+5**
    - 📏 Registrar medidas: **+5**
    - ❤️ Guardar receta favorita: **+5**
    - ⭐ Calificar receta: **+3**
    - 📅 Generar plan semanal: **+20**
    - 💪 Cada push-up en la Arena: **+1** (mínimo 10 por intento)
    - 🎓 Cada respuesta correcta del quiz: **según nivel**
    """))
    st.markdown("### 🏅 Ranking general")
    rank=leaderboard()
    medals={0:"🥇",1:"🥈",2:"🥉"}
    rows=""
    for i,row in rank.iterrows():
        cls=f"top{i+1}" if i<3 else ""
        pos=medals.get(i,f"#{i+1}")
        rows+=f'<div class="rank-row {cls}"><div class="rank-pos">{pos}</div><div class="rank-name">{row["name"]}</div><div class="rank-val">{int(row["points"])} pts</div></div>'
    st.markdown(rows if rows else '<div class="note">Aún no hay puntajes.</div>',unsafe_allow_html=True)

# ============================================================
# ARENA DE PUSH-UPS
# ============================================================

elif page=="💪 Arena de Push-Ups":
    need_profile(profile);section("PUSH-UP","Retos de 60 segundos entre perfiles","Reta a otro usuario; el conteo se hace con un esqueleto y anillo de progreso en vivo.")
    others=[p for p in list_profiles() if p["id"]!=profile["id"]]
    if others:
        mp={p["name"]:p["id"] for p in others};op=st.selectbox("Retar a",list(mp))
        if st.button("⚔️ Enviar reto",use_container_width=True):create_challenge(profile["id"],mp[op]);st.success("Reto enviado")
    else:st.info("Crea otro perfil para competir (ver 'Comunidad').")
    cs=challenges(profile["id"])
    incoming=[c for c in cs if c["opponent_id"]==profile["id"] and c["status"]=="pending"]
    for c in incoming:
        a,b=st.columns([.8,.2]);a.write(f"📥 {c['challenger_name']} te reta")
        if b.button("Aceptar",key=f"acc_{c['id']}"):accept_challenge(c["id"],profile["id"]);st.rerun()
    active=[c for c in cs if c["status"] in ("active","completed")]
    if active:
        opts={f"#{c['id']} {c['challenger_name']} vs {c['opponent_name']} · {c['status']}":c for c in active}
        label=st.selectbox("Reto",list(opts));c=opts[label];ats=attempts(c["id"])
        if ats:st.dataframe(pd.DataFrame(ats)[["name","reps","duration_seconds","created_at"]],hide_index=True,use_container_width=True)
        done={a["profile_id"] for a in ats}
        if c["status"]!="completed" and profile["id"] not in done:render_pushup_camera(c["id"],profile["id"])
        ats=attempts(c["id"])
        if len(ats)>=2:
            order=sorted(ats,key=lambda x:x["reps"],reverse=True)
            if order[0]["reps"]==order[1]["reps"]:
                st.markdown('<div class="winner-banner"><div class="wtitle">🤝 Empate</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="winner-banner"><span class="streak-flame" style="font-size:2rem">🏆</span><div class="wtitle">{order[0]["name"]}</div><div class="note">{order[0]["reps"]} push-ups en el minuto</div></div>',unsafe_allow_html=True)
            for a in ats:
                if a.get("video_path") and Path(a["video_path"]).exists():st.video(a["video_path"])
    st.markdown("### 🏅 Ranking Push-Up")
    pr=pushup_ranking();medals={0:"🥇",1:"🥈",2:"🥉"};rows=""
    for i,row in pr.iterrows():
        cls=f"top{i+1}" if i<3 else "";pos=medals.get(i,f"#{i+1}")
        rows+=f'<div class="rank-row {cls}"><div class="rank-pos">{pos}</div><div class="rank-name">{row["name"]}</div><div class="rank-val">{int(row["mejor_marca"])} reps</div></div>'
    st.markdown(rows if rows else '<div class="note">Sin intentos aún.</div>',unsafe_allow_html=True)

# ============================================================
# PROGRESO
# ============================================================

elif page=="📈 Mi progreso":
    need_profile(profile);section("PROGRESO","Peso y medidas","7, 30 o 90 días; tendencias, no diagnósticos.")
    t1,t2=st.tabs(["⚖️ Peso","📏 Medidas"])
    with t1:
        e=energy_estimate(profile);logs=weights(profile["id"])
        delta=None
        if len(logs)>=2:delta=logs[-1]["weight_kg"]-logs[0]["weight_kg"]
        d1,d2=st.columns(2)
        with d1:
            arrow="▲" if delta and delta>0 else "▼" if delta and delta<0 else "▬"
            dcolor="var(--orange)" if delta and delta>0 else "var(--green)" if delta and delta<0 else "var(--muted)"
            sub=f'<span style="color:{dcolor}">{arrow} {abs(delta):.1f} kg en el periodo</span>' if delta is not None else "Sin variación registrada aún"
            st.markdown(f'<div class="stat-card accent-green"><span class="icon">⚖️</span><div class="label">Peso actual</div><div class="value">{profile["weight_kg"]:.1f} kg</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)
        with d2:
            if e.get("enabled"):
                st.markdown(f'<div class="stat-card accent-blue"><span class="icon">🔥</span><div class="label">Mantenimiento</div><div class="value">{e["maintenance"]} kcal</div><div class="sub">rango {e["target_low"]}–{e["target_high"]} kcal/día</div></div>',unsafe_allow_html=True)
            else:
                st.info(e.get("reason"))
        with st.form("wform"):
            c1,c2,c3=st.columns(3);d=c1.date_input("Fecha",date.today());w=c2.number_input("Peso",30.,250.,float(profile["weight_kg"]),.1);wa=c3.number_input("Cintura opcional",0.,250.,0.,.5)
            note=st.text_input("Nota");save=st.form_submit_button("Guardar",type="primary")
        if save:add_weight(profile["id"],d,w,wa or None,note);st.rerun()
        if logs:
            df=pd.DataFrame(logs);df["log_date"]=pd.to_datetime(df["log_date"]);st.line_chart(df.set_index("log_date")[["weight_kg"]]);st.dataframe(df,hide_index=True,use_container_width=True)
    with t2:
        with st.form("mform"):
            d=st.date_input("Fecha",date.today(),key="md");cc=st.columns(5)
            waist=cc[0].number_input("Cintura",0.,250.,0.,.5);hip=cc[1].number_input("Cadera",0.,250.,0.,.5);chest=cc[2].number_input("Pecho",0.,250.,0.,.5);arm=cc[3].number_input("Brazo",0.,100.,0.,.5);thigh=cc[4].number_input("Muslo",0.,150.,0.,.5)
            note=st.text_input("Nota",key="mn");save=st.form_submit_button("Guardar medidas",type="primary")
        if save:add_measure(profile["id"],d,waist or None,hip or None,chest or None,arm or None,thigh or None,note);st.rerun()
        logs=measures(profile["id"])
        if logs:st.dataframe(pd.DataFrame(logs),hide_index=True,use_container_width=True)

# ============================================================
# ACADEMIA KSC (QUIZ CON NIVELES)
# ============================================================

elif page=="🎓 Academia KSC (quiz)":
    section("EDUCACIÓN","Academia KSC","Aprende sobre nutrición, sube de nivel y gana puntos extra.")

    QUIZ_LEVELS = {
        "🟢 Básico": {
            "points": 4,
            "questions": [
                ("¿Qué nutriente ayuda especialmente a construir/reparar tejidos?",["Proteína","Sodio","Azúcar"],0),
                ("¿Qué alimentos suelen aportar fibra?",["Frutas y verduras","Gaseosa","Sal"],0),
                ("¿Una foto mide exactamente las calorías?",["Sí","No, solo estima si no se pesa"],1),
                ("Si tienes alergia declarada, ¿qué haces?",["Verificar etiqueta","Ignorar ingredientes"],0),
                ("¿Una sola comida define toda tu alimentación?",["Sí","No"],1),
                ("¿Cuál de estos es un carbohidrato principal?",["Arroz","Pollo","Aceite"],0),
            ],
        },
        "🟡 Intermedio": {
            "points": 7,
            "questions": [
                ("¿Qué macronutriente aporta más calorías por gramo?",["Grasas","Proteínas","Carbohidratos"],0),
                ("¿Qué significa 'Nutri-Score A'?",["Perfil nutricional más favorable","Producto más barato","Producto sin envase"],0),
                ("¿Qué mide aproximadamente el IMC?",["Relación entre peso y talla","Cantidad de músculo exacta","Nivel de hidratación"],0),
                ("¿Qué grupo NOVA agrupa alimentos ultraprocesados?",["Grupo 4","Grupo 1","Grupo 2"],0),
                ("¿Por qué el sodio en exceso preocupa en las etiquetas?",["Se asocia a presión arterial elevada","Da energía extra","Mejora la digestión"],0),
                ("¿Qué aporta principalmente la fibra a la digestión?",["Favorece el tránsito intestinal","Sube el azúcar rápido","Es una grasa saturada"],0),
            ],
        },
        "🔴 Avanzado": {
            "points": 10,
            "questions": [
                ("En una etiqueta, ¿qué representa '% VD'?",["El aporte de ese nutriente respecto a una dieta de referencia","El precio del producto","La fecha de vencimiento"],0),
                ("¿Qué diferencia hay entre azúcares totales y azúcares añadidos?",["Los añadidos no están naturalmente en el alimento","Son exactamente lo mismo","Los añadidos siempre son menos"],0),
                ("¿Qué factor de actividad se usaría para alguien muy sedentario en el cálculo de mantenimiento?",["El más bajo de la escala","El más alto de la escala","No influye"],0),
                ("¿Por qué en menores de edad no se fijan déficits calóricos estrictos?",["Porque están en pleno crecimiento y desarrollo","Porque no sienten hambre","Porque no digieren grasas"],0),
                ("¿Qué indica un Nutri-Score E frente a uno A?",["Un perfil nutricional menos favorable en conjunto","Que el producto es orgánico","Que tiene menos envase"],0),
            ],
        },
    }

    level_choice = st.radio("Elige tu nivel", list(QUIZ_LEVELS.keys()), horizontal=True)
    qs = QUIZ_LEVELS[level_choice]["questions"]
    pts_per_q = QUIZ_LEVELS[level_choice]["points"]

    with st.form(f"quiz_{level_choice}"):
        answers=[]
        for i,(q,o,ci) in enumerate(qs):
            ans=st.radio(q,o,key=f"q_{level_choice}_{i}",index=None)
            answers.append((ans,o,ci))
        submitted=st.form_submit_button("Corregir",type="primary",use_container_width=True)
    if submitted:
        if any(a[0] is None for a in answers):
            st.warning("Responde todas las preguntas de este nivel.")
        else:
            score=sum(1 for ans,o,ci in answers if ans==o[ci])
            st.success(f"{score}/{len(qs)} correctas en nivel {level_choice}")
            if profile:
                gained=score*pts_per_q
                add_points(profile["id"],gained,f"Quiz {level_choice}")
                con=db();con.execute("INSERT INTO quiz_results(profile_id,quiz_date,level,correct,total) VALUES(?,?,?,?,?)",
                                     (profile["id"],str(date.today()),level_choice,score,len(qs)))
                con.commit();con.close()
                st.info(f"+{gained} puntos")
            else:
                st.info("Desbloquea un perfil para guardar tus puntos.")

    if profile:
        con=db();hist=con.execute("SELECT * FROM quiz_results WHERE profile_id=? ORDER BY id DESC LIMIT 10",(profile["id"],)).fetchall();con.close()
        if hist:
            st.markdown("### 📜 Tu historial de quiz")
            st.dataframe(pd.DataFrame([dict(r) for r in hist]),hide_index=True,use_container_width=True)

# ============================================================
# EUREKA LAB
# ============================================================

elif page=="🧪 Eureka Lab":
    section("CIENCIA","Eureka Lab","Precisión, errores, versiones y matriz de confusión.")
    con=db();df=pd.read_sql_query("""SELECT pt.*,p.name profile_name FROM plate_tests pt LEFT JOIN profiles p ON p.id=pt.profile_id ORDER BY pt.id""",con);con.close()
    if df.empty:st.warning("Aún no hay pruebas.")
    else:
        st.markdown(textwrap.dedent(f"""
        <div class="stat-grid" style="grid-template-columns:repeat(3,1fr)">
          <div class="stat-card accent-green"><span class="icon">🧪</span><div class="label">Pruebas</div><div class="value">{len(df)}</div></div>
          <div class="stat-card accent-blue"><span class="icon">🎯</span><div class="label">Precisión</div><div class="value">{df['correct'].mean()*100:.1f}%</div></div>
          <div class="stat-card accent-purple"><span class="icon">📊</span><div class="label">Confianza</div><div class="value">{df['avg_conf'].mean():.1f}%</div></div>
        </div>"""),unsafe_allow_html=True)
        ver=df.groupby("app_version",dropna=False).agg(pruebas=("id","count"),precision=("correct","mean"),confianza=("avg_conf","mean")).reset_index();ver["precision"]*=100
        st.markdown("### Versiones");st.dataframe(ver,hide_index=True,use_container_width=True)
        pred=df["predicted_foods"].fillna("").str.split(",").str[0].str.strip();actual=df["actual_foods"].fillna("").str.split(",").str[0].str.strip()
        st.markdown("### Matriz de confusión");st.dataframe(pd.crosstab(actual,pred,rownames=["Real"],colnames=["Predicción"]),use_container_width=True)
        st.markdown("### Errores");st.dataframe(df[df["correct"]==0][["created_at","predicted_foods","actual_foods","avg_conf","app_version"]],hide_index=True,use_container_width=True)
        st.download_button("Descargar CSV",df.to_csv(index=False).encode("utf-8-sig"),file_name="eureka_IA_KSC.csv",mime="text/csv")

# ============================================================
# CONFIG
# ============================================================

elif page=="⚙️ Configuración":
    section("SISTEMA","Configuración","Claves, módulos y estado de la app.")
    ok=bool(ai_key())
    st.markdown(textwrap.dedent(f"""
    <div class="stat-card {'accent-green' if ok else 'accent-orange'}" style="max-width:420px">
      <span class="icon">{'✅' if ok else '⚠️'}</span>
      <div class="label">Estado de IA KSC</div>
      <div class="value" style="font-size:1.1rem">{'GROQ_API_KEY encontrada' if ok else 'Falta GROQ_API_KEY'}</div>
    </div>"""),unsafe_allow_html=True)
    st.code('GROQ_API_KEY = "TU_TOKEN"\n# opcional:\nUSDA_API_KEY = "TU_CLAVE_USDA"',language="toml")
    if st.button("Probar IA",type="primary"):
        try:
            ids={m.id for m in ai_client(ai_key()).models.list().data}
            st.success("IA KSC lista." if AI_MODEL in ids else "Conexión OK, modelo no visible.")
        except Exception as e:st.error(str(e))
    st.markdown("### 💪 Arena Push-Up (cámara con esqueleto)")
    st.code("pip install streamlit-webrtc mediapipe av opencv-python-headless",language="powershell")
    st.markdown("### 🔊 Voz realista (gratis)")
    st.caption("Para hablar, IA KSC usa Edge TTS (voces neuronales de Microsoft, gratis y sin API key) — elige la voz en la barra lateral. Para escucharte, usa Whisper a través de tu misma GROQ_API_KEY: grabas con el micrófono del navegador (st.audio_input) y se transcribe y envía automáticamente, sin copiar ni pegar nada. No usa ElevenLabs.")
    if not EDGE_TTS_OK:
        st.error("El paquete 'edge-tts' no está instalado en este entorno. Agrégalo a requirements.txt y vuelve a desplegar para activar la voz.")
    st.markdown("### 🔎 Código de barras")
    st.caption("Usa la base pública y gratuita Open Food Facts. Los productos consultados se guardan en caché local (.ksc_data/barcode_cache.json) para funcionar más rápido la próxima vez.")
    st.markdown("### 💾 Persistencia de datos")
    st.info("Todos los perfiles, comidas, puntos y retos se guardan en .ksc_data/ dentro del servidor donde corre la app. No se borran al cerrar el navegador. Si despliegas en un hosting con almacenamiento temporal, monta un volumen persistente en esa carpeta.")
    st.info("IA KSC es un asistente nutricional educativo diseñado por los alumnos César Zapata, Alex Timaná García, Atarama Portocarrero y André Requena.")

import os
import io
import re
import json
import math
import time
import base64
import calendar as pycalendar
import hashlib
import html
import secrets as pysecrets
import sqlite3
import threading
import textwrap
from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
from groq import Groq, AuthenticationError, RateLimitError, APIConnectionError, BadRequestError

# ============================================================
# FitGlass / NutriVision — Liquid Glass Edition
# ============================================================

APP_VERSION = "8.0"
AI_MODEL = "qwen/qwen3.6-27b"
AI_CHAT_FALLBACK_MODELS = ["qwen/qwen3.6-27b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
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
    page_title="FitGlass · NutriVision",
    page_icon=None,
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

.fg-health-panel{padding:22px 24px;margin:8px 0 18px;position:relative;overflow:hidden;background:linear-gradient(135deg,rgba(255,255,255,.11),rgba(255,255,255,.035))}
.fg-health-panel:before{content:"";position:absolute;inset:-40% auto auto 55%;width:320px;height:220px;background:radial-gradient(circle,rgba(130,255,190,.16),transparent 68%);filter:blur(12px);pointer-events:none}
.fg-health-head{display:flex;align-items:center;justify-content:space-between;gap:14px;position:relative}
.fg-health-title{font-size:1.22rem;font-weight:900;color:#fff}
.fg-health-badge{padding:7px 12px;border-radius:999px;background:rgba(126,255,189,.08);border:1px solid rgba(126,255,189,.25);color:#9bffd1;font-size:12px;font-weight:850}
.fg-health-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px;position:relative}
.fg-health-grid>div{padding:14px;border-radius:17px;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08)}
.fg-health-grid span,.fg-health-grid small{display:block;color:#8fa79a;font-size:11px}.fg-health-grid b{display:block;color:#fff;font-size:1.18rem;margin:4px 0}.fg-health-foot{margin-top:13px;color:#95aea0;font-size:11px;position:relative}.fg-health-foot b{color:#dff9eb}
.fg-chat-row{display:flex;align-items:flex-start;gap:10px;margin:10px 0;animation:fadeInUp .28s var(--ease)}
.fg-chat-row.user{justify-content:flex-end}.fg-chat-row.user .fg-chat-bubble{order:0}.fg-chat-row.user .fg-chat-avatar{order:1}
.fg-chat-avatar{width:44px;height:44px;border-radius:50%;object-fit:cover;flex:0 0 44px;border:1px solid rgba(255,255,255,.14);box-shadow:0 10px 30px rgba(0,0,0,.2)}
.fg-chat-avatar.user{border-color:rgba(142,182,255,.32)}.fg-chat-avatar.assistant{border-color:rgba(126,255,189,.3)}
.fg-chat-avatar.fallback{display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1a3527,#1a263d);color:#fff;font-weight:900;font-size:16px}
.fg-chat-bubble{max-width:min(760px,78%);padding:12px 15px;border-radius:18px;background:linear-gradient(135deg,rgba(255,255,255,.085),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.1);backdrop-filter:blur(20px) saturate(160%);color:#eefaf4;line-height:1.64}
.fg-chat-row.user .fg-chat-bubble{background:linear-gradient(135deg,rgba(143,183,255,.10),rgba(255,255,255,.03))}.fg-chat-name{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#8fa89a;font-weight:850;margin-bottom:4px}

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

st.markdown("""
<style>
/* Liquid Glass UI layer */
.stApp{background:
 radial-gradient(700px 420px at 8% -8%, rgba(110,255,188,.22), transparent 62%),
 radial-gradient(520px 360px at 96% 4%, rgba(120,165,255,.20), transparent 60%),
 radial-gradient(780px 520px at 52% 102%, rgba(200,130,255,.14), transparent 62%),
 linear-gradient(180deg,#07110e 0%,#050b09 100%);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;background:
 linear-gradient(120deg,transparent 0%,rgba(255,255,255,.035) 45%,transparent 55%);background-size:220% 220%;animation:kscShine 12s ease-in-out infinite;z-index:0}
@keyframes kscShine{0%,100%{background-position:-30% 0}50%{background-position:130% 100%}}
.block-container{padding-top:1.25rem;max-width:1440px}
[data-testid="stSidebar"]{display:none}
.glass-surface{background:linear-gradient(135deg,rgba(255,255,255,.105),rgba(255,255,255,.035));border:1px solid rgba(255,255,255,.16);box-shadow:inset 0 1px 0 rgba(255,255,255,.16),0 20px 80px rgba(0,0,0,.28);backdrop-filter:blur(30px) saturate(170%);-webkit-backdrop-filter:blur(30px) saturate(170%);border-radius:28px;position:relative;overflow:hidden}
.glass-surface::before{content:"";position:absolute;inset:1px;border-radius:27px;pointer-events:none;background:linear-gradient(135deg,rgba(255,255,255,.14),transparent 28%,transparent 70%,rgba(255,255,255,.05))}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 14px 12px 18px;margin-bottom:16px}
.brandmark{display:flex;align-items:center;gap:10px;font-weight:900;letter-spacing:-.03em;color:#f5fff9}
.brand-svg{width:34px;height:34px;display:block;filter:drop-shadow(0 0 16px rgba(110,255,188,.38))}
.topnav{display:flex;justify-content:center;gap:8px;margin:8px auto 24px}
.topnav .stButton>button{border-radius:18px!important;border:1px solid rgba(255,255,255,.10)!important;background:rgba(255,255,255,.055)!important;color:#cfe2d8!important;min-height:44px;font-weight:850!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.08)!important;transition:.24s ease!important}
.topnav .stButton>button:hover{transform:translateY(-1px);background:rgba(255,255,255,.09)!important;border-color:rgba(255,255,255,.2)!important}
.glass-primary .stButton>button{background:linear-gradient(135deg,rgba(109,255,188,.34),rgba(86,156,255,.20))!important;border:1px solid rgba(155,255,218,.42)!important;box-shadow:0 12px 34px rgba(80,255,180,.12),inset 0 1px 0 rgba(255,255,255,.20)!important}
.metric-ring{width:148px;height:148px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--ring) calc(var(--pct)*1%),rgba(255,255,255,.08) 0);position:relative;box-shadow:0 0 0 1px rgba(255,255,255,.10),0 16px 38px rgba(0,0,0,.22)}
.metric-ring::after{content:"";width:116px;height:116px;border-radius:50%;background:rgba(6,15,12,.78);border:1px solid rgba(255,255,255,.10);box-shadow:inset 0 1px 0 rgba(255,255,255,.08)}
.metric-ring>div{position:absolute;text-align:center;z-index:2}
.ring-value{font-size:1.7rem;font-weight:950;color:#fff;letter-spacing:-.04em}
.ring-label{font-size:.72rem;color:#9fbcaf;margin-top:2px}
.bars{display:grid;gap:14px}.bar-row{display:grid;grid-template-columns:110px 1fr 64px;gap:12px;align-items:center}.bar{height:10px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}.bar>i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,rgba(112,255,192,.95),rgba(120,164,255,.95));box-shadow:0 0 16px rgba(112,255,192,.22)}
.form-hint{font-size:.84rem;color:#9eb7aa;line-height:1.5;margin:4px 0 14px}.onboarding-title{font-size:clamp(2.2rem,6vw,4.7rem);line-height:.96;font-weight:950;letter-spacing:-.065em;color:#fff}.onboarding-title span{background:linear-gradient(90deg,#d8fff0,#75ffba,#91baff);-webkit-background-clip:text;background-clip:text;color:transparent}
@media(max-width:820px){.topnav{position:sticky;top:8px;z-index:30}.topnav .stButton>button{min-height:48px}.glass-surface{border-radius:22px}.metric-ring{width:130px;height:130px}.metric-ring::after{width:102px;height:102px}}
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

def ensure_col(con, table, definition=None):
    """Safely add a missing SQLite column.

    Backward compatible with the old two-argument migration call used by
    earlier deployments: ensure_col(con, "column TYPE ..."). In that case
    the target table defaults to profiles because those legacy migrations
    only referred to profile fields.
    """
    if definition is None:
        definition = table
        table = "profiles"
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
    ensure_col(con, "profiles", "app_version TEXT DEFAULT ''")
    ensure_col(con, "quiz_results", "level TEXT DEFAULT 'Básico'")
    ensure_col(con, "profiles", "diet_style TEXT DEFAULT 'Omnívora'")
    ensure_col(con, "profiles", "intolerances TEXT DEFAULT ''")
    ensure_col(con, "profiles", "calorie_target REAL DEFAULT 0")
    ensure_col(con, "profiles", "protein_target REAL DEFAULT 0")
    ensure_col(con, "profiles", "carbs_target REAL DEFAULT 0")
    ensure_col(con, "profiles", "fat_target REAL DEFAULT 0")
    ensure_col(con, "profiles", "region TEXT DEFAULT 'Perú'")
    ensure_col(con, "profiles", "notes TEXT DEFAULT ''")
    ensure_col(con, "profiles", "reminders_enabled INTEGER DEFAULT 1")
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
        allergies,special_state,photo_path,created_at,pin_hash,water_goal_ml,
        diet_style,intolerances,calorie_target,protein_target,carbs_target,fat_target
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        d["name"],d["age"],d["sex_energy"],d["height_cm"],d["weight_kg"],
        d["activity"],d["goal"],d.get("favorite_foods",""),d.get("favorite_fruits",""),
        d.get("favorite_vegetables",""),d.get("avoid_foods",""),d.get("allergies",""),
        d.get("special_state","Ninguno"),d.get("photo_path",""),datetime.now().isoformat(timespec="seconds"),
        d.get("pin_hash",""),d.get("water_goal_ml",2000),d.get("diet_style","Omnívora"),
        d.get("intolerances",""),d.get("calorie_target",0),d.get("protein_target",0),
        d.get("carbs_target",0),d.get("fat_target",0)
    ))
    pid=cur.lastrowid
    con.execute("UPDATE profiles SET region=?, notes=?, reminders_enabled=? WHERE id=?",
                (d.get("region","Perú"),d.get("notes",""),int(d.get("reminders_enabled",1)),pid))
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
       allergies=?,special_state=?,photo_path=?,water_goal_ml=?,diet_style=?,intolerances=?,
       calorie_target=?,protein_target=?,carbs_target=?,fat_target=?,region=?,notes=?,reminders_enabled=?
      WHERE id=?
    """,(
        d["name"],d["age"],d["sex_energy"],d["height_cm"],d["weight_kg"],d["activity"],d["goal"],
        d.get("favorite_foods",""),d.get("favorite_fruits",""),d.get("favorite_vegetables",""),d.get("avoid_foods",""),
        d.get("allergies",""),d.get("special_state","Ninguno"),d.get("photo_path",""),d.get("water_goal_ml",2000),
        d.get("diet_style","Omnívora"),d.get("intolerances",""),d.get("calorie_target",0),d.get("protein_target",0),
        d.get("carbs_target",0),d.get("fat_target",0),d.get("region","Perú"),d.get("notes",""),int(d.get("reminders_enabled",1)),pid
    ))
    con.commit();con.close()
    sync_community()

def delete_profile(pid):
    con=db()
    for t in ["weight_logs","measurements","chat_messages","memories","meal_diary","hydration",
              "healthy_goals","point_events","favorites","recipe_ratings","weekly_plans","quiz_results"]:
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
    (0,"Semilla KSC",""),(100,"Explorador KSC",""),
    (250,"NutriRanger",""),(500,"Maestro KSC",""),
    (900,"Leyenda KSC",""),(1500,"Elite KSC","")
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

def water_amount_on(pid, d):
    con=db(); r=con.execute("SELECT COALESCE(SUM(ml),0) ml FROM hydration WHERE profile_id=? AND log_date=?",(pid,str(d))).fetchone(); con.close(); return int(r["ml"])

def streak_status(pid, d=None):
    d=d or date.today(); totals,meals=day_totals(pid,d); water=water_amount_on(pid,d); p=get_profile(pid); e=energy_estimate(p) if p else {}
    target=float(e.get("target",0) or 0); prot_target=float(e.get("protein_target",0) or 0); water_goal=float((p or {}).get("water_goal_ml",2000) or 2000)
    active=bool(meals or water>0)
    kcal_ok=bool(target and totals["kcal"]>=target*.90 and totals["kcal"]<=target*1.10)
    protein_ok=prot_target<=0 or totals["protein"]>=prot_target*.90
    water_ok=water>=water_goal*.90
    nutrition=bool(meals) and kcal_ok and protein_ok and water_ok
    perfect=bool(meals) and kcal_ok and protein_ok and water_ok and totals["fiber"]>=8
    return {"active":active,"nutrition":nutrition,"perfect":perfect,"kcal_ok":kcal_ok,"protein_ok":protein_ok,"water_ok":water_ok,"kcal":totals["kcal"],"target":target,"water":water,"water_goal":water_goal}

def current_streak(pid, kind="active"):
    d=date.today(); n=0
    while True:
        stt=streak_status(pid,d); ok=stt["active"] if kind=="active" else (stt["perfect"] if kind=="perfect" else stt["nutrition"])
        if not ok:
            if n==0 and d==date.today(): d-=timedelta(days=1); continue
            break
        n+=1; d-=timedelta(days=1)
    return n

def streak(pid): return current_streak(pid,"active")

def usage_days(pid):
    con=db(); dates=set()
    for table,col in (("meal_diary","meal_date"),("hydration","log_date"),("point_events","event_date")):
        rows=con.execute(f"SELECT DISTINCT {col} d FROM {table} WHERE profile_id=?",(pid,)).fetchall(); dates.update(str(r["d"]) for r in rows if r["d"])
    con.close(); return dates

def streak_calendar(pid, days=35):
    """Compatibilidad: mantiene estados recientes; la interfaz usa un calendario mensual real."""
    end=date.today(); start=end-timedelta(days=days-1); out=[]; d=start
    while d<=end:
        stt=streak_status(pid,d)
        status="perfect" if stt["perfect"] else "nutrition" if stt["nutrition"] else "active" if stt["active"] else "empty"
        out.append({"date":d,"status":status}); d+=timedelta(days=1)
    return out

def real_month_calendar_data(pid, year, month):
    matrix=pycalendar.monthcalendar(year, month)
    con=db(); prefix=f"{year:04d}-{month:02d}-%"
    meals_by_day={r["meal_date"]:{"count":int(r["c"]),"kcal":float(r["kcal"] or 0)} for r in con.execute("SELECT meal_date,COUNT(*) c,COALESCE(SUM(kcal),0) kcal FROM meal_diary WHERE profile_id=? AND meal_date LIKE ? GROUP BY meal_date",(pid,prefix)).fetchall()}
    water_by_day={r["log_date"]:int(r["ml"] or 0) for r in con.execute("SELECT log_date,COALESCE(SUM(ml),0) ml FROM hydration WHERE profile_id=? AND log_date LIKE ? GROUP BY log_date",(pid,prefix)).fetchall()}
    con.close(); return matrix,meals_by_day,water_by_day

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
    ("½ vaso", 125, ""),
    ("1 vaso", 250, ""),
    ("2 vasos", 500, ""),
    ("1 botella", 750, ""),
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
    """Estimate daily energy and always expose stored onboarding targets to the dashboard."""
    if not p:
        return {"enabled":False,"reason":"Sin perfil.","target":0,"protein_target":0,"carbs_target":0,"fat_target":0}
    stored_target=num(p.get("calorie_target"))
    stored_protein=num(p.get("protein_target"))
    stored_carbs=num(p.get("carbs_target"))
    stored_fat=num(p.get("fat_target"))
    age=int(p.get("age",0) or 0)
    if age < 18:
        return {"enabled":False,"reason":"En menores de 18 años FitGlass no fija déficit, superávit ni metas calóricas para cambiar de peso.",
                "target":stored_target,"protein_target":stored_protein,"carbs_target":stored_carbs,"fat_target":stored_fat}
    if p.get("special_state") in ("Embarazo","Lactancia"):
        return {"enabled":False,"reason":"En embarazo o lactancia la meta energética debe definirse con seguimiento profesional.",
                "target":stored_target,"protein_target":stored_protein,"carbs_target":stored_carbs,"fat_target":stored_fat}
    sex=p.get("sex_energy")
    w=float(p.get("weight_kg") or 0); h=float(p.get("height_cm") or 0)
    if sex not in ("Masculino","Femenino") or not h or not w or not age:
        if stored_target > 0:
            return {"enabled":True,"maintenance":0,"target_low":stored_target,"target_high":stored_target,
                    "target":round(stored_target),"bmi":round(w/((h/100)**2),1) if h else 0,
                    "protein_target":round(stored_protein),"carbs_target":round(stored_carbs),"fat_target":round(stored_fat),
                    "estimated":True,"reason":"Referencia basada en los datos registrados; falta una variable fisiológica para calcular el mantenimiento con Mifflin-St Jeor."}
        return {"enabled":False,"reason":"Selecciona la variable fisiológica usada por la ecuación para estimar energía.",
                "target":0,"protein_target":0,"carbs_target":0,"fat_target":0}
    bmr=10*w+6.25*h-5*age+(5 if sex=="Masculino" else -161)
    maintenance=bmr*ACTIVITY_FACTORS.get(p.get("activity"),1.375)
    goal=p.get("goal")
    if goal=="Perder peso": low,high=maintenance-400,maintenance-250
    elif goal in ("Ganar peso","Ganar masa muscular"): low,high=maintenance+150,maintenance+300
    else: low,high=maintenance-100,maintenance+100
    low=max(1200,low); high=max(low,high)
    target=stored_target if stored_target>0 else (low+high)/2
    bmi=w/((h/100)**2)
    protein=stored_protein if stored_protein>0 else (w*1.6 if goal in ("Perder peso","Ganar masa muscular") else w*1.2)
    fat=stored_fat if stored_fat>0 else (target*0.28/9)
    carbs=stored_carbs if stored_carbs>0 else max(0,(target-protein*4-fat*9)/4)
    return {"enabled":True,"maintenance":round(maintenance),"target_low":round(low),"target_high":round(high),
            "target":round(target),"bmi":round(bmi,1),"protein_target":round(protein),"carbs_target":round(carbs),
            "fat_target":round(fat),"estimated":stored_target<=0}
def calculate_targets(sex, age, height_cm, weight_kg, activity, goal):
    """Estimate daily energy and macro targets using the same logic as energy_estimate()."""
    age=int(age); h=float(height_cm); w=float(weight_kg)
    if age < 18 or sex not in ("Masculino", "Femenino") or not h or not w:
        # Safe neutral fallback. The profile can still be completed and targets edited later.
        target=max(1200, round(w * 24))
        protein=round(w * (1.4 if goal in ("Perder peso","Ganar masa muscular") else 1.2))
        fat=round(target * 0.28 / 9)
        carbs=max(0, round((target - protein*4 - fat*9) / 4))
        return target, protein, carbs, fat
    bmr=10*w + 6.25*h - 5*age + (5 if sex=="Masculino" else -161)
    maintenance=bmr*ACTIVITY_FACTORS.get(activity,1.375)
    if goal=="Perder peso":
        low,high=maintenance-400,maintenance-250
    elif goal in ("Ganar peso","Ganar masa muscular"):
        low,high=maintenance+150,maintenance+300
    else:
        low,high=maintenance-100,maintenance+100
    low=max(1200,low); high=max(low,high)
    target=round((low+high)/2)
    protein=round(w*(1.6 if goal in ("Perder peso","Ganar masa muscular") else 1.2))
    fat=round(target*0.28/9)
    carbs=max(0,round((target-protein*4-fat*9)/4))
    return target, protein, carbs, fat

def health_metrics(p):
    """Deterministic health indicators from profile data; AI explains them, it does not invent them."""
    if not p:
        return {"bmi":0,"bmi_label":"Sin perfil","bmr":0,"maintenance":0,"calorie_target":0,
                "protein_target":0,"carbs_target":0,"fat_target":0,"weight_range_low":0,"weight_range_high":0,
                "height_m":0,"waist_height":0,"waist_label":"Sin dato"}
    h=float(p.get("height_cm") or 0)/100
    w=float(p.get("weight_kg") or 0)
    age=int(p.get("age") or 0)
    sex=p.get("sex_energy")
    bmi=round(w/(h*h),1) if h>0 and w>0 else 0
    if bmi<=0: label="Sin dato"
    elif bmi<18.5: label="Bajo peso"
    elif bmi<25: label="Rango saludable"
    elif bmi<30: label="Sobrepeso"
    else: label="Obesidad"
    bmr=0
    maintenance=0
    if h>0 and w>0 and age>0 and sex in ("Masculino","Femenino"):
        bmr=10*w+6.25*(h*100)-5*age+(5 if sex=="Masculino" else -161)
        maintenance=bmr*ACTIVITY_FACTORS.get(p.get("activity"),1.375)
    e=energy_estimate(p)
    low=w if not h else 18.5*(h*h)
    high=w if not h else 24.9*(h*h)
    waist=float(p.get("waist_cm") or 0)
    whr=round(waist/(h*100),2) if waist>0 and h>0 else 0
    waist_label="Sin dato"
    if whr:
        waist_label="Referencia a vigilar" if whr>=0.5 else "Referencia favorable"
    return {
        "bmi":bmi,"bmi_label":label,"bmr":round(bmr),"maintenance":round(maintenance),
        "calorie_target":int(e.get("target") or p.get("calorie_target") or 0),
        "protein_target":int(e.get("protein_target") or p.get("protein_target") or 0),
        "carbs_target":int(e.get("carbs_target") or p.get("carbs_target") or 0),
        "fat_target":int(e.get("fat_target") or p.get("fat_target") or 0),
        "weight_range_low":round(low,1) if low else 0,"weight_range_high":round(high,1) if high else 0,
        "height_m":round(h,2),"waist_cm":waist,"waist_height":whr,"waist_label":waist_label,
    }

def metrics_explanation(m):
    if not m or not m.get("bmi"):
        return "Completa talla y peso para obtener tus indicadores."
    return (f"IMC {m['bmi']} ({m['bmi_label']}). Metabolismo basal estimado {m['bmr']} kcal/día; "
            f"gasto de mantenimiento estimado {m['maintenance']} kcal/día. Referencia diaria: "
            f"{m['calorie_target']} kcal, {m['protein_target']} g de proteína, {m['carbs_target']} g de carbohidratos y "
            f"{m['fat_target']} g de grasa. Rango de peso correspondiente a IMC 18,5–24,9: "
            f"{m['weight_range_low']}–{m['weight_range_high']} kg. Son estimaciones poblacionales y no un diagnóstico.")

def ensure_profile_targets(pid):
    """Backfill old profiles so the dashboard never renders 0/0 references."""
    p=get_profile(pid)
    if not p or p.get("special_state") in ("Embarazo","Lactancia"):
        return p
    try:
        needs=any(num(p.get(k))<=0 for k in ("calorie_target","protein_target","carbs_target","fat_target"))
        if needs:
            kcal,prot,carbs,fat=calculate_targets(p.get("sex_energy"),int(p.get("age") or 0),float(p.get("height_cm") or 0),float(p.get("weight_kg") or 0),p.get("activity"),p.get("goal"))
            con=db();con.execute("UPDATE profiles SET calorie_target=?,protein_target=?,carbs_target=?,fat_target=? WHERE id=?",(kcal,prot,carbs,fat,pid));con.commit();con.close()
            p=get_profile(pid)
    except Exception:
        pass
    return p

def personalized_plan_summary(p):
    e=energy_estimate(p)
    if not e.get("enabled"):
        return f"{p["name"]}, tu perfil está listo. {e.get("reason","Para este caso no fijaré una meta calórica automática.")}"
    return (f"{p["name"]}, tu punto de partida es de aproximadamente {e["maintenance"]} kcal para mantener tu gasto estimado. "
            f"Para tu objetivo de {p["goal"].lower()}, la referencia diaria queda en torno a {e["target"]} kcal, "
            f"con {e["protein_target"]} gramos de proteína, {e["carbs_target"]} gramos de carbohidratos y {e["fat_target"]} gramos de grasa. "
            f"Tu agua objetivo es {int(p.get("water_goal_ml") or 2000)} ml. Esto es una estimación basada en tus datos; tu gasto real puede variar.")

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
Patrón alimentario: {p.get('diet_style','Omnívora')}
Región del Perú: {p.get('region','Perú')}
Intolerancias: {p.get('intolerances','')}
Meta calórica manual (0 = automática): {p.get('calorie_target',0)}
Meta de proteína: {p.get('protein_target',0)} g/día
Meta de agua: {p.get('water_goal_ml',2000)} ml
Energía: {energy}
Indicadores: {metrics_explanation(health_metrics(p))}
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
Eres FitGlass, asistente nutricional educativo de NutriVision.

Si preguntan qué es FitGlass o quién la diseñó, responde:
"FitGlass es un asistente nutricional educativo. Fue creado por Requena Nuñez Andre, Atarama Portocarrero Sebastian, Zapata Mendoza Cesar Noe y Tima Garcia Alexander. Los encargados actuales son Andre Requena Nuñez y Sebastian Atarama Portocarrero."
No menciones proveedores técnicos, modelos o claves salvo que pregunten expresamente por la infraestructura. Mantén un tono elegante, sereno y preciso, sin emojis. Puedes hacer un chiste breve y limpio ocasionalmente. Si te preguntan si puedes hablar, responde que sí de manera natural. Puedes explicar IMC, metabolismo basal, gasto de mantenimiento, metas de macronutrientes y progreso a partir del perfil. No inventes cifras: distingue siempre entre estimaciones y mediciones.

SOLO HABLAS DE: alimentación, nutrición general, calorías, platos, porciones, recetas,
jugos, postres, frutas, verduras, etiquetas, códigos de barra, compras, preparación y hábitos alimentarios.
Si cambian de tema, responde brevemente que FitGlass se especializa en alimentación.

Usa siempre el perfil. Respeta alergias, gustos y alimentos evitados. Usa la región del Perú para contextualizar platos y disponibilidad, pero nunca conviertas un dato poblacional de una región en una predicción individual. No afirmes que ciertos alimentos son los más consumidos salvo que exista una fuente estadística; puedes tratarlos como referencias gastronómicas.
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

PROFILE_EDIT_PROMPT="""Convierte la instrucción del usuario en cambios de perfil. Devuelve SOLO JSON con un objeto updates. Si pide añadir algo, conserva lo existente y agrega el nuevo elemento; no borres información salvo petición explícita.
"""

def apply_profile_updates(pid,updates):
    p=get_profile(pid)
    if not p:return []
    allowed={"favorite_foods","favorite_fruits","favorite_vegetables","avoid_foods","allergies","intolerances","diet_style","region","notes","goal","activity","water_goal_ml","calorie_target","protein_target","carbs_target","fat_target","special_state"}
    changes={}
    for k,v in (updates or {}).items():
        if k not in allowed: continue
        if k in {"water_goal_ml","calorie_target","protein_target","carbs_target","fat_target"}:
            try:v=float(v)
            except:continue
        if k in {"favorite_foods","favorite_fruits","favorite_vegetables","avoid_foods","allergies","intolerances"}:
            oldv=str(p.get(k) or "").strip(); newv=str(v or "").strip(); parts=[x.strip() for x in oldv.split(",") if x.strip()]
            if newv and newv not in parts:parts.append(newv)
            v=", ".join(parts)
        if str(p.get(k,""))!=str(v):changes[k]=v
    if changes:
        merged=dict(p); merged.update(changes); update_profile(pid,merged)
    return sorted(changes)

def ai_edit_profile(p,text):
    data=ai_json(PROFILE_EDIT_PROMPT+"\nPERFIL ACTUAL:\n"+profile_context(p)+"\nINSTRUCCIÓN:\n"+text,max_tokens=1000)
    return apply_profile_updates(p["id"],data.get("updates",{}))

def coach_fallback(p,text):
    m=health_metrics(p)
    t=(text or "").lower()
    if any(k in t for k in ("imc","indice de masa","peso ideal")):
        return f"Puedo orientarte con tus indicadores. Tu IMC estimado es {m['bmi']} ({m['bmi_label']}). Tu rango de referencia por IMC 18,5–24,9 es aproximadamente {m['weight_range_low']}–{m['weight_range_high']} kg. Son estimaciones y no un diagnóstico."
    if any(k in t for k in ("calorias","caloría","proteína","macros","carbohidratos","grasas")):
        return f"Ahora mismo puedo darte una referencia con tu perfil: {m['calorie_target']} kcal, {m['protein_target']} g de proteína, {m['carbs_target']} g de carbohidratos y {m['fat_target']} g de grasa al día. Cuando el servicio vuelva a estar disponible, puedo afinar la recomendación."
    return "Puedo seguir ayudándote con alimentación, nutrición y tu perfil. En este momento no pude consultar al asistente, pero puedo darte una orientación basada en los datos que ya tienes guardados."

def ksc_chat(p,text):
    """Coach robusto: reintenta y cambia de modelo antes de caer al modo local."""
    key=ai_key()
    if not key:
        return coach_fallback(p,text)
    history=get_chat(p["id"],10)
    msgs=[{"role":"system","content":system_prompt(p)}]+history+[{
        "role":"user","content":str(text or "").strip()
    }]
    last_error=None
    for model in AI_CHAT_FALLBACK_MODELS:
        for attempt in range(2):
            try:
                r=ai_client(key).chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=.48,
                    top_p=.9,
                    max_completion_tokens=1400,
                    reasoning_effort="none",
                    stream=False,
                )
                ans=(r.choices[0].message.content or "").strip()
                if ans:
                    maybe_memory(p["id"],text)
                    return ans
                last_error=RuntimeError("Respuesta vacía del asistente")
            except RateLimitError as e:
                last_error=e
                time.sleep(0.35*(attempt+1))
                continue
            except (APIConnectionError, BadRequestError) as e:
                last_error=e
                time.sleep(0.25*(attempt+1))
                break
            except AuthenticationError as e:
                raise RuntimeError("Hubo un error.") from e
            except Exception as e:
                last_error=e
                time.sleep(0.2*(attempt+1))
                break
    # Nunca dejar el Coach en blanco.
    return coach_fallback(p,text)

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
{"dish_name_es":"...","summary":"...","total_estimated_grams":450,"foods":[{"name_es":"...","usda_query":"short generic English USDA query",
"estimated_grams":120,"confidence":90,"preparation":"..."}],"limitations":["..."]}
Máximo 6 alimentos. Los gramos son estimación visual de porción, no una báscula. Estima también el peso total del plato. Si no hay comida foods=[].
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
    return {"dish_name":str(d.get("dish_name_es","Plato no identificado"))[:120],"summary":str(d.get("summary",""))[:300],"total_estimated_grams":max(0,int(num(d.get("total_estimated_grams",0)))),"foods":foods,
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
    a.metric(" Energía",f"{v['kcal']:.0f} kcal");b.metric(" Proteína",f"{v['protein']:.1f} g")
    c.metric(" Carbohidratos",f"{v['carbs']:.1f} g");d.metric(" Grasas",f"{v['fat']:.1f} g")
    e,f,g,h=st.columns(4)
    e.metric(" Fibra",f"{v['fiber']:.1f} g");f.metric(" Azúcares",f"{v['sugars']:.1f} g")
    g.metric(" Sodio",f"{v['sodium_mg']:.0f} mg");h.metric(" Saturadas",f"{v['sat_fat']:.1f} g")

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
# VOZ IA — ELEVENLABS
# ============================================================

def eleven_key():
    return secret("ELEVENLABS_API_KEY", "")

def eleven_voice_id():
    return secret("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

@st.cache_data(ttl=1800, show_spinner=False)
def eleven_tts(text, voice_id):
    key=eleven_key()
    if not key:
        return None
    clean=re.sub(r"[#*_>`~]+", " ", text or "").strip()
    clean=re.sub(r"\s+", " ", clean)
    if not clean:
        return None
    clean=clean[:4500]
    url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    payload={
        "text":clean,
        "model_id":"eleven_multilingual_v2",
        "voice_settings":{
            "stability":0.38,
            "similarity_boost":0.84,
            "style":0.32,
            "use_speaker_boost":True
        }
    }
    r=requests.post(url,headers={"xi-api-key":key,"Content-Type":"application/json"},json=payload,timeout=60)
    r.raise_for_status()
    return r.content

def assistant_avatar_path():
    """Create a small SVG avatar for FitGlass without using emoji assets."""
    path=DATA_DIR/"fitglass_assistant_avatar.svg"
    if not path.exists():
        path.write_text("""<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop stop-color='#8fffd0'/><stop offset='1' stop-color='#9bb8ff'/></linearGradient></defs><rect width='160' height='160' rx='80' fill='#0b1512'/><circle cx='80' cy='80' r='59' fill='none' stroke='url(#g)' stroke-width='8'/><circle cx='58' cy='67' r='7' fill='#fff'/><circle cx='102' cy='67' r='7' fill='#fff'/><path d='M51 94 Q80 113 109 94' fill='none' stroke='url(#g)' stroke-width='7' stroke-linecap='round'/><path d='M80 21v20' stroke='#fff' stroke-width='6' stroke-linecap='round'/><circle cx='80' cy='16' r='7' fill='#8fffd0'/></svg>""",encoding='utf-8')
    return str(path)

def render_profile_avatar(p,size=46):
    path=(p or {}).get("photo_path","") if p else ""
    if path and Path(path).exists():
        b64=profile_photo_b64(path)
        return f"<img class='fg-chat-avatar user' src='{b64}' width='{size}' height='{size}' alt='Perfil'>"
    initial=html.escape(((p or {}).get("name") or "F")[:1].upper())
    return f"<div class='fg-chat-avatar user fallback'>{initial}</div>"

def render_assistant_avatar(size=46):
    b64=base64.b64encode(Path(assistant_avatar_path()).read_bytes()).decode()
    return f"<img class='fg-chat-avatar assistant' src='data:image/svg+xml;base64,{b64}' width='{size}' height='{size}' alt='FitGlass'>"

def render_ai_response(text, key="fitglass_ai", autoplay=True):
    """Prepara el TTS primero y sincroniza comienzo de audio + escritura del mensaje."""
    clean=str(text or "").strip()
    if not clean:
        return
    audio=None
    try:
        audio=eleven_tts(clean,eleven_voice_id()) if eleven_key() else None
    except Exception:
        audio=None
    if not audio:
        # Modo degradado: el Coach sigue funcionando aunque TTS esté temporalmente fuera de servicio.
        safe=html.escape(clean).replace(chr(10),'<br>')
        st.markdown(f"<div class='fg-ai-shell'><div class='fg-ai-meta'><span class='fg-ai-dot'></span><span>FitGlass</span></div><div class='fg-ai-text'>{safe}</div></div>",unsafe_allow_html=True)
        return
    b64=base64.b64encode(audio).decode()
    safe_key=re.sub(r"[^a-zA-Z0-9_]","_",key)
    payload=json.dumps(clean,ensure_ascii=False)
    components.html(f"""
<div class='fg-ai-shell' id='fgai_{safe_key}'>
  <div class='fg-ai-meta'>
    <span class='fg-ai-dot'></span><span>FitGlass</span>
    <button id='fgmute_{safe_key}' type='button'>Silenciar</button>
  </div>
  <div class='fg-ai-text' id='fgtext_{safe_key}'></div>
  <audio id='fgaudio_{safe_key}' preload='auto' autoplay></audio>
</div>
<script>
(function(){{
  const shell=document.getElementById('fgai_{safe_key}');
  const out=document.getElementById('fgtext_{safe_key}');
  const btn=document.getElementById('fgmute_{safe_key}');
  const audio=document.getElementById('fgaudio_{safe_key}');
  const text={payload};
  const src='data:audio/mp3;base64,{b64}';
  let started=false, muted=false;
  audio.src=src;
  audio.preload='auto';
  function startTogether(){{
    if(started)return;
    started=true;
    const words=text.split(/(\\s+)/); let i=0;
    function typeNext(){{
      if(i>=words.length)return;
      out.insertAdjacentText('beforeend',words[i]);
      const chunk=words[i++];
      setTimeout(typeNext, chunk.trim()?18:0);
    }}
    audio.muted=muted;
    const playPromise=audio.play();
    if(playPromise && playPromise.catch)playPromise.catch(()=>{{
      // La política del navegador puede bloquear autoplay. Una interacción posterior reanuda audio y texto juntos.
      started=false;
      const resume=()=>{{document.removeEventListener('pointerdown',resume);startTogether();}};
      document.addEventListener('pointerdown',resume,{{once:true}});
    }});
    typeNext();
  }}
  audio.addEventListener('canplaythrough',startTogether,{{once:true}});
  audio.addEventListener('loadeddata',()=>{{setTimeout(startTogether,60);}},{{once:true}});
  btn.onclick=()=>{{muted=!muted;audio.muted=muted;btn.textContent=muted?'Activar voz':'Silenciar';}};
  // Intento inmediato por si el navegador ya permite autoplay después de la interacción que generó el mensaje.
  setTimeout(startTogether,100);
}})();
</script>
<style>
.fg-ai-shell{{position:relative;font-family:Manrope,-apple-system,BlinkMacSystemFont,sans-serif;padding:16px 18px;border-radius:22px;border:1px solid rgba(255,255,255,.14);background:linear-gradient(135deg,rgba(255,255,255,.105),rgba(255,255,255,.032));backdrop-filter:blur(28px) saturate(180%);box-shadow:0 22px 70px rgba(0,0,0,.22);color:#edf9f2}}
.fg-ai-meta{{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:850;color:#b5cfc1}}
.fg-ai-meta button{{margin-left:auto;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.065);color:#fff;border-radius:999px;padding:6px 11px;font-weight:800;cursor:pointer}}
.fg-ai-dot{{width:7px;height:7px;border-radius:50%;background:#7effbd;box-shadow:0 0 16px rgba(126,255,189,.9);animation:fgdot 1.2s ease-in-out infinite}}
@keyframes fgdot{{0%,100%{{transform:scale(.86);opacity:.65}}50%{{transform:scale(1.15);opacity:1}}}}
.fg-ai-text{{margin-top:12px;font-size:14px;line-height:1.72;color:#f0faf4;white-space:pre-wrap;min-height:24px}}
</style>
""",height=max(110,min(500,130+len(clean)//2)))

def voice_reader_component(text, voice_hint="female", rate=1.0, autoplay=True, key="tts"):
    render_ai_response(text,key=key,autoplay=autoplay)

def voice_input_component(key="stt"):
    """Reliable Streamlit recorder + Groq Whisper transcription."""
    try:
        audio=st.audio_input("Habla con FitGlass",sample_rate=16000,key=key,label_visibility="collapsed")
    except Exception as e:
        st.error(f"El micrófono no está disponible en esta versión de Streamlit: {e}");return ""
    if not audio:
        st.caption("Pulsa el micrófono, habla y detén la grabación. FitGlass convertirá tu voz en texto automáticamente.");return ""
    key_groq=ai_key()
    if not key_groq:
        st.warning("Falta GROQ_API_KEY para transcribir el dictado.");return ""
    try:
        result=ai_client(key_groq).audio.transcriptions.create(file=("fitglass_dictado.wav",audio.getvalue()),model="whisper-large-v3-turbo",language="es",response_format="json",temperature=0.0,prompt="Español de Perú. Nutrición, alimentos, comidas y nombres de FitGlass.")
        transcript=str(getattr(result,"text","") or "").strip()
        if transcript:
            st.markdown(f"<div class='voice-transcript'><span>Transcripción</span>{html.escape(transcript)}</div>",unsafe_allow_html=True)
            return transcript
    except Exception as e:
        st.error(f"No pude transcribir el audio: {e}")
    return ""

# ============================================================
# FITGLASS INTELLIGENCE LAYER
# ============================================================
REGIONAL_FACTS={
    "Piura":25.9,"Cusco":22.3,"Huancavelica":15.6,"Huánuco":21.8,"Ica":34.8,"Junín":19.9,"La Libertad":29.7,"Lambayeque":29.3,"Lima":31.6,"Loreto":22.3,"Madre de Dios":36.1,"Moquegua":38.6,"Pasco":19.4,"Puno":23.9,"San Martín":23.6,"Tacna":41.0,"Tumbes":30.6,"Ucayali":25.0
}
REGION_FOODS={"Piura":["seco de chavelo","malarrabia","ceviche","cabrito","arroz con pato","chifles"],"Lambayeque":["arroz con pato","cabrito","king kong"],"Arequipa":["rocoto relleno","adobo","solterito"],"Cusco":["chiri uchu","pachamanca","cuy"],"Lima":["ceviche","pollo a la brasa","lomo saltado"]}

def regional_profile_note(p):
    region=p.get("region","Perú"); lines=[]
    if region in REGIONAL_FACTS: lines.append(f"ENDES 2024 reporta {REGIONAL_FACTS[region]:.1f}% de obesidad en personas de 15+ años para {region}. Es un dato poblacional y no predice tu resultado individual.")
    if region in REGION_FOODS: lines.append("Referencias gastronómicas regionales: "+", ".join(REGION_FOODS[region])+". Son ejemplos culturales, no un ranking estadístico.")
    if region=="Piura":lines.append("ENDES 2023 registró en Piura un promedio de 4.2 días por semana de consumo de frutas en personas de 15+ años.")
    return " ".join(lines) if lines else "No hay una cifra departamental cargada para esta región; FitGlass no inventa datos."

def browser_reminders(p):
    if not bool(int(p.get("reminders_enabled",1) or 0)):return
    water=water_today(p["id"]); goal=int(p.get("water_goal_ml") or 2000); total,_=day_totals(p["id"]); e=energy_estimate(p); target=float(e.get("target",0) or 0)
    msg="Recuerda beber agua. Todavía estás por debajo de la mitad de tu objetivo." if water<goal*.5 else ("Tu registro de energía va ligero. Revisa si todavía te falta una comida." if target and total["kcal"]<target*.45 and datetime.now().hour>=14 else "Tu seguimiento está en marcha. Una decisión pequeña y constante cuenta.")
    msg=json.dumps(msg,ensure_ascii=False)
    components.html(f"<script>(function(){{const send=()=>{{if(!('Notification'in window))return;if(Notification.permission==='default')Notification.requestPermission();if(Notification.permission==='granted')new Notification('FitGlass',{{body:{msg}}});}};setTimeout(send,1800);}})();</script>",height=1)

def render_streak_calendar(pid):
    today=date.today(); current=st.session_state.get("calendar_month",today.replace(day=1))
    if isinstance(current,str):
        try: current=date.fromisoformat(current)
        except Exception: current=today.replace(day=1)
    current=current.replace(day=1)
    matrix,meals_by_day,water_by_day=real_month_calendar_data(pid,current.year,current.month)
    month_name=["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"][current.month-1]
    st.markdown(f"<div class='glass-surface' style='padding:22px;margin-top:16px'><div class='kicker'>CALENDARIO REAL</div><div style='display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap'><div><div style='font-size:1.35rem;font-weight:900;color:#fff'>{month_name} {current.year}</div><div class='form-hint'>Hoy: {today.strftime('%d/%m/%Y')} · {['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][today.weekday()]}</div></div><div class='calendar-duration'>{len(usage_days(pid))} días usando FitGlass</div></div></div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    if a.button("‹",key=f"calprev_{pid}",use_container_width=True):
        st.session_state.calendar_month=(current-timedelta(days=1)).replace(day=1); st.rerun()
    if b.button("Hoy",key=f"caltoday_{pid}",use_container_width=True):
        st.session_state.calendar_month=today.replace(day=1); st.session_state.calendar_selected_date=today; st.rerun()
    if c.button("›",key=f"calnext_{pid}",use_container_width=True):
        st.session_state.calendar_month=(current+timedelta(days=32)).replace(day=1); st.rerun()
    st.markdown("<div class='fg-calendar-head'>"+"".join(f"<span>{x}</span>" for x in ['L','M','X','J','V','S','D'])+"</div>",unsafe_allow_html=True)
    selected=st.session_state.get("calendar_selected_date",today)
    if isinstance(selected,str):
        try: selected=date.fromisoformat(selected)
        except Exception: selected=today
    for week in matrix:
        cols=st.columns(7,gap="small")
        for i,daynum in enumerate(week):
            with cols[i]:
                if not daynum:
                    st.markdown("<div class='cal-empty'></div>",unsafe_allow_html=True); continue
                d=date(current.year,current.month,daynum)
                stt=streak_status(pid,d) if d<=today else {"perfect":False,"nutrition":False,"active":False}
                status="perfect" if stt["perfect"] else "nutrition" if stt["nutrition"] else "active" if stt["active"] else "empty"
                label=str(daynum)
                if d==today: label=f"{daynum} · hoy"
                if st.button(label,key=f"realcal_{pid}_{d.isoformat()}",disabled=d>today,use_container_width=True):
                    st.session_state.calendar_selected_date=d; st.rerun()
                rec=meals_by_day.get(d.isoformat(),{}); water=water_by_day.get(d.isoformat(),0)
                st.markdown(f"<div class='cal-cell {status}{' today' if d==today else ''}' title='{rec.get('count',0)} comidas · {rec.get('kcal',0):.0f} kcal · {water} ml'></div>",unsafe_allow_html=True)
    sst=streak_status(pid,selected) if selected<=today else {"active":False,"nutrition":False,"perfect":False,"water":0}
    totals,meals=day_totals(pid,selected); weekday=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][selected.weekday()]
    st.markdown(f"<div class='glass-surface calendar-day-detail'><div class='kicker'>{weekday.upper()}</div><div class='calendar-detail-title'>{selected.strftime('%d/%m/%Y')}</div><div class='calendar-detail-metrics'><span>{totals['kcal']:.0f} kcal</span><span>{totals['protein']:.1f} g proteína</span><span>{sst.get('water',0)} ml agua</span><span>{len(meals)} comidas</span></div><div class='form-hint'>Racha activa: {'sí' if sst['active'] else 'no'} · nutricional: {'sí' if sst['nutrition'] else 'no'} · perfecta: {'sí' if sst['perfect'] else 'no'}</div></div>",unsafe_allow_html=True)
    if meals:
        st.markdown("<div class='history-title'>Historial de comidas</div>",unsafe_allow_html=True)
        for m in meals:
            img=m.get("image_path"); c1,c2=st.columns([0.22,0.78])
            with c1:
                if img and Path(img).exists(): st.image(img,use_container_width=True)
                else: st.markdown("<div class='history-photo-placeholder'>Sin foto</div>",unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='history-meal'><div class='history-meal-title'>{html.escape(str(m.get('title') or 'Comida'))}</div><div class='history-meal-meta'>{html.escape(str(m.get('meal_type') or ''))} · {html.escape(str(m.get('meal_time') or ''))}</div><div class='history-meal-kcal'>{num(m.get('kcal')):.0f} kcal</div><div class='history-meal-macros'>Proteína {num(m.get('protein')):.1f} g · Carbohidratos {num(m.get('carbs')):.1f} g · Grasas {num(m.get('fat')):.1f} g · Fibra {num(m.get('fiber')):.1f} g</div></div>",unsafe_allow_html=True)
    else: st.markdown("<div class='note'>No hay comidas registradas en este día.</div>",unsafe_allow_html=True)

st.markdown('<style>.voice-glass{display:flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid rgba(255,255,255,.14);border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.10),rgba(255,255,255,.04));backdrop-filter:blur(24px) saturate(170%);margin-top:10px}.voice-glass button{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.07);color:#fff;border-radius:999px;padding:7px 12px;font-weight:800}.voice-glass span{color:#a7c7b7;font-size:.78rem}.streak-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.streak-metrics>div{padding:12px 14px;border:1px solid rgba(255,255,255,.10);border-radius:16px;background:rgba(255,255,255,.045)}.streak-metrics b{display:block;color:#fff;font-size:1.5rem}.streak-metrics span{color:#9eb8aa;font-size:.75rem}.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.fg-calendar-head{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin:12px 0 6px}.fg-calendar-head span{text-align:center;color:#77988a;font-weight:850;font-size:.72rem}.calendar-duration{font-size:.76rem;color:#9fb7aa}.calendar-day-detail{padding:18px;margin-top:12px}.calendar-detail-title{font-size:1.55rem;color:#fff;font-weight:900;margin-top:2px}.calendar-detail-metrics{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}.calendar-detail-metrics span{padding:8px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.09);color:#d8ebe2;font-size:.76rem;font-weight:800}.history-title{font-size:1rem;font-weight:900;color:#fff;margin:18px 0 10px}.history-meal{padding:14px 16px;border:1px solid rgba(255,255,255,.09);border-radius:18px;background:linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.028));backdrop-filter:blur(20px);margin-bottom:10px}.history-meal-title{font-weight:900;color:#fff;font-size:1rem}.history-meal-meta{color:#8fac9e;font-size:.76rem;margin-top:3px}.history-meal-kcal{color:#7effbd;font-size:1.15rem;font-weight:900;margin-top:9px}.history-meal-macros{color:#b8cec2;font-size:.74rem;margin-top:4px;line-height:1.5}.history-photo-placeholder{height:78px;border-radius:16px;display:grid;place-items:center;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#8aa798;font-size:.7rem}.fg-cal-state{height:3px;border-radius:99px;margin:-5px 8px 8px;box-shadow:0 0 12px currentColor}.fg-cal-state.perfect{background:#b697ff;color:#b697ff}.fg-cal-state.nutrition{background:#8eb6ff;color:#8eb6ff}.fg-cal-state.active{background:#7effbd;color:#7effbd}.fg-cal-state.empty{background:transparent;box-shadow:none}@media(max-width:760px){.fg-calendar-head{gap:3px}.fg-calendar-head span{font-size:.64rem}.calendar-detail-metrics{gap:6px}.calendar-detail-metrics span{font-size:.67rem;padding:7px 8px}}.cal-empty{height:42px}.cal-cell.today{box-shadow:0 0 0 2px rgba(255,255,255,.88),0 0 24px rgba(126,255,189,.24);transform:translateY(-1px)}.cal-cell{aspect-ratio:1;border-radius:12px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);display:grid;place-items:center;color:#aac4b4;font-weight:800}.cal-cell.active{background:rgba(110,255,188,.18);border-color:rgba(110,255,188,.30);color:#d9fff0}.cal-cell.nutrition{background:rgba(120,165,255,.20);border-color:rgba(120,165,255,.36);color:#e7efff}.cal-cell.perfect{background:linear-gradient(135deg,rgba(182,132,255,.30),rgba(110,255,188,.18));border-color:rgba(194,157,255,.48);color:#fff}.cal-cell span{font-size:.78rem}.region-note{padding:14px 16px;border-radius:18px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.03));color:#c5d9ce;line-height:1.55}.analysis-card{padding:24px;border-radius:24px;background:radial-gradient(circle at 50% 0,rgba(120,165,255,.18),transparent 55%),linear-gradient(135deg,rgba(255,255,255,.10),rgba(255,255,255,.035));border:1px solid rgba(255,255,255,.14);box-shadow:0 20px 70px rgba(0,0,0,.24);backdrop-filter:blur(28px) saturate(175%);text-align:center}.analysis-orb{width:86px;height:86px;border-radius:50%;margin:0 auto 16px;background:radial-gradient(circle at 35% 30%,#fff,rgba(129,255,200,.75) 18%,rgba(104,155,255,.28) 45%,transparent 70%);box-shadow:0 0 45px rgba(110,220,255,.28);animation:analysisPulse 1.7s ease-in-out infinite}.analysis-ring{width:124px;height:124px;border-radius:50%;margin:0 auto 16px;border:1px solid rgba(255,255,255,.15);box-shadow:inset 0 0 25px rgba(130,255,210,.15),0 0 45px rgba(120,170,255,.10);animation:analysisRotate 2.8s linear infinite}@keyframes analysisPulse{0%,100%{transform:scale(.92);opacity:.7}50%{transform:scale(1.08);opacity:1}}@keyframes analysisRotate{from{transform:rotate(0)}to{transform:rotate(360deg)}}</style>', unsafe_allow_html=True)

# ============================================================
# BIENVENIDA POST-PERFIL
# ============================================================
def maybe_show_profile_welcome():
    if st.session_state.get("pending_welcome") and not st.session_state.get("pending_welcome_shown"):
        st.session_state["pending_welcome_shown"]=True
        render_ai_response(st.session_state.pop("pending_welcome"),key="profile_welcome",autoplay=True)

# ============================================================
# UI
# ============================================================

def hero(p=None):
    extra=""
    if p:
        pts,lvl,_=level_info(p["id"]);r=streak(p["id"])
        extra=(
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;position:relative;z-index:1">'
            f'<span class="chip"> Hola, {p["name"]}</span>'
            f'<span class="chip">{lvl[2]} {lvl[1]}</span>'
            f'<span class="chip"> {pts} pts</span>'
            f'<span class="chip"> {r} día{"s" if r!=1 else ""} de racha</span>'
            '</div>'
        )
    st.markdown(
        '<div class="hero">'
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
    pid=st.session_state.get("pid")
    return get_profile(pid) if pid else None


def need_profile(p):
    if not p:
        st.markdown('<div class="lock"><h3> Desbloquea un perfil</h3><div class="note">Selecciona o crea un perfil en la barra lateral para continuar.</div></div>',unsafe_allow_html=True)
        st.stop()

# ============================================================
# NAVEGACIÓN — SOLO 3 SECCIONES
# ============================================================

def svg_icon(name, size=24, stroke="currentColor"):
    icons={
        "leaf":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4C11 4 5 8 4 18c5-5 10-5 16-5-3 3-6 5-10 6"/><path d="M4 18c1-5 4-9 9-12"/></svg>',
        "home":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m3 10 9-7 9 7"/><path d="M5 9v11h14V9"/><path d="M9 20v-6h6v6"/></svg>',
        "spark":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 1.4 5.3L19 9l-5.6 1.7L12 16l-1.4-5.3L5 9l5.6-1.7L12 2Z"/><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15Z"/></svg>',
        "calendar":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="17" rx="3"/><path d="M8 2v4M16 2v4M3 9h18"/></svg>',
        "camera":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h4l2-2h4l2 2h4v12H4z"/><circle cx="12" cy="13" r="4"/></svg>',
        "chart":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5M4 19h17"/><path d="M8 16v-3M12 16V8M16 16v-6M20 16V4"/></svg>',
        "user":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c.8-4 3.5-6 8-6s7.2 2 8 6"/></svg>',
    }
    return icons.get(name,icons["spark"])

def render_topbar(profile):
    avatar=(profile.get("name", "K")[0].upper() if profile else "K")
    left=f'<div class="brandmark">{svg_icon("leaf",30,"#9affd0")}<div><div style="font-size:1.08rem">FitGlass</div><div style="font-size:.68rem;color:#98b4a6;font-weight:700;letter-spacing:.08em">NUTRIVISION</div></div></div>'
    right=f'<div style="display:flex;align-items:center;gap:10px"><div style="width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(135deg,rgba(130,255,200,.28),rgba(120,160,255,.28));border:1px solid rgba(255,255,255,.18);font-weight:900">{avatar}</div></div>'
    st.markdown(f'<div class="glass-surface topbar">{left}{right}</div>',unsafe_allow_html=True)

def onboarding():
    """First-run experience: one question at a time, with animated Liquid Glass UI."""
    defaults = {
        "name":"", "age":30, "sex":"Prefiero no indicar", "height":170.0, "weight":70.0,
        "activity":list(ACTIVITY_FACTORS)[1], "goal":"Mantener", "diet":"Omnívora",
        "allergies":"", "intolerances":"", "avoid":"", "favorites":"", "special":"Ninguna",
        "water_goal":2200, "calorie_target":0, "protein_target":0, "pin":"", "region":"Piura"
    }
    ob = st.session_state.setdefault("onboarding", defaults.copy())
    st.session_state.setdefault("onboarding_step", 0)
    step = st.session_state["onboarding_step"]
    total = 7

    questions = [
        ("Tu nombre", "Vamos a personalizar todo para ti.", "Escribe cómo quieres que te llame."),
        ("Datos de referencia", "Necesitamos una base para estimar tu energía.", "Edad, sexo, talla y peso permiten una estimación inicial."),
        ("Tu movimiento diario", "Ahora cuéntame cómo es tu día normal.", "No necesitas ser exacto. Elige lo que más se parezca a tu rutina."),
        ("Tu objetivo", "¿Qué quieres conseguir?", "Esto ajustará la orientación de calorías y nutrientes."),
        ("Cómo comes", "Cuéntame cómo sueles alimentarte.", "Tus preferencias se usarán al analizar platos y crear recetas."),
        ("Lo que debemos cuidar", "Hay datos que tu asistente debe respetar.", "Puedes dejar cualquier campo vacío si no aplica."),
        ("Tu punto de partida", "Un último ajuste y entramos.", "Definiremos agua, objetivos opcionales y acceso al perfil."),
    ]
    title, subtitle, hint = questions[step]
    progress = int(((step + 1) / total) * 100)

    particle_html="".join([f"<i class='fg-particle p{i}'></i>" for i in range(16)])
    st.markdown(f"""
    <div class="onboarding-stage question-stage stage-{step}">
      <div class="onboarding-orb orb-a"></div><div class="onboarding-orb orb-b"></div>
      <div class="fg-particles">{particle_html}</div>
      <div class="onboarding-shell">
        <div class="onboarding-top">
          <div class="brandmark">{svg_icon('leaf',28,'#d7fff0')}<div><div class="ob-brand">FitGlass</div><div class="ob-kicker">PERSONALIZACIÓN</div></div></div>
          <div class="ob-step">{step+1} / {total}</div>
        </div>
        <div class="ob-progress"><div style="width:{progress}%"></div></div>
        <div class="ob-copy">
          <div class="ob-eyebrow">{subtitle}</div>
          <div class="onboarding-title">{title}</div>
          <div class="form-hint ob-hint">{hint}</div>
        </div>
    </div></div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ob-card-wrap'>", unsafe_allow_html=True)
    with st.form(f"onboarding_step_{step}", clear_on_submit=False):
        if step == 0:
            ob["name"] = st.text_input("¿Cómo quieres que te llame?", value=ob["name"], placeholder="Tu nombre", label_visibility="visible")
            st.markdown("<div class='ob-tip'>Tu nombre solo se usa para personalizar la experiencia.</div>", unsafe_allow_html=True)

        elif step == 1:
            a,b = st.columns(2)
            with a:
                st.markdown("<div class='control-label'>Edad</div>", unsafe_allow_html=True)
                age_slider=st.slider("Edad", 10, 100, int(ob["age"]), 1, key="ob_age_slider", label_visibility="collapsed")
                age_direct=st.number_input("Edad exacta", 10, 100, int(age_slider), 1, key="ob_age_direct", label_visibility="collapsed")
                ob["age"]=int(age_direct)
                st.markdown("<div class='control-hint'>Desliza o escribe directamente.</div>", unsafe_allow_html=True)
                ob["height"] = st.number_input("Talla (cm)", 120.0, 230.0, float(ob["height"]), 0.5)
            with b:
                ob["sex"] = st.selectbox("Referencia fisiológica", ["Prefiero no indicar","Masculino","Femenino"], index=["Prefiero no indicar","Masculino","Femenino"].index(ob["sex"]))
                ob["weight"] = st.number_input("Peso (kg)", 30.0, 250.0, float(ob["weight"]), 0.1)
            regions=list(dict.fromkeys(list(REGIONAL_FACTS.keys())+["Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Callao","Huancavelica","Pasco","San Martín","Tumbes","Ucayali"]))
            ob["region"]=st.selectbox("¿De qué parte del Perú eres?",regions,index=regions.index(ob.get("region","Piura")) if ob.get("region","Piura") in regions else 0)

        elif step == 2:
            activity_options = list(ACTIVITY_FACTORS)
            ob["activity"] = st.radio("¿Cómo es tu actividad habitual?", activity_options, index=activity_options.index(ob["activity"]), label_visibility="visible")

        elif step == 3:
            goals=["Mantener","Perder peso","Ganar masa muscular","Ganar peso"]
            ob["goal"] = st.radio("¿Cuál es tu objetivo principal?", goals, index=goals.index(ob["goal"]), horizontal=False)
            if ob["goal"] == "Perder peso":
                st.markdown("<div class='ob-soft-note'>La app calculará una orientación moderada. No sustituye la evaluación de un profesional.</div>", unsafe_allow_html=True)
            elif ob["goal"] == "Ganar masa muscular":
                st.markdown("<div class='ob-soft-note'>Priorizaremos proteína, energía suficiente y constancia.</div>", unsafe_allow_html=True)

        elif step == 4:
            diets=["Omnívora","Vegetariana","Vegana","Pescetariana","Baja en carbohidratos","Otra"]
            ob["diet"] = st.selectbox("Patrón alimentario", diets, index=diets.index(ob["diet"]))
            ob["favorites"] = st.text_input("Alimentos que disfrutas", value=ob["favorites"], placeholder="Ej.: pollo, arroz, yogur")

        elif step == 5:
            ob["allergies"] = st.text_input("Alergias", value=ob["allergies"], placeholder="Ej.: maní, mariscos")
            ob["intolerances"] = st.text_input("Intolerancias", value=ob["intolerances"], placeholder="Ej.: lactosa, gluten")
            ob["avoid"] = st.text_input("Alimentos que prefieres evitar", value=ob["avoid"], placeholder="Separados por comas")
            special=["Ninguna","Embarazo","Lactancia"]
            ob["special"] = st.selectbox("Situación especial", special, index=special.index(ob["special"]))

        elif step == 6:
            ob["water_goal"] = st.slider("Meta de agua diaria (ml)", 1000, 5000, int(ob["water_goal"]), 100)
            c1,c2 = st.columns(2)
            with c1:
                ob["calorie_target"] = st.number_input("Calorías/día", 0, 6000, int(ob["calorie_target"]), 50, help="0 = cálculo automático")
            with c2:
                ob["protein_target"] = st.number_input("Proteína g/día", 0.0, 400.0, float(ob["protein_target"]), 1.0, help="0 = cálculo automático")
            ob["pin"] = st.text_input("PIN opcional", value=ob["pin"], type="password", max_chars=4, placeholder="4 dígitos")
            st.markdown("<div class='ob-finish'>Tu perfil se guardará en este dispositivo y podrás editarlo después.</div>", unsafe_allow_html=True)

        st.markdown("<div class='ob-actions-spacer'></div>", unsafe_allow_html=True)
        back_col, _, next_col = st.columns([1.2, 3.2, 1.8])
        with back_col:
            back = st.form_submit_button("Atrás", disabled=(step == 0), use_container_width=True)
        with next_col:
            label = "Entrar a FitGlass" if step == total - 1 else "Continuar"
            forward = st.form_submit_button(label, type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if back and step > 0:
        st.session_state["onboarding_step"] = step - 1
        st.rerun()
    if forward:
        if step == 0 and not ob["name"].strip():
            st.error("Escribe tu nombre para continuar.")
            return
        if step == total - 1:
            if ob["pin"] and not re.fullmatch(r"\d{4}", ob["pin"]):
                st.error("El PIN debe tener exactamente 4 dígitos.")
                return
            kcal, prot, carbs, fat = calculate_targets(ob["sex"], int(ob["age"]), float(ob["height"]), float(ob["weight"]), ob["activity"], ob["goal"])
            if ob["calorie_target"] <= 0:
                ob["calorie_target"] = kcal
            if ob["protein_target"] <= 0:
                ob["protein_target"] = prot
            ob["carbs_target"] = carbs
            ob["fat_target"] = fat
            payload={
                "name":ob["name"].strip(), "age":int(ob["age"]), "sex_energy":ob["sex"],
                "height_cm":float(ob["height"]), "weight_kg":float(ob["weight"]), "activity":ob["activity"],
                "goal":ob["goal"], "favorite_foods":ob["favorites"], "favorite_fruits":"", "favorite_vegetables":"",
                "avoid_foods":ob["avoid"], "allergies":ob["allergies"], "special_state":ob["special"],
                "photo_path":"", "pin_hash":hash_pin(ob["pin"]) if ob["pin"] else "", "water_goal_ml":int(ob["water_goal"]),
                "diet_style":ob["diet"], "intolerances":ob["intolerances"], "region":ob.get("region","Piura"), "notes":"", "reminders_enabled":1, "calorie_target":float(ob["calorie_target"]),
                "protein_target":float(ob["protein_target"]), "carbs_target":float(carbs), "fat_target":float(fat)
            }
            pid=create_profile(payload)
            add_points(pid,20,"Perfil creado")
            st.session_state["pid"]=pid
            st.session_state[f"unlocked_{pid}"]=True
            st.session_state["welcome_summary"]=personalized_plan_summary(get_profile(pid))
            st.session_state.pop("onboarding", None)
            st.session_state.pop("onboarding_step", None)
            st.rerun()
        else:
            st.session_state["onboarding_step"] = step + 1
            st.rerun()


profiles=list_profiles()
if not profiles:
    onboarding()
    st.stop()

pid=st.session_state.get("pid")
if pid not in [p["id"] for p in profiles]: pid=profiles[0]["id"];st.session_state["pid"]=pid
profile=ensure_profile_targets(pid) or get_profile(pid)
maybe_show_profile_welcome()
render_topbar(profile)
if st.session_state.get("welcome_summary"):
    welcome=st.session_state.pop("welcome_summary")
    st.markdown(f'<div class="glass-surface" style="padding:24px;margin-bottom:14px"><div class="kicker">TU PLAN DE PARTIDA</div><div style="font-size:1.35rem;font-weight:900;color:#fff">Perfil listo</div><div class="form-hint" style="margin:8px 0 0">{welcome}</div></div>',unsafe_allow_html=True)
    render_ai_response(welcome,autoplay=True,key="welcome_tts")

c1,c2,c3=st.columns(3)
with c1:
    if st.button("Inicio",use_container_width=True,key="main_inicio"):
        st.session_state["main_section"]="Inicio";st.rerun()
with c2:
    if st.button("Hoy",use_container_width=True,key="main_hoy"):
        st.session_state["main_section"]="Hoy";st.rerun()
with c3:
    if st.button("Coach",use_container_width=True,key="main_coach"):
        st.session_state["main_section"]="Coach";st.rerun()
main_section=st.session_state.get("main_section","Inicio")

# ============================================================
# INICIO — DASHBOARD
# ============================================================
if main_section=="Inicio":
    browser_reminders(profile)
    rem_col1,rem_col2=st.columns([1,3])
    with rem_col1:
        reminders_on=st.toggle("Recordatorios",value=bool(int(profile.get("reminders_enabled",1) or 0)),key="fg_reminders")
    if reminders_on != bool(int(profile.get("reminders_enabled",1) or 0)):
        update_profile(profile["id"],{**profile,"reminders_enabled":int(reminders_on)})
        profile=get_profile(profile["id"])
    total,meals=day_totals(profile["id"]);water=water_today(profile["id"]);goal=int(profile.get("water_goal_ml") or 2000)
    energy=energy_estimate(profile);pts,lvl,nxt=level_info(profile["id"]);racha=streak(profile["id"])
    weekday_es={"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"}.get(datetime.now().strftime("%A"),datetime.now().strftime("%A"))
    target=float(energy.get("target",0) or 0); kcal_pct=min(100,(total["kcal"]/target*100 if target else 0))
    protein_target=float(energy.get("protein_target",0) or 0); protein_pct=min(100,total["protein"]/protein_target*100 if protein_target else 0)
    water_pct=min(100,water/goal*100 if goal else 0)
    st.markdown(f'<div class="glass-surface" style="padding:30px;margin-top:14px"><div class="form-hint" style="margin:0;color:#89f4bc;font-weight:850;letter-spacing:.1em">HOY</div><div style="font-size:clamp(2rem,5vw,3.6rem);font-weight:950;letter-spacing:-.06em;color:#fff;margin-top:7px">Hola, {profile["name"].split()[0]}.</div><div style="color:#9fb7aa;max-width:760px;margin-top:8px">{weekday_es}, {datetime.now().strftime("%d/%m/%Y")}. Tu panel se adapta a tus metas y a lo que ya registraste. Nada de ruido: solo lo que necesitas para decidir tu siguiente comida.</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    r1,r2,r3=st.columns(3)
    rings=[("Calorías",kcal_pct,f'{total["kcal"]:.0f}/{target:.0f} kcal','#83ffbe'),("Proteína",protein_pct,f'{total["protein"]:.0f}/{protein_target:.0f} g','#83b7ff'),("Agua",water_pct,f'{water}/{goal} ml','#b798ff')]
    for col,(lab,pct,val,ring) in zip((r1,r2,r3),rings):
        with col:
            st.markdown(f'<div class="glass-surface" style="padding:24px;text-align:center"><div class="metric-ring" style="--pct:{pct:.1f};--ring:{ring};margin:auto"><div><div class="ring-value">{pct:.0f}%</div><div class="ring-label">{lab}</div></div></div><div style="font-weight:800;margin-top:14px;color:#fff">{val}</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    left,right=st.columns([1.4,1])
    with left:
        st.markdown('<div class="glass-surface" style="padding:24px"><div style="font-size:1.2rem;font-weight:900;color:#fff">Distribución nutricional</div><div class="form-hint">Seguimiento de lo registrado hoy frente a tus referencias diarias.</div>',unsafe_allow_html=True)
        carb_target=float(energy.get("carbs_target",0) or 0);fat_target=float(energy.get("fat_target",0) or 0)
        rows=[("Proteína",total["protein"],protein_target),("Carbohidratos",total["carbs"],carb_target),("Grasas",total["fat"],fat_target),("Fibra",total["fiber"],30)]
        st.markdown('<div class="bars">'+''.join([f'<div class="bar-row"><span style="color:#c9dcd1;font-weight:750">{a}</span><div class="bar"><i style="width:{min(100,(b/c*100 if c else 0)):.0f}%"></i></div><b style="color:#fff;text-align:right">{b:.0f} g</b></div>' for a,b,c in rows])+'</div></div>',unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="glass-surface" style="padding:24px"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-size:1.2rem;font-weight:900;color:#fff">Racha</div><div class="form-hint">Actividad registrada</div></div><div style="font-size:2.2rem;font-weight:950;color:#fff">{racha}</div></div><div style="height:10px"></div><div style="height:8px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden"><div style="height:100%;width:{min(100,racha/7*100):.0f}%;background:linear-gradient(90deg,#7effbd,#8eb2ff);border-radius:99px"></div></div><div class="form-hint" style="margin-top:12px">{lvl[1]} · {pts} puntos</div></div>',unsafe_allow_html=True)
    if meals:
        st.markdown("### Actividad reciente")
        df=pd.DataFrame(meals)
        st.dataframe(df[["meal_time","meal_type","title","kcal","protein","fiber"]],hide_index=True,use_container_width=True)
    else:
        st.markdown('<div class="glass-surface" style="padding:22px;margin-top:16px"><b style="color:#fff">Tu día todavía está vacío.</b><div class="form-hint">Abre Hoy para tomar una foto y empezar tu registro.</div></div>',unsafe_allow_html=True)
    if profile.get("special_state") in ("Embarazo","Lactancia"):
        st.markdown(f'<div class="region-note"><b style="color:#fff">Modo especial</b><div style="margin-top:6px">{profile.get("special_state")} requiere una orientación distinta. FitGlass evita metas agresivas y recomienda validación profesional.</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="region-note" style="margin-top:18px"><b style="color:#fff">Contexto de {profile.get("region","Perú")}</b><div style="margin-top:6px">{regional_profile_note(profile)}</div></div>',unsafe_allow_html=True)
    render_streak_calendar(profile["id"])
    st.stop()

# ============================================================
# HOY — ACCIÓN PRINCIPAL + DIARIO
# ============================================================
if main_section=="Hoy":
    need_profile(profile)
    section("HOY","Registra lo que vas a comer","Foto, análisis y guardado en pocos pasos.")
    t1,t2=st.tabs(["Tomar foto","Historial de hoy"])
    with t1:
        src=st.radio("Fuente",["Cámara","Subir imagen"],horizontal=True,key="today_src")
        f=st.camera_input("Toma una foto") if src=="Cámara" else st.file_uploader("Selecciona una foto",type=["jpg","jpeg","png"],key="today_upload")
        if f:
            jpeg=compact_jpeg(f.getvalue());st.image(jpeg,width=480)
            if st.button("Analizar comida",type="primary",use_container_width=True,key="today_analyze"):
                try:
                    ph=st.empty(); ph.markdown("<div class=\"analysis-card\"><div class=\"analysis-ring\"><div class=\"analysis-orb\"></div></div><div style=\"font-size:1.15rem;font-weight:900;color:#fff\">Analizando tu comida</div><div class=\"form-hint\">Reconociendo plato, porciones y peso estimado.</div></div>",unsafe_allow_html=True); time.sleep(.25); st.session_state["mealres"]=detect_foods(ai_key(),jpeg); st.session_state["mealjpeg"]=jpeg; ph.empty()
                except Exception as e:st.error(f"No pude analizar la imagen: {e}")
            res=st.session_state.get("mealres")
            if res and res.get("foods"):
                st.markdown(f'<div class="glass-surface" style="padding:18px 20px;margin:14px 0"><div class="kicker">PLATO DETECTADO</div><div style="font-size:1.55rem;font-weight:950;color:#fff">{res.get("dish_name","Plato no identificado")}</div><div class="form-hint">Peso estimado del plato: <b style="color:#fff">{res.get("total_estimated_grams",0)} g</b>. Es una estimación visual, no una báscula.</div></div>',unsafe_allow_html=True)
                calc=enrich(res,"today")
                if calc:
                    tot=total_nutrition(calc);show_metrics(tot)
                    mt,title,note=st.columns([.8,1.2,1])
                    with mt: meal_type=st.selectbox("Momento",["Desayuno","Media mañana","Almuerzo","Merienda","Cena","Otro"],key="today_type")
                    with title: meal_title=st.text_input("Nombre de la comida",res.get("summary","Comida")[:80],key="today_title")
                    with note: meal_note=st.text_input("Nota",key="today_note")
                    if st.button("Guardar en Hoy",type="primary",use_container_width=True,key="today_save"):
                        ip=save_jpeg(st.session_state["mealjpeg"],MEAL_DIR,f"meal_{profile['id']}")
                        add_meal(profile["id"],meal_type,meal_title,calc,tot,ip,meal_note);st.success("Comida guardada. Tu racha se ha actualizado.");st.rerun()
            elif res is not None: st.info("No detecté alimentos claros. Prueba con mejor luz y encuadre.")
    with t2:
        total,meals=day_totals(profile["id"]);water=water_today(profile["id"])
        st.markdown(f'<div class="glass-surface" style="padding:20px"><div style="font-weight:900;color:#fff">Resumen de hoy</div><div style="margin-top:8px;color:#a6bbae">{total["kcal"]:.0f} kcal · {total["protein"]:.1f} g proteína · {water} ml agua</div></div>',unsafe_allow_html=True)
        if meals:st.dataframe(pd.DataFrame(meals)[["meal_time","meal_type","title","kcal","protein","fiber"]],hide_index=True,use_container_width=True)
    st.stop()

# ============================================================
# COACH — HERRAMIENTAS EXISTENTES ORGANIZADAS
# ============================================================
section("COACH","Tu asistente","FitGlass, recetas, escáneres, planes, progreso y cuenta en un solo lugar.")
coach_group=st.selectbox("Área",["IA y nutrición","Comida y escáneres","Plan y hábitos","Progreso y cuenta"],label_visibility="collapsed",key="coach_group")
coach_options={
    "IA y nutrición":[("Asistente IA"," Chat por voz con FitGlass"),("Comparar platos"," Comparar platos")],
    "Comida y escáneres":[("Diario de comidas"," Diario de comidas"),("Código de barras"," Escáner de código de barras"),("Etiqueta nutricional"," Escáner de etiqueta"),("Cocina inteligente"," Cocina inteligente")],
    "Plan y hábitos":[("Plan semanal"," Plan semanal"),("Agua y hábitos"," Agua & hábitos"),("Recompensas"," Recompensas KSC"),("Actividad"," Arena de Push-Ups")],
    "Progreso y cuenta":[("Progreso"," Mi progreso"),("Academia"," Academia FitGlass (quiz)"),("Comunidad"," Comunidad"),("Mi perfil"," Mi perfil"),("Configuración"," Configuración")]
}
coach_items=coach_options[coach_group]
labels=[x[0] for x in coach_items]
sel=st.selectbox("Herramienta",labels,label_visibility="collapsed",key="coach_tool")
page=dict(coach_items)[sel]

hero(profile)
sync_community()

# ============================================================
# INICIO
# ============================================================

if page==" Inicio":
    section("HOY","Panel principal","Calorías, agua, puntos, racha y qué comer ahora — todo de un vistazo.")
    if not profile:
        st.info("Crea o desbloquea un perfil en la barra lateral para comenzar.")
        st.markdown("###  Empieza rápido")
        c1,c2,c3=st.columns(3)
        c1.markdown('<div class="mini"><div class="kicker">PASO 1</div><div class="big"> Crea tu perfil</div><div class="note">Ve a "Mi perfil" y regístrate con foto, gustos y PIN.</div></div>',unsafe_allow_html=True)
        c2.markdown('<div class="mini"><div class="kicker">PASO 2</div><div class="big"> Registra tu comida</div><div class="note">Sube una foto y deja que FitGlass calcule los nutrientes.</div></div>',unsafe_allow_html=True)
        c3.markdown('<div class="mini"><div class="kicker">PASO 3</div><div class="big"> Gana puntos</div><div class="note">Cada acción saludable suma puntos y sube tu nivel.</div></div>',unsafe_allow_html=True)
    else:
        total,meals=day_totals(profile["id"]);water=water_today(profile["id"]);goal=int(profile.get("water_goal_ml") or 2000)
        pts,lvl,nxt=level_info(profile["id"]);racha=streak(profile["id"])
        if water==0:st.markdown('<div class="water-alert"><b> Falta registrar agua hoy.</b> Ve a "Agua & hábitos" para anotar tus vasos.</div>',unsafe_allow_html=True)

        wpct=min(100,round(water/goal*100)) if goal else 0
        st.markdown(textwrap.dedent(f"""
        <div class="stat-grid">
          <div class="stat-card accent-orange" style="animation-delay:.02s">
            <span class="icon"></span>
            <div class="label">Calorías hoy</div>
            <div class="value">{total['kcal']:.0f}</div>
            <div class="sub">kcal registradas</div>
          </div>
          <div class="stat-card accent-blue" style="animation-delay:.08s">
            <span class="icon"></span>
            <div class="label">Hidratación</div>
            <div class="value">{water} <span style="font-size:.95rem;color:var(--muted);font-weight:700">/ {goal} ml</span></div>
            <div class="sub">{wpct}% de tu meta</div>
          </div>
          <div class="stat-card accent-purple" style="animation-delay:.14s">
            <span class="icon"></span>
            <div class="label">Puntos KSC</div>
            <div class="value">{pts}</div>
            <div class="sub">Nivel {lvl[2]} {lvl[1]}</div>
          </div>
          <div class="stat-card accent-green" style="animation-delay:.2s">
            <div class="streak-card" style="border:none;background:none;padding:0;gap:10px">
              <span class="streak-flame"></span>
              <div>
                <div class="streak-days">{racha}</div>
                <div class="streak-label">día{'s' if racha!=1 else ''} de racha</div>
              </div>
            </div>
          </div>
        </div>
        """),unsafe_allow_html=True)
        hm=health_metrics(profile)
        st.markdown(f"""
        <div class='glass-surface fg-health-panel'>
          <div class='fg-health-head'><div><div class='kicker'>LECTURA DEL PERFIL</div><div class='fg-health-title'>Tu mapa nutricional</div></div><div class='fg-health-badge'>{hm['bmi_label']}</div></div>
          <div class='fg-health-grid'>
            <div><span>IMC</span><b>{hm['bmi']:.1f}</b><small>{hm['bmi_label']}</small></div>
            <div><span>Metabolismo basal</span><b>{hm['bmr'] or 0}</b><small>kcal/día estimadas</small></div>
            <div><span>Mantenimiento</span><b>{hm['maintenance'] or 0}</b><small>kcal/día estimadas</small></div>
            <div><span>Objetivo diario</span><b>{hm['calorie_target'] or 0}</b><small>kcal/día</small></div>
          </div>
          <div class='fg-health-foot'>Referencia de peso por IMC 18,5–24,9: <b>{hm['weight_range_low']}–{hm['weight_range_high']} kg</b>. FitGlass usa estimaciones y no sustituye una evaluación clínica.</div>
        </div>
        """,unsafe_allow_html=True)
        show_metrics(total)
        st.markdown("###  ¿Qué puedo comer ahora?")
        hour=datetime.now().hour
        moment="desayuno" if hour<10 else "media mañana" if hour<12 else "almuerzo" if hour<16 else "merienda" if hour<19 else "cena"
        if st.button(f"Recomiéndame {moment}",type="primary",use_container_width=True):
            prompt=f"Es {moment}. Hoy llevo {total['kcal']:.0f} kcal, {total['protein']:.1f} g proteína, {total['fiber']:.1f} g fibra y {water} ml de agua. Dame 3 opciones para mi perfil y mis gustos."
            try:
                ans=ksc_chat(profile,prompt)
                st.markdown(ans)
                voice_reader_component(ans, autoplay=True, key="home_tts")
            except RuntimeError as e:st.error(str(e))
        if meals:st.dataframe(pd.DataFrame(meals)[["meal_time","meal_type","title","kcal","protein","fiber"]],hide_index=True,use_container_width=True)

# ============================================================
# MI PERFIL
# ============================================================

elif page==" Mi perfil":
    section("USUARIOS","Mi perfil","Nombre, foto, talla, peso, gustos, alergias, objetivo y agua. Se guarda para siempre en la base local.")
    t1,t2=st.tabs([" Crear perfil nuevo","Editar mi perfil"])
    with t1:
        with st.form("newp"):
            c1,c2=st.columns(2)
            with c1:
                name=st.text_input("Nombre");age=st.number_input("Edad",10,100,16,1)
                sex=st.selectbox("Variable fisiológica",["Prefiero no indicar","Masculino","Femenino"])
                height=st.number_input("Talla cm",120.,230.,165.,.5);weight=st.number_input("Peso kg",30.,250.,60.,.1)
                activity=st.selectbox("Actividad",list(ACTIVITY_FACTORS),index=1)
                goal=st.selectbox("Objetivo",["Mantener","Ganar masa muscular","Ganar peso","Perder peso"])
                region=st.selectbox("Región del Perú",list(dict.fromkeys(list(REGIONAL_FACTS.keys())+["Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Callao"])),index=0)
                water_goal=st.number_input("Meta de agua ml/día",500,5000,2000,100)
            with c2:
                photo=st.file_uploader("Foto",type=["jpg","jpeg","png"],key="np_photo")
                fav=st.text_area("Comidas favoritas");fr=st.text_area("Frutas favoritas");veg=st.text_area("Verduras favoritas")
                avoid=st.text_area("No me gusta / evito");allerg=st.text_area("Alergias/restricciones")
                special=st.selectbox("Estado especial",["Ninguno","Embarazo","Lactancia"])
                diet=st.selectbox("Patrón alimentario",["Omnívora","Vegetariana","Vegana","Pescetariana","Baja en carbohidratos","Otra"])
                intolerances=st.text_area("Intolerancias")
                pin=st.text_input("PIN 4 dígitos",type="password",max_chars=4);pin2=st.text_input("Repite PIN",type="password",max_chars=4)
            save=st.form_submit_button("Crear perfil",type="primary",use_container_width=True)
        if save:
            if not name.strip():st.error("Escribe nombre.")
            elif not re.fullmatch(r"\d{4}",pin or "") or pin!=pin2:st.error("PIN inválido o no coincide.")
            else:
                pp=save_jpeg(photo.getvalue(),PROFILE_DIR,"profile",700) if photo else ""
                tkcal,tprot,tcarb,tfat=calculate_targets(sex,int(age),float(height),float(weight),activity,goal)
                pid=create_profile({"name":name.strip(),"age":int(age),"sex_energy":sex,"height_cm":height,"weight_kg":weight,
                    "activity":activity,"goal":goal,"favorite_foods":fav,"favorite_fruits":fr,"favorite_vegetables":veg,
                    "avoid_foods":avoid,"allergies":allerg,"special_state":special,"photo_path":pp,
                    "pin_hash":hash_pin(pin),"water_goal_ml":int(water_goal),"diet_style":diet,"intolerances":intolerances,"region":region,"notes":"","reminders_enabled":1,"calorie_target":tkcal,"protein_target":tprot,"carbs_target":tcarb,"fat_target":tfat})
                add_points(pid,20,"Perfil creado");st.session_state["pid"]=pid;st.session_state[f"unlocked_{pid}"]=True
                st.session_state["pending_welcome"]=personalized_plan_summary(get_profile(pid))+" También puedo explicarte tu IMC y cómo interpretar estas cifras."
                st.success("Perfil creado y guardado permanentemente. ");st.rerun()

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
                    diet=st.selectbox("Patrón alimentario",["Omnívora","Vegetariana","Vegana","Pescetariana","Baja en carbohidratos","Otra"],index=["Omnívora","Vegetariana","Vegana","Pescetariana","Baja en carbohidratos","Otra"].index(profile.get("diet_style","Omnívora")) if profile.get("diet_style","Omnívora") in ["Omnívora","Vegetariana","Vegana","Pescetariana","Baja en carbohidratos","Otra"] else 0)
                    intolerances=st.text_area("Intolerancias",profile.get("intolerances",""))
                save=st.form_submit_button("Guardar cambios",type="primary",use_container_width=True)
            if save:
                pp=profile.get("photo_path","")
                if photo:pp=save_jpeg(photo.getvalue(),PROFILE_DIR,f"profile_{profile['id']}",700)
                update_profile(profile["id"],{"name":name,"age":int(age),"sex_energy":sex,"height_cm":height,"weight_kg":weight,
                    "activity":activity,"goal":goal,"favorite_foods":fav,"favorite_fruits":fr,"favorite_vegetables":veg,
                    "avoid_foods":avoid,"allergies":allerg,"special_state":special,"photo_path":pp,"water_goal_ml":int(water_goal),"diet_style":diet,"intolerances":intolerances,"region":region,"notes":profile.get("notes",""),"reminders_enabled":int(profile.get("reminders_enabled",1) or 0),"calorie_target":float(profile.get("calorie_target") or 0),"protein_target":float(profile.get("protein_target") or 0),"carbs_target":float(profile.get("carbs_target") or 0),"fat_target":float(profile.get("fat_target") or 0)})
                st.success("Guardado.");st.rerun()
            st.markdown("---")
            with st.expander("Eliminar este perfil (irreversible)"):
                st.warning("Esto borra el perfil y todo su historial de forma permanente.")
                confirm=st.text_input("Escribe ELIMINAR para confirmar",key="del_confirm")
                if st.button("Eliminar perfil definitivamente") and confirm=="ELIMINAR":
                    delete_profile(profile["id"]);st.session_state.pop("pid",None);st.rerun()

# ============================================================
# COMUNIDAD (ver otros perfiles, retar, mensajes locales)
# ============================================================

elif page==" Comunidad":
    need_profile(profile)
    section("COMUNIDAD","Otros usuarios de FitGlass","Perfiles creados en esta app (sin contraseñas). Puedes retarlos o escribirles.")
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
                if b1.button("Retar",key=f"chal_{c['id']}",use_container_width=True):
                    create_challenge(profile["id"],c["id"]);st.success("Reto de push-ups enviado")
                if b2.button(" Ver chat",key=f"openmsg_{c['id']}",use_container_width=True):
                    st.session_state["dm_target"]=c["id"]

    target=st.session_state.get("dm_target")
    if target:
        tp=get_profile(target)
        if tp:
            st.markdown(f"###  Mensajes con {tp['name']}")
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

    st.markdown("###  Retos pendientes de otros")
    cs=challenges(profile["id"])
    incoming=[c for c in cs if c["opponent_id"]==profile["id"] and c["status"]=="pending"]
    if not incoming:
        st.caption("No tienes retos pendientes.")
    for c in incoming:
        a,b=st.columns([.8,.2]);a.write(f" {c['challenger_name']} te reta a push-ups")
        if b.button("Aceptar",key=f"acc_comm_{c['id']}"):accept_challenge(c["id"],profile["id"]);st.rerun()

# ============================================================
# DIARIO DE COMIDAS
# ============================================================

elif page==" Diario de comidas":
    need_profile(profile);section("DIARIO","Diario fotográfico de comidas","Foto -> FitGlass -> nutrientes -> guardar.")
    ta,tb=st.tabs([" Nueva comida","Historial"])
    with ta:
        src=st.radio("Fuente",["Subir foto","Cámara"],horizontal=True)
        f=st.file_uploader("Imagen",type=["jpg","jpeg","png"],key="mealfile") if src=="Subir foto" else st.camera_input("Toma una foto")
        if f:
            jpeg=compact_jpeg(f.getvalue());st.image(jpeg,width=380)
            if st.button(" Analizar",type="primary"):
                try:
                    st.session_state["mealres"]=detect_foods(ai_key(),jpeg);st.session_state["mealjpeg"]=jpeg
                except Exception as e:st.error(f"No pude analizar la foto: {e}")
            res=st.session_state.get("mealres")
            if res and res.get("foods"):
                st.write(res.get("summary",""));calc=enrich(res,"meal")
                if calc:
                    tot=total_nutrition(calc);st.markdown("## Total del plato");show_metrics(tot)
                    if profile.get("allergies"):st.warning("Alergias/restricciones del perfil: "+profile["allergies"])
                    if st.button(" Analizar para mi perfil"):
                        try:
                            ans=ksc_chat(profile,f"Analiza este plato para mí: {[(x['name'],x['grams']) for x in calc]}. Totales {tot}.")
                            render_ai_response(ans,autoplay=True,key="meal_tts")
                        except Exception as e:st.error(str(e))
                    c1,c2=st.columns(2);mt=c1.selectbox("Momento",["Desayuno","Media mañana","Almuerzo","Merienda","Cena","Otro"]);title=c2.text_input("Nombre",res.get("summary","Mi comida")[:80])
                    note=st.text_input("Nota")
                    if st.button("Guardar en diario",type="primary",use_container_width=True):
                        ip=save_jpeg(st.session_state["mealjpeg"],MEAL_DIR,f"meal_{profile['id']}")
                        add_meal(profile["id"],mt,title,calc,tot,ip,note);st.success("+15 puntos guardados")
                    pred=", ".join(x["name"] for x in calc);actual=st.text_input("Realmente había",pred);correct=st.radio("¿Acertó?",["Sí","No"],horizontal=True)=="Sí"
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

elif page==" Escáner de código de barras":
    need_profile(profile)
    section("CÓDIGO DE BARRAS","Escáner de productos","Escribe el número o sube una foto legible; FitGlass lo interpreta con una base pública de productos.")
    t1,t2=st.tabs(["Número manual"," Foto del código"])
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
            if st.button(" ¿Me conviene este producto?",type="primary"):
                try:
                    ans=ksc_chat(profile,f"Analiza este producto escaneado para mi perfil: {json.dumps(info,ensure_ascii=False)}")
                    render_ai_response(ans,autoplay=True,key="bc_tts")
                except Exception as e:st.error(str(e))

# ============================================================
# ETIQUETA
# ============================================================

elif page==" Escáner de etiqueta":
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
        for w in d.get("warnings",[]):st.warning(w)

# ============================================================
# COMPARAR PLATOS
# ============================================================

elif page==" Comparar platos":
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
            if st.button(" ¿Cuál encaja mejor conmigo?"):
                try:
                    ans=ksc_chat(profile,f"Compara estos platos para mi perfil. A={totals[0]}, B={totals[1]}. Explica contexto y alternativa.")
                    render_ai_response(ans,autoplay=True,key="cmp_tts")
                except Exception as e:st.error(str(e))

# ============================================================
# CHAT POR VOZ
# ============================================================

elif page==" Chat por voz con FitGlass":
    need_profile(profile)
    section("CHAT","Habla con FitGlass","Escribe o habla por micrófono; FitGlass te puede responder también con voz. Solo alimentación, recetas y nutrición.")

    st.markdown("####  Dictado por voz (gratis, funciona en Chrome)")
    st.markdown("### Ajustes por conversación")
    edit_req=st.text_input("Dile a FitGlass qué quieres cambiar",placeholder="Agrega zanahoria a mis verduras favoritas")
    if st.button("Actualizar mi perfil con FitGlass",key="ai_profile_edit",type="primary") and edit_req.strip():
        try:
            changed=ai_edit_profile(profile,edit_req)
            if changed: st.success("Perfil actualizado: "+", ".join(changed)); st.rerun()
            else: st.info("No detecté un cambio concreto que aplicar.")
        except Exception as e: st.error(f"No pude actualizar el perfil: {e}")

    hm=health_metrics(profile)
    with st.expander("Mi resumen nutricional",expanded=False):
        a,b,c,d=st.columns(4)
        a.metric("IMC",hm['bmi'] or "—")
        b.metric("Basal",f"{hm['bmr']} kcal" if hm['bmr'] else "—")
        c.metric("Mantenimiento",f"{hm['maintenance']} kcal" if hm['maintenance'] else "—")
        d.metric("Proteína",f"{hm['protein_target']} g" if hm['protein_target'] else "—")
        st.caption(metrics_explanation(hm))

    dictated_prompt=voice_input_component(key="chat_stt")

    for m in get_chat(profile["id"],30):
        st.markdown(f"<div class='fg-chat-row {'assistant' if m['role']=='assistant' else 'user'}'>{render_assistant_avatar() if m['role']=='assistant' else render_profile_avatar(profile)}<div class='fg-chat-bubble'><div class='fg-chat-name'>{'FitGlass' if m['role']=='assistant' else html.escape(profile['name'])}</div><div>{m['content']}</div></div></div>",unsafe_allow_html=True)
    csave1, csave2 = st.columns(2)
    if csave1.button("Guardar chat",use_container_width=True,key="save_chat_now"):
        st.session_state["chat_saved_at"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.success("Chat guardado en tu perfil.")
    if csave2.button("Borrar chat",use_container_width=True,key="clear_chat_now"):
        con=db();con.execute("DELETE FROM chat_messages WHERE profile_id=?",(profile["id"],));con.commit();con.close();st.rerun()
    prompt=st.chat_input("Pregunta sobre alimentación...")
    if not prompt and dictated_prompt:
        prompt=dictated_prompt
    if prompt:
        st.markdown(f"<div class='fg-chat-row user'>{render_profile_avatar(profile)}<div class='fg-chat-bubble'><div class='fg-chat-name'>{html.escape(profile['name'])}</div><div>{html.escape(prompt)}</div></div></div>",unsafe_allow_html=True)
        try:
            ans=ksc_chat(profile,prompt)
        except Exception:
            ans="Hubo un error. Puedo seguir ayudándote con una alternativa basada en la información de tu perfil."
        add_chat(profile["id"],"user",prompt);add_chat(profile["id"],"assistant",ans)
        st.markdown(f"<div class='fg-chat-row assistant'>{render_assistant_avatar()}<div class='fg-chat-bubble'><div class='fg-chat-name'>FitGlass</div></div></div>",unsafe_allow_html=True)
        render_ai_response(f"Tu pregunta fue: {prompt}. {ans}", autoplay=True, key=f"chat_tts_{len(get_chat(profile['id']))}")
    st.markdown("###  Memoria alimentaria")
    for m in memories(profile["id"]):st.write("•",m)

# ============================================================
# COCINA
# ============================================================

elif page==" Cocina inteligente":
    need_profile(profile);section("CHEF KSC","Cocina inteligente","Recetas, refrigeradora, presupuesto, Perú, jugos, postres, favoritos y sustituciones.")
    tabs=st.tabs(["Recetas"," Tengo esto"," Presupuesto"," Perú","Favoritos"])
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
            if c1.button("Guardar"):save_favorite(profile["id"],cat+" KSC",st.session_state["recipe"],cat);st.success("Guardada")
            replacement=c2.text_input("Ingrediente a sustituir",key="sub")
            if c2.button(" Sustituir") and replacement:
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
            if st.button(" Detectar ingredientes de la foto",type="primary"):
                try:
                    with st.spinner("Analizando tu foto..."):
                        d=ai_json(FRIDGE_PROMPT,compact_jpeg(pic.getvalue()))
                    st.session_state["fridge"]=d.get("ingredients",[])
                    if not st.session_state["fridge"]:
                        st.warning("No pude identificar ingredientes claros en esta foto. Prueba con más luz o escribe manualmente abajo.")
                except Exception as e:
                    st.error(f"No pude analizar la foto: {e}")
        detected=st.session_state.get("fridge",[])
        if detected:st.write("Detectados en la foto:",", ".join(detected))
        if st.button(" Crear recetas con esto",type="primary",use_container_width=True):
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
            with st.expander(""+f["title"]):
                st.markdown(f["recipe"]);rating=st.slider("Puntuación",1,5,4,key=f"rt_{f['id']}");comment=st.text_input("Comentario",key=f"cm_{f['id']}")
                if st.button("Guardar valoración",key=f"sv_{f['id']}"):save_rating(profile["id"],f["id"],f["title"],rating,comment);st.success("Listo")

# ============================================================
# PLAN
# ============================================================

elif page==" Plan semanal":
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
        st.markdown("###  Lista de compras")
        for x in plan.get("shopping_list",[]):st.write("•",x)
        c1,c2=st.columns(2)
        if c1.button("Guardar plan"):save_plan(profile["id"],week,plan);st.success("Guardado")
        html="<html><body><h1>Plan semanal FitGlass</h1>"+pd.DataFrame(days).to_html(index=False)+"<h2>Compras</h2><ul>"+"".join(f"<li>{x}</li>" for x in plan.get("shopping_list",[]))+"</ul></body></html>"
        c2.download_button("Exportar HTML",html.encode(),file_name="plan_IA_KSC.html",mime="text/html",use_container_width=True)

# ============================================================
# AGUA & HÁBITOS
# ============================================================

elif page==" Agua & hábitos":
    need_profile(profile);section("HÁBITOS","Agua y retos diarios","Registra tu hidratación en vasos y completa hábitos saludables.")
    water=water_today(profile["id"]);goal=int(profile.get("water_goal_ml") or 2000);pct=min(100,round(water/goal*100) if goal else 0)
    st.markdown(f'<div class="water-alert" style="display:flex;justify-content:space-between;align-items:center"><div><div class="kicker">HIDRATACIÓN DE HOY</div><div class="big"> {water} <span style="color:var(--muted);font-size:1rem">/ {goal} ml</span></div></div><div style="font-size:1.6rem;font-weight:950;color:var(--blue)">{pct}%</div></div>',unsafe_allow_html=True)
    st.progress(min(1.,water/goal if goal else 0))
    st.markdown("#### Toma agua y regístrala")
    cc=st.columns(len(WATER_UNITS))
    for col,(label,ml,icon) in zip(cc,WATER_UNITS):
        if col.button(f"{icon}\n{label}",use_container_width=True,key=f"w_{ml}"):log_water(profile["id"],ml);st.rerun()
    with st.expander("Cantidad personalizada"):
        custom_ml=st.number_input("ml",50,3000,250,50)
        if st.button("Registrar cantidad personalizada"):log_water(profile["id"],custom_ml);st.rerun()
    st.markdown("###  Retos de hoy")
    goals=daily_goals(profile["id"]);done_n=sum(1 for g in goals if g["completed"])
    st.progress(done_n/len(goals) if goals else 0);st.caption(f"{done_n}/{len(goals)} retos completados hoy")
    for g in goals:
        with st.container(border=True):
            a,b=st.columns([.8,.2]);a.write(("" if g["completed"] else "")+g["goal_name"])
            if not g["completed"] and b.button("Completar",key=f"goal_{g['id']}"):complete_goal(g["id"],profile["id"]);st.rerun()

# ============================================================
# RECOMPENSAS KSC (juego)
# ============================================================

elif page==" Recompensas KSC":
    need_profile(profile);section("JUEGO","Puntos, niveles y desbloqueos","Completa hábitos, recetas, quizzes y retos para subir de nivel.")
    pts,lvl,nxt=level_info(profile["id"]);r=streak(profile["id"])
    lc,rc=st.columns([2,1])
    with lc:
        st.markdown(f'<div class="level-card"><div class="kicker">NIVEL ACTUAL</div><div class="big" style="font-size:2rem">{lvl[2]} {lvl[1]}</div><div class="note">{pts} puntos acumulados</div></div>',unsafe_allow_html=True)
    with rc:
        st.markdown(f'<div class="streak-card"><span class="streak-flame"></span><div><div class="streak-days">{r}</div><div class="streak-label">día{"s" if r!=1 else ""} seguidos</div></div></div>',unsafe_allow_html=True)
    if nxt:st.progress((pts-lvl[0])/(nxt[0]-lvl[0]));st.caption(f"Faltan {nxt[0]-pts} puntos para {nxt[2]} {nxt[1]}")
    st.markdown("###  Desbloqueos")
    cards=""
    for need,name in [(100," Laboratorio de batidos"),(250," Chef de postres"),(500," Planificador Maestro"),(900," Leyenda"),(1500," Elite")]:
        icon,label=name.split(" ",1);opened=pts>=need
        cards+=f'<div class="unlock-card {"open" if opened else "closed"}"><span class="uicon">{icon if opened else ""}</span><div class="uname">{label}</div><div class="ureq">{need} puntos</div></div>'
    st.markdown(f'<div class="unlock-grid">{cards}</div>',unsafe_allow_html=True)
    st.markdown("###  Cómo ganar puntos")
    st.markdown(textwrap.dedent("""
    -  Registrar una comida: **+15**
    -  Registrar agua: **+3**
    -  Completar un reto diario: **+10**
    -  Registrar peso: **+5**
    -  Registrar medidas: **+5**
    - Guardar receta favorita: **+5**
    - Calificar receta: **+3**
    -  Generar plan semanal: **+20**
    -  Cada push-up en la Arena: **+1** (mínimo 10 por intento)
    -  Cada respuesta correcta del quiz: **según nivel**
    """))
    st.markdown("###  Ranking general")
    rank=leaderboard()
    medals={0:"",1:"",2:""}
    rows=""
    for i,row in rank.iterrows():
        cls=f"top{i+1}" if i<3 else ""
        pos=medals.get(i,f"#{i+1}")
        rows+=f'<div class="rank-row {cls}"><div class="rank-pos">{pos}</div><div class="rank-name">{row["name"]}</div><div class="rank-val">{int(row["points"])} pts</div></div>'
    st.markdown(rows if rows else '<div class="note">Aún no hay puntajes.</div>',unsafe_allow_html=True)

# ============================================================
# ARENA DE PUSH-UPS
# ============================================================

elif page==" Arena de Push-Ups":
    need_profile(profile);section("PUSH-UP","Retos de 60 segundos entre perfiles","Reta a otro usuario; el conteo se hace con un esqueleto y anillo de progreso en vivo.")
    others=[p for p in list_profiles() if p["id"]!=profile["id"]]
    if others:
        mp={p["name"]:p["id"] for p in others};op=st.selectbox("Retar a",list(mp))
        if st.button("Enviar reto",use_container_width=True):create_challenge(profile["id"],mp[op]);st.success("Reto enviado")
    else:st.info("Crea otro perfil para competir (ver 'Comunidad').")
    cs=challenges(profile["id"])
    incoming=[c for c in cs if c["opponent_id"]==profile["id"] and c["status"]=="pending"]
    for c in incoming:
        a,b=st.columns([.8,.2]);a.write(f" {c['challenger_name']} te reta")
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
                st.markdown('<div class="winner-banner"><div class="wtitle"> Empate</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="winner-banner"><span class="streak-flame" style="font-size:2rem"></span><div class="wtitle">{order[0]["name"]}</div><div class="note">{order[0]["reps"]} push-ups en el minuto</div></div>',unsafe_allow_html=True)
            for a in ats:
                if a.get("video_path") and Path(a["video_path"]).exists():st.video(a["video_path"])
    st.markdown("###  Ranking Push-Up")
    pr=pushup_ranking();medals={0:"",1:"",2:""};rows=""
    for i,row in pr.iterrows():
        cls=f"top{i+1}" if i<3 else "";pos=medals.get(i,f"#{i+1}")
        rows+=f'<div class="rank-row {cls}"><div class="rank-pos">{pos}</div><div class="rank-name">{row["name"]}</div><div class="rank-val">{int(row["mejor_marca"])} reps</div></div>'
    st.markdown(rows if rows else '<div class="note">Sin intentos aún.</div>',unsafe_allow_html=True)

# ============================================================
# PROGRESO
# ============================================================

elif page==" Mi progreso":
    need_profile(profile);section("PROGRESO","Peso y medidas","7, 30 o 90 días; tendencias, no diagnósticos.")
    t1,t2=st.tabs([" Peso"," Medidas"])
    with t1:
        e=energy_estimate(profile);logs=weights(profile["id"])
        delta=None
        if len(logs)>=2:delta=logs[-1]["weight_kg"]-logs[0]["weight_kg"]
        d1,d2=st.columns(2)
        with d1:
            arrow="▲" if delta and delta>0 else "▼" if delta and delta<0 else "▬"
            dcolor="var(--orange)" if delta and delta>0 else "var(--green)" if delta and delta<0 else "var(--muted)"
            sub=f'<span style="color:{dcolor}">{arrow} {abs(delta):.1f} kg en el periodo</span>' if delta is not None else "Sin variación registrada aún"
            st.markdown(f'<div class="stat-card accent-green"><span class="icon"></span><div class="label">Peso actual</div><div class="value">{profile["weight_kg"]:.1f} kg</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)
        with d2:
            if e.get("enabled"):
                st.markdown(f'<div class="stat-card accent-blue"><span class="icon"></span><div class="label">Mantenimiento</div><div class="value">{e["maintenance"]} kcal</div><div class="sub">rango {e["target_low"]}–{e["target_high"]} kcal/día</div></div>',unsafe_allow_html=True)
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
# ACADEMIA FITGLASS (QUIZ CON NIVELES)
# ============================================================

elif page==" Academia FitGlass (quiz)":
    section("EDUCACIÓN","Academia FitGlass","Aprende sobre nutrición, sube de nivel y gana puntos extra.")

    QUIZ_LEVELS = {
        " Básico": {
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
        " Intermedio": {
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
        " Avanzado": {
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
            st.markdown("###  Tu historial de quiz")
            st.dataframe(pd.DataFrame([dict(r) for r in hist]),hide_index=True,use_container_width=True)

# ============================================================
# CONFIG
# ============================================================
# ============================================================
# CONFIG
# ============================================================

elif page==" Configuración":
    section("SISTEMA","Configuración","Claves, módulos y estado de la app.")
    ok=bool(ai_key())
    st.markdown(textwrap.dedent(f"""
    <div class="stat-card {'accent-green' if ok else 'accent-orange'}" style="max-width:420px">
      <span class="icon">{'' if ok else ''}</span>
      <div class="label">Estado de FitGlass</div>
      <div class="value" style="font-size:1.1rem">{'GROQ_API_KEY encontrada' if ok else 'Falta GROQ_API_KEY'}</div>
    </div>"""),unsafe_allow_html=True)
    st.code('GROQ_API_KEY = "TU_TOKEN"\n# opcional:\nUSDA_API_KEY = "TU_CLAVE_USDA"',language="toml")
    if st.button("Probar IA",type="primary"):
        try:
            ids={m.id for m in ai_client(ai_key()).models.list().data}
            st.success("FitGlass lista." if AI_MODEL in ids else "Conexión OK, modelo no visible.")
        except Exception as e:st.error(str(e))
    st.markdown("###  Arena Push-Up (cámara con esqueleto)")
    st.code("pip install streamlit-webrtc mediapipe av opencv-python-headless",language="powershell")
    st.markdown("###  Voz del asistente")
    st.caption("FitGlass puede leer sus respuestas automáticamente cuando la voz esté disponible.")
    st.markdown("###  Código de barras")
    st.caption("Usa la base pública y gratuita Open Food Facts. Los productos consultados se guardan en caché local (.ksc_data/barcode_cache.json) para funcionar más rápido la próxima vez.")
    st.markdown("###  Persistencia de datos")
    st.info("Todos los perfiles, comidas, puntos y retos se guardan en .ksc_data/ dentro del servidor donde corre la app. No se borran al cerrar el navegador. Si despliegas en un hosting con almacenamiento temporal, monta un volumen persistente en esa carpeta.")
    st.info("FitGlass es un asistente nutricional educativo. Encargados actuales: Requena Núñez Juan Carlos, André y Atarama Sebastián. Créditos del proyecto: César y Alexander.")

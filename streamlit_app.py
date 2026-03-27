"""
Medical Knowledge RAG — Streamlit frontend v3
Improvements:
  - Citations sorted by accuracy (highest first)
  - Sidebar toggle button always visible
  - Professional UI with glassmorphism header, smooth cards
  - Live animated search status with step-by-step progress text
"""
import streamlit as st
import requests, uuid, json, time, re
from datetime import datetime
from pathlib import Path

API_BASE     = "http://localhost:8000/api/v1"
HISTORY_FILE = Path("qa_history.json")

st.set_page_config(
    page_title="MedRAG — Health Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp{background:#0f172a}
#MainMenu,footer,header{visibility:hidden}
section[data-testid="stSidebar"]{background:#111827;border-right:1px solid #1e293b}
section[data-testid="stSidebar"] *{color:#e2e8f0 !important}
section[data-testid="stSidebar"] .stSlider *{color:#e2e8f0 !important}
section[data-testid="stSidebar"] hr{border-color:#1e293b}
div[data-testid="stSidebarCollapseButton"]{display:none}

/* ── Sidebar toggle button ── */
.sidebar-toggle{
  position:fixed;top:14px;left:14px;z-index:9999;
  background:#1e40af;color:white !important;border:none;border-radius:8px;
  padding:8px 13px;font-size:16px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.4);line-height:1;transition:background .2s
}
.sidebar-toggle:hover{background:#2563eb}

/* ── Header ── */
.top-header{
  background:linear-gradient(135deg,#1e3a8a 0%,#1e40af 50%,#1d4ed8 100%);
  color:white;padding:24px 32px;border-radius:16px;margin-bottom:24px;
  border:1px solid rgba(255,255,255,.1);
  box-shadow:0 4px 24px rgba(0,0,0,.3)
}
.top-header h1{margin:0;font-size:26px;color:white;letter-spacing:-.5px}
.top-header p{margin:6px 0 0;font-size:13px;color:#93c5fd;line-height:1.6}
.top-header .badge{display:inline-block;background:rgba(255,255,255,.15);
  color:white;font-size:11px;padding:2px 10px;border-radius:20px;margin:6px 4px 0 0}

/* ── Chat bubbles ── */
.user-bubble{
  background:linear-gradient(135deg,#1d4ed8,#3b82f6);
  color:white;padding:13px 18px;border-radius:20px 20px 4px 20px;
  margin:12px 0 12px auto;max-width:74%;font-size:15px;line-height:1.65;
  width:fit-content;box-shadow:0 2px 8px rgba(59,130,246,.35)
}
.bot-bubble{
  background:#1e293b;color:#e2e8f0;padding:18px 22px;
  border-radius:4px 20px 20px 20px;margin:12px 0;max-width:92%;
  border:1px solid #334155;font-size:15px;line-height:1.8;
  box-shadow:0 2px 12px rgba(0,0,0,.2)
}

/* ── Search progress ── */
.search-progress{
  background:#1e293b;border:1px solid #334155;border-radius:12px;
  padding:16px 20px;margin:12px 0
}
.search-step{
  display:flex;align-items:center;gap:10px;padding:6px 0;
  font-size:14px;color:#94a3b8;transition:color .3s
}
.search-step.active{color:#60a5fa;font-weight:600}
.search-step.done{color:#34d399}
.search-step .dot{
  width:8px;height:8px;border-radius:50%;background:#334155;flex-shrink:0
}
.search-step.active .dot{background:#3b82f6;animation:pulse-dot 1s infinite}
.search-step.done .dot{background:#10b981}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.search-step .icon{font-size:16px;width:20px;text-align:center}

/* ── Accuracy bar ── */
.acc-wrap{margin:12px 0 16px;background:#1e293b;border-radius:10px;padding:12px 16px;
  border:1px solid #334155}
.acc-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.acc-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8}
.acc-score{font-size:20px;font-weight:800}
.acc-bar-bg{background:#0f172a;border-radius:4px;height:10px;overflow:hidden}
.acc-bar-fill{height:10px;border-radius:4px;transition:width .8s ease}
.acc-sublabel{font-size:11px;color:#64748b;margin-top:5px}

/* ── Pills ── */
.pill{display:inline-block;padding:4px 12px;border-radius:20px;
  font-size:11px;font-weight:700;margin:2px 3px 2px 0;letter-spacing:.2px}
.p-green{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.p-yellow{background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.3)}
.p-red{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)}
.p-orange{background:rgba(249,115,22,.15);color:#fb923c;border:1px solid rgba(249,115,22,.3)}
.p-blue{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}
.p-gray{background:rgba(148,163,184,.1);color:#94a3b8;border:1px solid rgba(148,163,184,.2)}

/* ── Citation cards ── */
.cit-header{font-size:13px;font-weight:700;color:#94a3b8;
  text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px}
.cit-rank{font-size:10px;font-weight:800;color:white;
  width:20px;height:20px;border-radius:50%;display:inline-flex;
  align-items:center;justify-content:center;margin-right:6px}
.rank-1{background:#f59e0b}
.rank-2{background:#94a3b8}
.rank-3{background:#cd7c2e}
.rank-n{background:#334155}
.cit-card{
  background:#1e293b;border:1px solid #334155;border-radius:12px;
  padding:14px 16px;margin:6px 0;transition:border-color .2s;border-left:4px solid
}
.cit-card:hover{border-color:#3b82f6}
.cit-who{border-left-color:#0ea5e9}
.cit-pubmed{border-left-color:#a78bfa}
.cit-medline{border-left-color:#34d399}
.cit-fda{border-left-color:#fbbf24}
.cit-local{border-left-color:#64748b}
.cit-title{font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:3px}
.cit-org{font-size:11px;color:#64748b}
.cit-excerpt{font-size:12px;color:#94a3b8;font-style:italic;
  margin-top:8px;padding-top:8px;border-top:1px solid #334155;line-height:1.6}
.src-link{
  display:inline-flex;align-items:center;gap:6px;margin-top:10px;
  padding:5px 14px;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.35);
  border-radius:6px;color:#60a5fa;font-size:12px;font-weight:600;
  text-decoration:none;transition:all .2s
}
.src-link:hover{background:rgba(59,130,246,.25);border-color:#60a5fa;color:#93c5fd}
.badge-pill{font-size:10px;font-weight:700;background:rgba(255,255,255,.07);
  color:#94a3b8;padding:2px 8px;border-radius:8px;margin-right:6px}

/* ── Proof / warning boxes ── */
.proof-box{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);
  border-radius:10px;padding:12px 16px;font-size:13px;color:#34d399;margin:8px 0}
.proof-box strong{color:#6ee7b7}
.warn-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);
  border-radius:10px;padding:10px 14px;font-size:13px;color:#fbbf24;margin:6px 0}
.unsafe-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);
  border-radius:10px;padding:10px 14px;font-size:13px;color:#f87171;margin:6px 0}
.web-banner{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);
  border-radius:10px;padding:10px 14px;font-size:13px;color:#60a5fa;margin:6px 0}
.no-ctx{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);
  border-radius:12px;padding:16px 20px;font-size:15px;color:#fbbf24;margin:10px 0}

/* ── Stat cards ── */
.stat-card{background:#1e293b;border-radius:12px;padding:16px 18px;
  border:1px solid #334155;text-align:center}
.stat-val{font-size:28px;font-weight:800;color:#60a5fa}
.stat-lbl{font-size:11px;color:#64748b;margin-top:3px;text-transform:uppercase;letter-spacing:.4px}

/* ── History ── */
.hist-card{background:#1e293b;border:1px solid #334155;border-radius:12px;
  padding:14px 18px;margin:8px 0;transition:border-color .2s}
.hist-card:hover{border-color:#3b82f6}
.hist-q{font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:5px}
.hist-meta{font-size:11px;color:#64748b}
.hist-snippet{font-size:13px;color:#94a3b8;margin-top:8px;line-height:1.55}

/* ── Empty state ── */
.empty-state{text-align:center;padding:80px 20px;color:#475569}

/* ── Hospital cards ── */
.hosp-section{margin:16px 0}
.hosp-header{font-size:13px;font-weight:700;color:#94a3b8;text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.hosp-card{background:#1e293b;border:1px solid #334155;border-radius:12px;
  padding:14px 16px;margin:8px 0;transition:border-color .2s}
.hosp-card:hover{border-color:#3b82f6}
.hosp-card.emergency{border-left:4px solid #ef4444}
.hosp-card.specialist{border-left:4px solid #8b5cf6}
.hosp-card.general{border-left:4px solid #10b981}
.hosp-name{font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:4px}
.hosp-type{font-size:11px;font-weight:700;padding:2px 8px;border-radius:8px;
  display:inline-block;margin-bottom:6px}
.hosp-type.emergency{background:rgba(239,68,68,.15);color:#f87171}
.hosp-type.specialist{background:rgba(139,92,246,.15);color:#a78bfa}
.hosp-type.general{background:rgba(16,185,129,.15);color:#34d399}
.hosp-addr{font-size:12px;color:#64748b;margin:3px 0}
.hosp-dist{font-size:12px;color:#60a5fa;font-weight:600}
.hosp-phone{font-size:12px;color:#94a3b8}
.hosp-maps{display:inline-flex;align-items:center;gap:5px;margin-top:8px;
  padding:5px 12px;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);
  border-radius:6px;color:#60a5fa;font-size:12px;font-weight:600;
  text-decoration:none;transition:all .2s}
.hosp-maps:hover{background:rgba(59,130,246,.25);color:#93c5fd}
.loc-prompt{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.25);
  border-radius:12px;padding:16px 20px;margin:12px 0}
.loc-prompt-title{font-size:15px;font-weight:700;color:#60a5fa;margin-bottom:6px}
.loc-prompt-sub{font-size:13px;color:#94a3b8;margin-bottom:12px}

/* ── Input area ── */
.stTextInput input{
  background:#1e293b !important;color:#e2e8f0 !important;
  border:1px solid #334155 !important;border-radius:10px !important;
  font-size:15px !important;padding:12px 16px !important
}
.stTextInput input:focus{border-color:#3b82f6 !important;
  box-shadow:0 0 0 3px rgba(59,130,246,.15) !important}
.stTextInput input::placeholder{color:#475569 !important}

/* ── Buttons ── */
.stButton button{
  background:#1e293b !important;color:#e2e8f0 !important;
  border:1px solid #334155 !important;border-radius:8px !important;
  font-size:13px !important
}
.stButton button:hover{border-color:#3b82f6 !important;color:#60a5fa !important}
[data-testid="baseButton-primary"]{
  background:linear-gradient(135deg,#1d4ed8,#3b82f6) !important;
  color:white !important;border:none !important
}
[data-testid="baseButton-primary"]:hover{
  background:linear-gradient(135deg,#1e40af,#2563eb) !important
}
div[data-testid="stDivider"]{border-color:#1e293b}
</style>
""", unsafe_allow_html=True)

# ── Sidebar toggle — safe iframe approach ────────────────────────────────────
# Direct onclick on injected HTML causes React error #231 in Streamlit.
# An iframe with its own script can reach window.parent DOM safely.
st.markdown("""
<iframe srcdoc="
<html><body style='margin:0;padding:0'>
<button id='sb' style='
  width:44px;height:38px;background:#1e40af;color:white;
  border:none;border-radius:8px;font-size:19px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.45);display:flex;
  align-items:center;justify-content:center' title='Toggle sidebar'>&#9776;</button>
<script>
document.getElementById('sb').onclick = function() {
  var sel = '[data-testid=stSidebarCollapseButton] button, [data-testid=collapsedControl] button';
  var btn = window.parent.document.querySelector(sel);
  if (btn) btn.click();
};
</script>
</body></html>
" style="position:fixed;top:10px;left:10px;z-index:9999;
  width:50px;height:44px;border:none;background:transparent;overflow:hidden"
scrolling="no"></iframe>
""", unsafe_allow_html=True)

# ── Persistent history helpers ────────────────────────────────────────────────
def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history: list):
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

def append_to_history(question, answer, citations, accuracy, support_level):
    history = load_history()
    entry = {
        "id":            str(uuid.uuid4())[:8],
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question":      question,
        "answer":        answer[:1200],
        "accuracy":      round(accuracy, 3),
        "support_level": support_level,
        "sources": [
            {"title": c.get("source",""), "org": c.get("organization",""),
             "url": c.get("source_url",""), "score": c.get("relevance_score",0),
             "badge": c.get("web_source_name","")}
            for c in (citations or [])[:6]
        ],
    }
    history.insert(0, entry)
    save_history(history[:200])
    return entry

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"messages":[],"conversation_id":None,"top_k":5,"temporal":True,
              "pending_q":"","kb_count":0,"total_queries":0,
              "view":"chat","selected_history":None,"searching":False,
              "user_location":"","pending_hospital_q":"",
              "awaiting_location":False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helpers ───────────────────────────────────────────────────────────────
def api(method, path, **kw):
    try:
        r = getattr(requests, method)(f"{API_BASE}{path}", timeout=90, **kw)
        return r.json() if r.ok else None
    except: return None

def new_conv():
    r = api("post", "/conversation/new")
    return r.get("conversation_id") if r else str(uuid.uuid4())

SAMPLES = [
    ("WHO Guidelines","WHO Guidelines","2024-01-15",
     "Diabetes mellitus characterized by hyperglycemia. Symptoms: polydipsia, polyuria, weight loss, fatigue, blurred vision. Type 1: autoimmune. Type 2: insulin resistance. Diagnosis: Fasting glucose ≥126 mg/dL or HbA1c ≥6.5%."),
    ("JNC 8 Hypertension 2024","Clinical Research","2024-03-01",
     "Hypertension first-line: DASH diet, sodium <2.3g/day, exercise 150min/week. Drugs: ACE inhibitors, ARBs, CCBs, thiazides. Target BP: <130/80 mmHg."),
    ("WHO COVID-19 2024","WHO Guidelines","2024-06-01",
     "COVID-19 mild: symptomatic treatment. Moderate: nirmatrelvir-ritonavir within 5 days. SpO2<94%: oxygen. Severe: Dexamethasone 6mg/day x10. Tocilizumab for inflammation."),
    ("Harrison's Lab Reference","Medical Textbook","2023-09-01",
     "Haemoglobin M:13.5-17.5 F:12.0-15.5 g/dL. Fasting glucose: 70-100 normal. HbA1c:<5.7% normal. LDL<100 optimal, HDL>40 men, >50 women mg/dL."),
    ("AHA Heart Attack 2023","Clinical Research","2023-12-15",
     "MI: chest pain, radiation to arm/jaw, dyspnoea, nausea. IMMEDIATE: emergency services + aspirin 325mg. STEMI: PCI within 90 min. Long-term: DAPT, statins, ACE-I, beta-blockers."),
]

def seed_data():
    ok = 0
    for title, src, date, text in SAMPLES:
        try:
            requests.post(f"{API_BASE}/ingest/text",
                data={"text":text,"source_type":src,"document_date":date,"title":title},
                timeout=30)
            ok += 1
        except: pass
    return ok

# ── Helpers ───────────────────────────────────────────────────────────────────
def acc_color(score):
    if score >= 0.75: return "#10b981","p-green","High"
    if score >= 0.50: return "#f59e0b","p-yellow","Medium"
    return "#ef4444","p-red","Low"

def src_badge(web_src, org):
    s = (web_src + org).lower()
    if "who"    in s: return "cit-who",    "WHO"
    if "pubmed" in s: return "cit-pubmed", "PubMed"
    if "medline"in s: return "cit-medline","MedlinePlus"
    if "fda"    in s: return "cit-fda",    "FDA"
    return "cit-local", "Local KB"

# ── Hospital Finder ──────────────────────────────────────────────────────────
MEDICAL_KEYWORDS = {
    "disease","symptom","symptoms","diagnosis","treatment","pain","fever","cancer",
    "diabetes","hypertension","heart","cardiac","kidney","liver","lung","asthma",
    "infection","virus","bacteria","surgery","therapy","medicine","medication",
    "doctor","hospital","clinic","specialist","emergency","injury","fracture",
    "allergy","rash","psoriasis","arthritis","depression","anxiety","stroke",
    "cholesterol","blood pressure","blood sugar","obesity","thyroid","epilepsy",
    "migraine","pneumonia","tuberculosis","malaria","dengue","covid","hiv","aids",
    "pregnant","pregnancy","menstrual","gastric","ulcer","diarrhea","constipation",
    "cough","cold","flu","sore throat","ear","eye","dental","bone","skin","hair",
    "wound","bleeding","swelling","vomiting","nausea","fatigue","dizziness",
    "chest pain","shortness of breath","unconscious","fit","seizure","paralysis",
}

def is_medical_question(query: str) -> bool:
    """Detect if the question is about a health/medical topic."""
    q = query.lower()
    return any(kw in q for kw in MEDICAL_KEYWORDS)

def _reverse_geocode(lat: float, lon: float) -> str:
    """Get a human-readable address from coordinates using Nominatim reverse geocode."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 17},
            headers={"User-Agent": "MedRAG-HealthApp/1.0"},
            timeout=5,
        )
        data = r.json()
        addr = data.get("address", {})
        # Build a clean address from the returned components
        parts = []
        for key in ["road", "neighbourhood", "suburb", "city_district", "city", "state"]:
            val = addr.get(key, "")
            if val and val not in parts:
                parts.append(val)
        return ", ".join(parts[:4]) if parts else data.get("display_name","").split(",")[0]
    except Exception:
        return ""


def search_hospitals_osm(location: str, specialty: str = "", limit: int = 6) -> list:
    """
    Search for hospitals near a location using OpenStreetMap Nominatim + Overpass API.
    Completely free, no API key needed.
    Addresses resolved via reverse geocoding for accuracy.
    """
    import math
    hospitals = []
    try:
        # Step 1: Forward geocode the user's location text → lat/lon
        geo_r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "MedRAG-HealthApp/1.0"},
            timeout=8,
        )
        geo_data = geo_r.json()
        if not geo_data:
            return []

        lat      = float(geo_data[0]["lat"])
        lon      = float(geo_data[0]["lon"])
        loc_city = geo_data[0].get("address",{}).get("city","") or                    geo_data[0].get("address",{}).get("state","") or                    geo_data[0].get("display_name","").split(",")[0]

        # Step 2: Overpass query — hospitals + clinics within 15 km
        radius = 15000
        overpass_q = f"""
[out:json][timeout:20];
(
  node["amenity"="hospital"](around:{radius},{lat},{lon});
  way["amenity"="hospital"](around:{radius},{lat},{lon});
  node["amenity"="clinic"](around:{radius},{lat},{lon});
  way["amenity"="clinic"](around:{radius},{lat},{lon});
  node["healthcare"="hospital"](around:{radius},{lat},{lon});
  node["healthcare"="clinic"](around:{radius},{lat},{lon});
);
out center {limit * 3};
"""
        ov_r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=overpass_q, timeout=20,
        )
        elements = ov_r.json().get("elements", [])

        seen = set()
        for el in elements:
            tags  = el.get("tags", {})
            name  = (tags.get("name") or tags.get("name:en") or
                     tags.get("operator") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)

            # Coordinates
            if el["type"] == "node":
                elat, elon = el["lat"], el["lon"]
            else:
                elat = el.get("center", {}).get("lat", lat)
                elon = el.get("center", {}).get("lon", lon)

            # Distance (Haversine)
            dlat = math.radians(elat - lat)
            dlon = math.radians(elon - lon)
            a    = (math.sin(dlat/2)**2 +
                    math.cos(math.radians(lat)) *
                    math.cos(math.radians(elat)) *
                    math.sin(dlon/2)**2)
            dist_km = round(6371 * 2 * math.asin(math.sqrt(a)), 1)

            # Type classification
            amenity = tags.get("amenity", "")
            spec    = tags.get("healthcare:speciality","") or tags.get("speciality","")
            is_hosp = amenity == "hospital" or tags.get("healthcare") == "hospital"
            has_er  = tags.get("emergency","") in ("yes","designated")
            if has_er or is_hosp:
                h_type = "emergency"
            elif spec or amenity == "clinic":
                h_type = "specialist"
            else:
                h_type = "general"

            # Build address — prefer OSM tags, fall back to reverse geocode
            osm_parts = list(filter(None,[
                tags.get("addr:housenumber",""),
                tags.get("addr:street",""),
                tags.get("addr:suburb","") or tags.get("addr:neighbourhood",""),
                tags.get("addr:city","") or tags.get("addr:district","") or loc_city,
                tags.get("addr:state",""),
            ]))
            if len(osm_parts) >= 2:
                address = ", ".join(osm_parts)
            else:
                # Reverse geocode for a proper address
                address = _reverse_geocode(elat, elon) or loc_city

            # Google Maps URL — search by name+location for better results
            maps_query = requests.utils.quote(f"{name} {address}")
            maps_url   = f"https://www.google.com/maps/search/{maps_query}/@{elat},{elon},15z"

            hospitals.append({
                "name":      name,
                "type":      h_type,
                "address":   address,
                "distance":  dist_km,
                "phone":     tags.get("phone") or tags.get("contact:phone",""),
                "website":   tags.get("website") or tags.get("contact:website",""),
                "maps_url":  maps_url,
                "specialty": spec,
                "lat":       elat,
                "lon":       elon,
            })

        hospitals.sort(key=lambda h: h["distance"])
        return hospitals[:limit]

    except Exception as e:
        return []

def render_hospitals(hospitals: list, location: str):
    """Render hospital cards with working links via st.link_button."""
    if not hospitals:
        st.markdown("""
        <div class="warn-box">
          🏥 No hospitals found via OpenStreetMap. Try a more specific area,
          e.g. <em>Hyderabad, Banjara Hills</em> or <em>Mumbai, Andheri</em>.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="hosp-section">
      <div class="hosp-header">
        🏥 Nearby hospitals &amp; clinics &nbsp;
        <span style="font-size:11px;color:#475569;font-weight:400">
          near {location} · sorted by distance · OpenStreetMap data
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    for idx, h in enumerate(hospitals):
        h_type   = h.get("type","general")
        name     = h.get("name","Unknown Hospital")
        address  = h.get("address","") or "Address not listed in OpenStreetMap"
        dist     = h.get("distance","?")
        phone    = h.get("phone","")
        maps_url = h.get("maps_url","")
        website  = h.get("website","")
        spec     = h.get("specialty","")

        tl       = {"emergency":"🚨 Emergency","specialist":"🔬 Specialist",
                    "general":"🏥 General"}.get(h_type,"🏥 Hospital")
        phone_html = f'<div class="hosp-phone">📞 {phone}</div>' if phone else ""
        spec_html  = f'<div class="hosp-addr">🔬 Speciality: {spec}</div>' if spec else ""

        # Card info — no anchor tags (they don't work in Streamlit iframe)
        st.markdown(f"""
        <div class="hosp-card {h_type}">
          <div class="hosp-name">{name}</div>
          <span class="hosp-type {h_type}">{tl}</span>
          <div class="hosp-dist">📍 {dist} km away</div>
          <div class="hosp-addr">🗺️ {address}</div>
          {spec_html}
          {phone_html}
        </div>""", unsafe_allow_html=True)

        # Native Streamlit link buttons — guaranteed to open in browser
        if maps_url or website:
            cols = st.columns([1, 1, 4])
            with cols[0]:
                if maps_url:
                    st.link_button("📍 Maps", maps_url, use_container_width=True)
            with cols[1]:
                if website:
                    st.link_button("🌐 Site", website, use_container_width=True)

# ── Live search progress display ──────────────────────────────────────────────
# Search steps used by st.status() display
SEARCH_STEPS = [
    ("🔍", "Embedding your question"),
    ("🗄️", "Searching local knowledge base"),
    ("🌐", "Fetching WHO guidelines"),
    ("🔬", "Searching PubMed research papers"),
    ("💊", "Checking MedlinePlus database"),
    ("📊", "Ranking results by accuracy"),
    ("🤖", "Generating evidence-based answer"),
    ("✅", "Verifying accuracy and safety"),
]

# ── Render helpers ────────────────────────────────────────────────────────────
def render_accuracy_bar(score, level):
    color, _, label = acc_color(score)
    pct = int(score * 100)
    st.markdown(f"""
    <div class="acc-wrap">
      <div class="acc-header">
        <span class="acc-title">Evidence Accuracy</span>
        <span class="acc-score" style="color:{color}">{pct}%</span>
      </div>
      <div class="acc-bar-bg">
        <div class="acc-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}99)"></div>
      </div>
      <div class="acc-sublabel">{label} confidence · Based on {len(SEARCH_STEPS)-1} source checks</div>
    </div>""", unsafe_allow_html=True)

def render_pills(d):
    conf  = d.get("confidence_score", 0)
    level = d.get("support_level","medium")
    hall  = d.get("hallucination_detected", False)
    safe  = d.get("is_safe", True)
    sev   = d.get("safety_severity","none")
    web   = d.get("web_fallback_used", False)
    c_cls = "p-green" if conf>=0.7 else "p-yellow" if conf>=0.4 else "p-red"
    h_cls = "p-green" if not hall else "p-red"
    s_cls = "p-green" if safe else ("p-orange" if sev=="medium" else "p-red")
    l_cls = "p-green" if level=="high" else "p-yellow" if level=="medium" else "p-red"
    web_p = '<span class="pill p-blue">🌐 Web enriched</span>' if web else ""
    st.markdown(f"""
    <div style="margin:8px 0 6px">
      <span class="pill {c_cls}">Confidence {conf:.0%}</span>
      <span class="pill {l_cls}">{level.title()} evidence</span>
      <span class="pill {h_cls}">{"✓ Verified" if not hall else "⚠ Check claims"}</span>
      <span class="pill {s_cls}">{"✓ Safe" if safe else "⚠ Safety concern"}</span>
      {web_p}
    </div>""", unsafe_allow_html=True)

def render_proof_panel(d):
    supported   = d.get("well_supported_claims",[])
    unsupported = d.get("unsupported_claims",[])
    explanation = d.get("accuracy_explanation","")
    if supported or explanation:
        items = "".join(f"<li style='margin:3px 0'>{c}</li>" for c in supported[:3])
        st.markdown(f"""
        <div class="proof-box">
          <strong>✅ Verified by sources:</strong>
          {"<ul style='margin:6px 0 0;padding-left:18px'>" + items + "</ul>" if items else ""}
          {f"<p style='margin:6px 0 0;color:#94a3b8'>{explanation}</p>" if explanation else ""}
        </div>""", unsafe_allow_html=True)
    if unsupported:
        items = "".join(f"<li style='margin:3px 0'>{c}</li>" for c in unsupported[:3])
        st.markdown(f"""
        <div class="warn-box">
          <strong>⚠️ Verify independently:</strong>
          <ul style="margin:4px 0 0;padding-left:18px">{items}</ul>
        </div>""", unsafe_allow_html=True)

def render_citations(cits):
    """Sort citations by relevance_score (highest first) then render."""
    if not cits: return

    sorted_cits = sorted(cits, key=lambda c: c.get("relevance_score", 0), reverse=True)
    rank_colors = {1:"rank-1", 2:"rank-2", 3:"rank-3"}

    st.markdown('<div class="cit-header">📎 Proof &amp; Sources — sorted by accuracy</div>',
                unsafe_allow_html=True)

    for i, c in enumerate(sorted_cits, 1):
        sc       = int(c.get("relevance_score", 0) * 100)
        bar_col  = "#10b981" if sc>=70 else "#f59e0b" if sc>=40 else "#ef4444"
        title    = c.get("source") or "Unknown"
        org      = c.get("organization") or ""
        excerpt  = (c.get("excerpt") or "")[:260]
        date     = (c.get("document_date") or "")[:10]
        url      = c.get("source_url") or ""
        web_src  = c.get("web_source_name") or ""
        card_cls, badge = src_badge(web_src, org)
        rank_cls = rank_colors.get(i, "rank-n")
        gold_border = ' style="border-color:#f59e0b"' if i == 1 else ""

        st.markdown(f"""
        <div class="cit-card {card_cls}"{gold_border}>
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                <span class="cit-rank {rank_cls}">#{i}</span>
                <span class="badge-pill">{badge}</span>
                <span class="cit-title">{title}</span>
              </div>
              <div class="cit-org">🏛️ {org}{"&nbsp;·&nbsp;" + date if date else ""}</div>
            </div>
            <div style="text-align:right;margin-left:14px;min-width:52px">
              <div style="font-size:18px;font-weight:800;color:{bar_col}">{sc}%</div>
              <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.3px">match</div>
              <div style="height:3px;background:#0f172a;border-radius:2px;margin-top:4px">
                <div style="width:{sc}%;height:3px;background:{bar_col};border-radius:2px"></div>
              </div>
            </div>
          </div>
          <div class="cit-excerpt">"{excerpt}{"..." if len(c.get("excerpt",""))>260 else ""}"</div>
        </div>""", unsafe_allow_html=True)

        # st.link_button is the ONLY Streamlit component that reliably opens URLs
        if url:
            st.link_button(f"🔗 Open source — {badge} ↗", url, use_container_width=False)
        else:
            st.caption("📄 Local knowledge base document")


def render_safety(d):
    if d.get("is_safe", True):
        disc = d.get("safety_disclaimer","")
        if disc:
            st.markdown(f'<div class="warn-box">⚕️ {disc}</div>', unsafe_allow_html=True)
    else:
        concerns = d.get("safety_concerns",[])
        items = "".join(f"<li>{c}</li>" for c in concerns) if concerns else ""
        st.markdown(f"""
        <div class="unsafe-box">
          <strong>⚠️ Safety concern — {d.get("safety_severity","").upper()}</strong>
          {f"<ul style='margin:4px 0 0;padding-left:18px'>{items}</ul>" if items else ""}
          <p style="margin:6px 0 0;color:#fca5a5">Consult a healthcare professional before acting on this.</p>
        </div>""", unsafe_allow_html=True)

def render_msg(msg):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑&nbsp; {msg["content"]}</div>',
                    unsafe_allow_html=True)
        return

    d      = msg.get("data", {})
    answer = d.get("answer", msg["content"])
    no_ctx = "unable to find sufficient" in answer.lower() or "insufficient medical" in answer.lower()

    if no_ctx:
        st.markdown(f'<div class="no-ctx">⚠️&nbsp; {answer}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🏥&nbsp; {answer}</div>', unsafe_allow_html=True)
        if d:
            render_accuracy_bar(d.get("accuracy_score", d.get("confidence_score",0)),
                                d.get("support_level","medium"))
            render_pills(d)
            if d.get("web_fallback_used"):
                st.markdown("""<div class="web-banner">
                  🌐 <strong>Web sources included</strong> — Evidence gathered from
                  WHO, PubMed and MedlinePlus for comprehensive coverage.
                </div>""", unsafe_allow_html=True)
            render_proof_panel(d)
            render_citations(d.get("citations",[]))
            render_safety(d)

    # Show hospitals if they were fetched for this message
    hospitals = msg.get("hospitals")
    if hospitals is not None:
        render_hospitals(hospitals, msg.get("hospital_location","your location"))

    # Show location prompt if this message requests it
    if msg.get("ask_location"):
        st.markdown("""
        <div class="loc-prompt">
          <div class="loc-prompt-title">📍 Find nearby hospitals?</div>
          <div class="loc-prompt-sub">
            Share your locality or city to find relevant hospitals near you.
          </div>
        </div>""", unsafe_allow_html=True)
        col_loc, col_go = st.columns([5,1])
        with col_loc:
            loc_val = st.text_input("location_input", label_visibility="collapsed",
                placeholder="e.g. Hyderabad, Banjara Hills  or  Mumbai, Andheri",
                key=f"loc_{msg.get('msg_id','0')}",
                value=st.session_state.get("user_location",""))
        with col_go:
            if st.button("Find 🏥", key=f"find_{msg.get('msg_id','0')}",
                         type="primary", use_container_width=True):
                if loc_val.strip():
                    st.session_state.user_location = loc_val.strip()
                    # Trigger hospital search and update message
                    msg["ask_location"] = False
                    with st.spinner(f"🔍 Finding hospitals near {loc_val.strip()}…"):
                        hosp = search_hospitals_osm(loc_val.strip())
                    msg["hospitals"] = hosp
                    msg["hospital_location"] = loc_val.strip()
                    st.rerun()

# ── History view ──────────────────────────────────────────────────────────────
def render_history_view():
    history = load_history()

    col_back, col_title, col_del = st.columns([1,7,1])
    with col_back:
        if st.button("← Chat"):
            st.session_state.view = "chat"
            st.session_state.selected_history = None
            st.rerun()
    with col_title:
        st.markdown(f"### 📖 Q&A History &nbsp;<span style='font-size:14px;color:#64748b'>({len(history)} saved)</span>",
                    unsafe_allow_html=True)
    with col_del:
        if history and st.button("🗑️ Clear"):
            save_history([])
            st.rerun()

    if not history:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:40px;margin-bottom:12px">📭</div>
          <div style="font-size:17px;color:#64748b">No history yet</div>
          <div style="font-size:13px;margin-top:6px;color:#475569">
            Every answer is saved automatically as you chat.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    search = st.text_input("🔍 Search", placeholder="Filter by keyword…", key="hist_search")
    filtered = [h for h in history
                if not search or search.lower() in h["question"].lower()
                               or search.lower() in h["answer"].lower()]
    st.caption(f"{len(filtered)} of {len(history)} entries")

    sel = st.session_state.selected_history
    if sel:
        entry = next((h for h in history if h["id"]==sel), None)
        if entry:
            if st.button("← Back to list"):
                st.session_state.selected_history = None
                st.rerun()
            st.divider()
            acc = entry.get("accuracy",0)
            color, _, lbl = acc_color(acc)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
              <span style="color:#64748b;font-size:13px">🗓️ {entry["timestamp"]}</span>
              <span class="pill" style="background:rgba(255,255,255,.05);
                color:{color};border:1px solid {color}40">
                {int(acc*100)}% · {lbl}
              </span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">🧑&nbsp; {entry["question"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="bot-bubble">🏥&nbsp; {entry["answer"]}</div>',
                        unsafe_allow_html=True)

            # Sources sorted by score
            sources = sorted(entry.get("sources",[]), key=lambda s:s.get("score",0), reverse=True)
            if sources:
                st.markdown('<div class="cit-header">📎 Sources — sorted by accuracy</div>',
                            unsafe_allow_html=True)
                rank_cls_map = {1:"rank-1",2:"rank-2",3:"rank-3"}
                for j, s in enumerate(sources, 1):
                    url    = s.get("url","")
                    stitle = s.get("title","Unknown")
                    sorg   = s.get("org","")
                    sbadge = s.get("badge","")
                    sscore = int(s.get("score",0)*100)
                    card_cls, badge_txt = src_badge(sbadge, sorg)
                    bar_col = "#10b981" if sscore>=70 else "#f59e0b" if sscore>=40 else "#ef4444"
                    rank_c  = rank_cls_map.get(j,"rank-n")
                    link    = (f'<a href="{url}" target="_blank" class="src-link">🔗 Open source ↗</a>'
                               if url else "")
                    st.markdown(f"""
                    <div class="cit-card {card_cls}" style="margin:4px 0">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                          <span class="cit-rank {rank_c}">#{j}</span>
                          <span class="badge-pill">{badge_txt}</span>
                          <strong style="color:#e2e8f0">{stitle}</strong><br>
                          <span class="cit-org">🏛️ {sorg}</span>
                        </div>
                        <div style="text-align:right;min-width:44px">
                          <div style="font-size:16px;font-weight:800;color:{bar_col}">{sscore}%</div>
                          <div style="height:3px;background:#0f172a;border-radius:2px;margin-top:3px">
                            <div style="width:{sscore}%;height:3px;background:{bar_col};border-radius:2px"></div>
                          </div>
                        </div>
                      </div>
                      {link}
                    </div>""", unsafe_allow_html=True)

            if st.button("💬 Ask again"):
                st.session_state.pending_q = entry["question"]
                st.session_state.view = "chat"
                st.session_state.selected_history = None
                st.rerun()
            return

    for entry in filtered:
        acc = entry.get("accuracy",0)
        color,_,lbl = acc_color(acc)
        n_src = len(entry.get("sources",[]))
        c1, c2 = st.columns([10,1])
        with c1:
            st.markdown(f"""
            <div class="hist-card">
              <div class="hist-q">{entry["question"]}</div>
              <div class="hist-meta">
                🗓️ {entry["timestamp"]} &nbsp;·&nbsp;
                <span style="color:{color};font-weight:700">{int(acc*100)}% {lbl}</span>
                &nbsp;·&nbsp; {n_src} source{"s" if n_src!=1 else ""}
              </div>
              <div class="hist-snippet">{entry["answer"][:180]}{"…" if len(entry["answer"])>180 else ""}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("View", key=f"v_{entry['id']}"):
                st.session_state.selected_history = entry["id"]
                st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px">
      <div style="font-size:20px;font-weight:800;color:#60a5fa">🏥 MedRAG AI</div>
      <div style="font-size:12px;color:#64748b;margin-top:3px">Health Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    health = api("get","/health")
    if health:
        kb = health.get("vector_store",{}).get("total_documents",0)
        st.session_state.kb_count = kb
        st.markdown(f'<span class="online" style="color:#34d399 !important">● Online</span>'
                    f'&nbsp;·&nbsp;<strong>{kb}</strong> local docs',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#f87171 !important">● Offline</span> — run `python main.py`',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("**🗂️ Navigate**")
    ca, cb2 = st.columns(2)
    with ca:
        if st.button("💬 Chat", use_container_width=True,
                     type="primary" if st.session_state.view=="chat" else "secondary"):
            st.session_state.view = "chat"
            st.session_state.selected_history = None
            st.rerun()
    with cb2:
        hc = len(load_history())
        if st.button(f"📖 History ({hc})", use_container_width=True,
                     type="primary" if st.session_state.view=="history" else "secondary"):
            st.session_state.view = "history"
            st.session_state.selected_history = None
            st.rerun()

    st.divider()
    st.markdown("**⚙️ Settings**")
    st.session_state.top_k    = st.slider("Docs to retrieve", 1, 10, 5)
    st.session_state.temporal = st.toggle("Prefer recent guidelines", value=True)

    st.divider()
    st.markdown("**📍 Your Location**")
    st.caption("Save your locality for automatic hospital suggestions")
    loc_input = st.text_input("loc_sidebar", label_visibility="collapsed",
        placeholder="e.g. Hyderabad, Banjara Hills",
        value=st.session_state.get("user_location",""),
        key="sidebar_location")
    if loc_input != st.session_state.get("user_location",""):
        st.session_state.user_location = loc_input
    if st.session_state.get("user_location"):
        st.markdown(f'<span style="color:#34d399;font-size:12px">✓ Location saved: {st.session_state.user_location}</span>',
                    unsafe_allow_html=True)
        if st.button("🔄 Clear location", use_container_width=True):
            st.session_state.user_location = ""
            st.rerun()

    st.divider()
    st.markdown("**💬 Quick questions**")
    SAMPLE_Qs = [
        "Symptoms of diabetes?",
        "How is hypertension treated?",
        "Heart attack warning signs?",
        "Normal blood glucose level?",
        "COVID-19 treatment options?",
        "What does high LDL mean?",
        "Symptoms of kidney disease?",
        "How is depression treated?",
        "Type 1 vs Type 2 diabetes?",
        "Causes of high blood pressure?",
    ]
    for q in SAMPLE_Qs:
        if st.button(q, key=f"sq_{hash(q)}"):
            st.session_state.pending_q = q
            st.session_state.view = "chat"
            st.rerun()

    st.divider()
    st.markdown("**📚 Knowledge base**")
    ca2, cb3 = st.columns(2)
    with ca2:
        if st.button("🌱 Seed", use_container_width=True, type="primary"):
            with st.spinner("Seeding…"):
                n = seed_data()
            st.success(f"✅ {n} docs!") if n else st.error("Failed.")
            st.rerun()
    with cb3:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    with st.expander("📤 Upload document"):
        upf = st.file_uploader("PDF / TXT / MD", type=["pdf","txt","md"])
        ups = st.selectbox("Source", ["WHO Guidelines","PubMed","Medical Textbook",
                                       "Clinical Research","Lab Report","Other"])
        upd = st.date_input("Date")
        if st.button("Ingest") and upf:
            with st.spinner("Processing…"):
                try:
                    res = requests.post(f"{API_BASE}/ingest/file",
                        files={"file":(upf.name,upf.getvalue(),upf.type)},
                        data={"source_type":ups,"document_date":str(upd)}, timeout=60)
                    st.success(f"✅ {res.json()['chunks_created']} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <h1>🏥 MedRAG — Health Intelligence</h1>
  <p>Evidence-based medical answers with real-time source verification</p>
  <div>
    <span class="badge">WHO</span>
    <span class="badge">PubMed</span>
    <span class="badge">MedlinePlus</span>
    <span class="badge">FDA</span>
    <span class="badge">Local KB</span>
    <span class="badge">Accuracy Scoring</span>
    <span class="badge">Sorted by Relevance</span>
  </div>
</div>""", unsafe_allow_html=True)

if st.session_state.view == "history":
    render_history_view()

else:
    if health:
        stats = api("get","/stats") or {}
        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl in [
            (c1, st.session_state.kb_count,          "Local chunks"),
            (c2, stats.get("active_conversations",0), "Sessions"),
            (c3, st.session_state.total_queries,      "This session"),
            (c4, len(load_history()),                  "Saved Q&As"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:56px;margin-bottom:16px">🏥</div>
          <div style="font-size:22px;font-weight:700;color:#60a5fa;margin-bottom:8px">
            Ask any health question
          </div>
          <div style="font-size:14px;color:#475569;line-height:1.7">
            Get evidence-based answers with accuracy scores, ranked source citations,<br>
            and clickable links to WHO, PubMed, MedlinePlus and more.<br>
            Every answer is automatically saved to your History.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_msg(msg)

    st.divider()
    ci, cb4 = st.columns([8, 1])
    with ci:
        val = st.session_state.pop("pending_q","") if st.session_state.get("pending_q") else ""
        user_input = st.text_input("q", label_visibility="collapsed",
                                    placeholder="e.g. What are the symptoms of diabetes?",
                                    value=val, key="main_input")
    with cb4:
        send = st.button("Send ➤", type="primary", use_container_width=True)

    if send and user_input.strip():
        q = user_input.strip()
        if not st.session_state.conversation_id:
            st.session_state.conversation_id = new_conv()

        st.session_state.messages.append({"role":"user","content":q})
        st.session_state.total_queries += 1

        # ── Live search progress using st.status (renders correctly) ────────────
        data = None
        err  = None
        with st.status("🔍 Searching medical sources…", expanded=True) as status:
            try:
                for icon, label in SEARCH_STEPS[:6]:
                    st.write(f"{icon} {label}…")
                    time.sleep(0.45)

                st.write("🤖 Generating evidence-based answer…")
                resp = requests.post(f"{API_BASE}/query", json={
                    "query": q,
                    "conversation_id": st.session_state.conversation_id,
                    "top_k": st.session_state.top_k,
                    "include_temporal_ranking": st.session_state.temporal,
                }, timeout=90)
                resp.raise_for_status()
                data = resp.json()

                st.write("✅ Verifying accuracy and safety…")
                time.sleep(0.3)
                status.update(label="✅ Done — answer ready", state="complete", expanded=False)

            except requests.exceptions.ConnectionError:
                status.update(label="❌ Backend offline", state="error", expanded=False)
                err = "❌ Cannot reach the backend. Is `python main.py` running?"
            except Exception as e:
                status.update(label="❌ Error occurred", state="error", expanded=False)
                err = f"❌ Error: {e}"

        if data:
            # Detect if this is a medical question — if yes, offer hospital finder
            is_medical = is_medical_question(q)
            msg_id = str(uuid.uuid4())[:8]

            # If user already has a saved location, search immediately
            saved_loc = st.session_state.get("user_location","").strip()
            hospitals = None
            hospital_location = ""
            ask_location = False

            if is_medical and saved_loc:
                with st.status(f"🏥 Finding hospitals near {saved_loc}…", expanded=False):
                    hospitals = search_hospitals_osm(saved_loc)
                hospital_location = saved_loc
            elif is_medical:
                ask_location = True   # prompt the user for location

            st.session_state.messages.append({
                "role":             "assistant",
                "content":          data.get("answer",""),
                "data":             data,
                "msg_id":           msg_id,
                "ask_location":     ask_location,
                "hospitals":        hospitals,
                "hospital_location":hospital_location,
            })
            append_to_history(
                question      = q,
                answer        = data.get("answer",""),
                citations     = data.get("citations",[]),
                accuracy      = data.get("accuracy_score", data.get("confidence_score",0)),
                support_level = data.get("support_level","medium"),
            )
        elif err:
            st.session_state.messages.append({"role":"assistant","content":err,"data":{}})

        st.rerun()
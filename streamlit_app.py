"""
Medical Knowledge RAG System - Streamlit Frontend
Run with: streamlit run streamlit_app.py
Requires FastAPI backend running: python main.py
"""
import streamlit as st
import requests
import uuid
from datetime import datetime

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #f0f4f8; }
#MainMenu, footer, header { visibility: hidden; }

.top-header {
    background: #1e40af;
    color: white;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.top-header h1 { margin:0; font-size:26px; color:white; }
.top-header p  { margin:6px 0 0; font-size:14px; color:#bfdbfe; }

.user-bubble {
    background: #1d4ed8;
    color: white;
    padding: 13px 18px;
    border-radius: 20px 20px 4px 20px;
    margin: 10px 0 10px auto;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.6;
    width: fit-content;
}
.bot-bubble {
    background: white;
    color: #1e293b;
    padding: 16px 20px;
    border-radius: 4px 20px 20px 20px;
    margin: 10px 0;
    max-width: 90%;
    border: 1px solid #e2e8f0;
    font-size: 15px;
    line-height: 1.75;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.citation-card {
    border-left: 4px solid #3b82f6;
    background: #f8fafc;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.citation-who   { border-left-color: #0ea5e9; }
.citation-pub   { border-left-color: #8b5cf6; }
.citation-lab   { border-left-color: #f59e0b; }
.citation-text  { border-left-color: #10b981; }

.pill {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 3px 2px 0;
}
.pill-green  { background:#dcfce7; color:#166534; }
.pill-yellow { background:#fef9c3; color:#854d0e; }
.pill-red    { background:#fee2e2; color:#991b1b; }
.pill-blue   { background:#dbeafe; color:#1e40af; }

.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #92400e;
    margin-top: 10px;
}
.highlight-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 9px 13px;
    font-size: 13px;
    color: #166534;
    margin: 5px 0;
    font-style: italic;
}
.stat-card {
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.stat-val { font-size: 28px; font-weight: 700; color: #1e40af; }
.stat-lbl { font-size: 12px; color: #64748b; margin-top: 2px; }
.empty-state {
    text-align: center;
    padding: 70px 20px;
    color: #94a3b8;
}
.empty-state .icon { font-size: 52px; margin-bottom: 14px; }
.empty-state .title { font-size: 20px; font-weight: 600; color: #64748b; margin-bottom: 8px; }
.empty-state .sub   { font-size: 14px; }
.online-dot  { color: #22c55e; font-weight: 700; }
.offline-dot { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "conversation_id": None,
    "top_k": 5,
    "temporal": True,
    "pending_q": "",
    "kb_count": 0,
    "total_queries": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helpers ───────────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except:
        return None

def api_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=3)
        return r.json() if r.ok else {}
    except:
        return {}

def api_new_conv():
    try:
        r = requests.post(f"{API_BASE}/conversation/new", timeout=5)
        return r.json().get("conversation_id") if r.ok else str(uuid.uuid4())
    except:
        return str(uuid.uuid4())

def api_query(q, conv_id, top_k, temporal):
    r = requests.post(f"{API_BASE}/query", json={
        "query": q,
        "conversation_id": conv_id,
        "top_k": top_k,
        "include_temporal_ranking": temporal,
    }, timeout=90)
    r.raise_for_status()
    return r.json()

def api_ingest_text(text, source, date, title):
    r = requests.post(f"{API_BASE}/ingest/text",
        data={"text": text, "source_type": source,
              "document_date": date, "title": title}, timeout=30)
    r.raise_for_status()
    return r.json()

def api_ingest_file(f, source, date):
    r = requests.post(f"{API_BASE}/ingest/file",
        files={"file": (f.name, f.getvalue(), f.type)},
        data={"source_type": source, "document_date": date}, timeout=60)
    r.raise_for_status()
    return r.json()

SAMPLE_DOCS = [
    ("WHO Diabetes Guidelines 2024", "WHO Guidelines", "2024-01-15",
     """Diabetes mellitus is a group of metabolic diseases characterized by hyperglycemia.
Symptoms: excessive thirst (polydipsia), frequent urination (polyuria), unexplained weight loss,
fatigue, blurred vision, slow-healing wounds, frequent infections.
Type 1 results from autoimmune destruction of pancreatic beta cells.
Type 2 results from progressive loss of beta-cell insulin secretion with insulin resistance.
Diagnosis: Fasting glucose >= 126 mg/dL or HbA1c >= 6.5% on two separate occasions."""),

    ("JNC 8 Hypertension Guidelines 2024", "Clinical Research", "2024-03-01",
     """Hypertension Treatment Guidelines 2024: First-line includes lifestyle modifications:
DASH diet, sodium restriction < 2.3g/day, aerobic exercise 150 min/week, weight loss if BMI > 25.
Pharmacological treatment when BP > 140/90 mmHg. First-line medications: ACE inhibitors or ARBs,
Calcium channel blockers, Thiazide diuretics. Target: < 130/80 mmHg for most adults."""),

    ("WHO COVID-19 Clinical Management 2024", "WHO Guidelines", "2024-06-01",
     """COVID-19 Treatment 2024: Mild: symptomatic treatment, hydration, monitor deterioration.
Moderate: nirmatrelvir-ritonavir within 5 days for high-risk. O2 if SpO2 < 94%.
Severe: Dexamethasone 6mg/day x10 days. Remdesivir if hospitalised. Tocilizumab for severe inflammation.
Vaccination remains primary prevention."""),

    ("Harrison's Lab Reference Ranges", "Medical Textbook", "2023-09-01",
     """Normal Blood Test Ranges (Adults): Hemoglobin M:13.5-17.5 g/dL F:12.0-15.5 g/dL.
WBC: 4500-11000/mcL. Platelets: 150k-400k/mcL.
Fasting Glucose: 70-100 mg/dL normal, 100-125 prediabetes.
Creatinine M:0.74-1.35 F:0.59-1.04 mg/dL. Na: 136-145. K: 3.5-5.0 mEq/L.
Total Cholesterol < 200, LDL < 100 optimal, HDL > 40 men / > 50 women mg/dL."""),

    ("AHA Heart Attack Guidelines 2023", "Clinical Research", "2023-12-15",
     """Myocardial Infarction Warning Signs: Chest pain/pressure, pain to left arm/jaw/back,
shortness of breath, nausea, cold sweats. Women may have atypical symptoms.
IMMEDIATE: Call emergency services. Aspirin 325mg if not allergic. Time is Muscle.
STEMI: Primary PCI gold standard within 90 minutes.
Long-term: Dual antiplatelet therapy, statins, ACE inhibitors, beta-blockers, cardiac rehab."""),
]

def seed_data():
    ok = 0
    for title, source, date, text in SAMPLE_DOCS:
        try:
            api_ingest_text(text, source, date, title)
            ok += 1
        except:
            pass
    return ok

# ── Render helpers ────────────────────────────────────────────────────────────
def conf_pill(score):
    if score >= 0.7: return "pill-green", f"Confidence {score:.0%}"
    if score >= 0.4: return "pill-yellow", f"Confidence {score:.0%}"
    return "pill-red", f"Confidence {score:.0%}"

def src_cls(source):
    s = source.lower()
    if "who" in s: return "citation-who"
    if "pubmed" in s or "clinical" in s or "research" in s: return "citation-pub"
    if "lab" in s: return "citation-lab"
    return "citation-text"

def render_citations(cits):
    if not cits: return
    st.markdown("**📎 Sources used**")
    for i, c in enumerate(cits, 1):
        sc = int(c.get("relevance_score", 0) * 100)
        bar = "#10b981" if sc>=70 else "#f59e0b" if sc>=40 else "#ef4444"
        meta = " · ".join(filter(None, [
            (c.get("document_date") or "")[:10],
            c.get("guideline_version") or "",
            f"p.{c['page']}" if c.get("page") else ""
        ]))
        ex = (c.get("excerpt") or "")[:180]
        st.markdown(f"""
        <div class="citation-card {src_cls(c.get('source',''))}">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <strong>[{i}] {c.get('source','Unknown')}</strong>
            <span style="font-size:11px;color:#64748b">{meta}</span>
          </div>
          <div style="font-style:italic;color:#475569;margin-bottom:6px;font-size:12px">"{ex}…"</div>
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:11px;color:#64748b">Relevance</span>
            <div style="flex:1;height:5px;background:#e2e8f0;border-radius:3px">
              <div style="width:{sc}%;height:5px;background:{bar};border-radius:3px"></div>
            </div>
            <strong style="font-size:12px;color:{bar}">{sc/100:.2f}</strong>
          </div>
        </div>""", unsafe_allow_html=True)

def render_pills(data):
    cc, ct = conf_pill(data.get("confidence_score", 0))
    h = data.get("hallucination_detected", False)
    s = data.get("is_safe", True)
    hp = ("pill-green","No hallucination") if not h else ("pill-red","Hallucination detected")
    sp = ("pill-green","Safe") if s else ("pill-red","Safety concern")
    st.markdown(f"""
    <div style="margin:8px 0 12px">
      <span class="pill {cc}">{ct}</span>
      <span class="pill {hp[0]}">{hp[1]}</span>
      <span class="pill {sp[0]}">{sp[1]}</span>
    </div>""", unsafe_allow_html=True)

def render_contexts(ctxs):
    if not ctxs: return
    with st.expander("📄 View retrieved context snippets"):
        for i, c in enumerate(ctxs[:3], 1):
            st.markdown(f"""
            <div class="highlight-box">
              <strong>Snippet {i}:</strong> {c[:280]}{'…' if len(c)>280 else ''}
            </div>""", unsafe_allow_html=True)

def render_msg(msg):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑&nbsp; {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        d = msg.get("data", {})
        st.markdown(f'<div class="bot-bubble">🏥&nbsp; {d.get("answer", msg["content"])}</div>',
                    unsafe_allow_html=True)
        if d:
            render_pills(d)
            render_citations(d.get("citations", []))
            render_contexts(d.get("highlighted_contexts", []))
            disc = d.get("safety_disclaimer", "")
            if disc:
                st.markdown(f'<div class="disclaimer">⚠️ {disc}</div>',
                            unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical RAG")
    st.caption("AI-powered medical knowledge assistant")
    st.divider()

    health = api_health()
    if health:
        vs = health.get("vector_store", {})
        kb = vs.get("total_documents", 0)
        st.session_state.kb_count = kb
        st.markdown(f'<span class="online-dot">● Online</span> &nbsp;|&nbsp; <strong>{kb}</strong> chunks in KB',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="offline-dot">● Offline</span> — run `python main.py`',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Query settings")
    st.session_state.top_k = st.slider("Docs to retrieve", 1, 10, 5)
    st.session_state.temporal = st.toggle("Prefer recent guidelines", value=True)

    st.divider()
    st.markdown("### 💬 Try a sample question")
    SAMPLES = [
        "What are the symptoms of diabetes?",
        "Latest COVID-19 treatment guidelines?",
        "Normal blood glucose range?",
        "How is hypertension treated?",
        "Warning signs of a heart attack?",
        "What does high LDL cholesterol indicate?",
        "What is a normal HbA1c value?",
        "Difference between Type 1 and Type 2 diabetes?",
    ]
    for q in SAMPLES:
        if st.button(q, key=f"sq_{hash(q)}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge base")
    if st.session_state.kb_count == 0:
        st.info("Empty KB — seed sample data first.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌱 Seed data", use_container_width=True, type="primary"):
            with st.spinner("Seeding…"):
                n = seed_data()
            if n:
                st.success(f"Seeded {n} docs!")
                st.rerun()
            else:
                st.error("Failed.")
    with col_b:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    st.divider()
    st.markdown("### 📤 Upload document")
    with st.expander("Add to knowledge base"):
        up_file = st.file_uploader("PDF / TXT / MD", type=["pdf","txt","md"])
        up_src  = st.selectbox("Source", ["WHO Guidelines","PubMed","Medical Textbook",
                                          "Clinical Research","Lab Report","Other"])
        up_date = st.date_input("Date")
        if st.button("Upload & ingest") and up_file:
            with st.spinner("Processing…"):
                try:
                    res = api_ingest_file(up_file, up_src, str(up_date))
                    st.success(f"✅ {res['chunks_created']} chunks created!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <h1>🏥 Medical Knowledge Assistant</h1>
  <p>Evidence-based answers · Source citations · Hallucination detection · Safety filtering · Temporal ranking</p>
</div>""", unsafe_allow_html=True)

# Stats row
if health:
    stats = api_stats()
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, stats.get("vector_store",{}).get("total_documents",0), "Knowledge chunks"),
        (c2, stats.get("active_conversations",0), "Active sessions"),
        (c3, st.session_state.total_queries, "Queries this session"),
        (c4, stats.get("config",{}).get("retrieval_top_k",5), "Docs per query"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-val">{val}</div>
          <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# KB empty warning
if st.session_state.kb_count == 0 and health:
    st.warning("📭 Knowledge base is empty. Click **Seed data** in the sidebar to load sample WHO, clinical & textbook data.")

# Chat area
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">🏥</div>
      <div class="title">Ask a medical question</div>
      <div class="sub">
        Get evidence-based answers from WHO guidelines, clinical research,<br>
        and medical textbooks — with citations and confidence scores.
      </div>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_msg(msg)

st.divider()

# Input row
c_inp, c_btn = st.columns([7, 1])
with c_inp:
    default_val = st.session_state.pop("pending_q", "") if st.session_state.get("pending_q") else ""
    user_input = st.text_input("query", label_visibility="collapsed",
                                placeholder="Ask a medical question…",
                                value=default_val, key="main_input")
with c_btn:
    send = st.button("Send ➤", type="primary", use_container_width=True)

# Process
if send and user_input.strip():
    q = user_input.strip()
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = api_new_conv()

    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.total_queries += 1

    with st.spinner("🔍 Searching medical knowledge base and generating answer…"):
        try:
            resp = api_query(q, st.session_state.conversation_id,
                             st.session_state.top_k, st.session_state.temporal)
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.get("answer",""),
                "data": resp,
            })
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role":"assistant",
                "content":"❌ Cannot reach the backend. Is `python main.py` running on port 8000?",
                "data":{},
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":"assistant",
                "content":f"❌ Error: {e}",
                "data":{},
            })
    st.rerun()
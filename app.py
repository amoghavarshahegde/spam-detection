import html
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuroShield | AI Spam Detection",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_spam")

MAX_LEN = 128        # tokens per window
STRIDE = 32          # overlap between windows for long emails
MAX_CHUNKS = 8       # longest email analysed = 8 windows
BATCH_SIZE = 32
SPAM_LABEL = 1       # 1 = spam, 0 = not spam (same as training data)

EXAMPLES = {
    "Prize scam": "URGENT! You have won a 1000 cash prize. Call 09061701461 to claim now. T&Cs apply.",
    "Bank phishing": "Your bank account is suspended. Verify your password immediately at http://secure-bank-login.example or it will be closed.",
    "Job offer": "Dear Candidate, we are pleased to inform you that you have been selected for the position of Software Developer Intern at TechNova Solutions. Duration: 6 months. Location: Bengaluru. Please reply with your confirmation by 25 November 2026.",
    "Team message": "Hi team, the sprint review is moved to Friday 3 pm. Please update your tasks on the board and share any blockers before the meeting.",
}

# ============================================================
# STYLE
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root{
  --bg:#070A12; --panel:#0E1424; --panel2:#121A2F; --line:#1E2A44;
  --text:#EAF0FF; --muted:#8D9BB8;
  --blue:#4C8DFF; --safe:#2DD4A7; --warn:#FFB020; --spam:#FF4D6D;
}
html, body, .stApp, [class*="css"]{ font-family:'Manrope', system-ui, 'Segoe UI', sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
#MainMenu, footer{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:1080px; padding-top:1.6rem; padding-bottom:3rem; }
section[data-testid="stSidebar"]{ background:#0A0F1C; border-right:1px solid var(--line); }

/* Brand header */
.brand{ display:flex; align-items:center; gap:14px; margin-bottom:4px; }
.brand-mark{ width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
  font-size:22px; background:linear-gradient(135deg,#1c3a78,#4C8DFF); }
.brand-name{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:30px; letter-spacing:-0.5px; }
.brand-sub{ color:var(--muted); font-size:15px; margin:0 0 18px 56px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ background:transparent; color:var(--muted); font-weight:600; padding:10px 16px; }
.stTabs [aria-selected="true"]{ color:var(--text); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--blue); }

/* Inputs */
.stTextArea textarea{ background:var(--panel) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:12px !important; font-size:15px !important; line-height:1.55; }
.stTextArea textarea:focus{ border-color:var(--blue) !important; box-shadow:0 0 0 3px #4C8DFF33 !important; }
.stTextArea label, .stSelectbox label, .stFileUploader label, .stSlider label{ color:var(--muted) !important; font-weight:600; }

/* Buttons */
.stButton > button, .stDownloadButton > button{ border-radius:10px; font-weight:700; border:1px solid var(--line);
  background:var(--panel2); color:var(--text); transition:border-color .15s, background .15s; }
.stButton > button:hover, .stDownloadButton > button:hover{ border-color:var(--blue); color:#fff; }
.stButton > button[kind="primary"]{ background:var(--blue); border-color:var(--blue); color:#06101F; height:48px; font-size:16px; }
.stButton > button[kind="primary"]:hover{ background:#6AA2FF; border-color:#6AA2FF; color:#06101F; }
.stButton > button:focus-visible{ outline:2px solid var(--blue); outline-offset:2px; }

/* Verdict card: the one memorable element */
.verdict{ border-radius:18px; padding:26px 28px 22px; background:var(--panel); border:1px solid var(--line); margin:6px 0 18px; }
.verdict.safe{ border-color:#2DD4A766; background:linear-gradient(180deg,#0C2220 0%, var(--panel) 70%); }
.verdict.warn{ border-color:#FFB02066; background:linear-gradient(180deg,#2A2008 0%, var(--panel) 70%); }
.verdict.spam{ border-color:#FF4D6D77; background:linear-gradient(180deg,#2C0F1A 0%, var(--panel) 70%); }
.v-top{ display:flex; justify-content:space-between; align-items:flex-end; gap:20px; flex-wrap:wrap; }
.v-title{ font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:700; letter-spacing:-0.6px; }
.safe .v-title{ color:var(--safe); } .warn .v-title{ color:var(--warn); } .spam .v-title{ color:var(--spam); }
.v-sub{ color:var(--muted); margin-top:4px; max-width:520px; }
.v-pct{ font-family:'Space Grotesk',sans-serif; font-size:64px; font-weight:700; line-height:1; text-align:right; }
.v-pct span{ font-size:28px; color:var(--muted); margin-left:2px; }
.v-cap{ color:var(--muted); font-size:13px; text-align:right; margin-top:2px; }
.meter{ position:relative; height:10px; border-radius:99px; background:#1A2440; margin-top:22px; }
.meter-fill{ height:100%; border-radius:99px; }
.safe .meter-fill{ background:var(--safe); } .warn .meter-fill{ background:var(--warn); } .spam .meter-fill{ background:var(--spam); }
.meter-tick{ position:absolute; top:-5px; width:2px; height:20px; background:#EAF0FF; opacity:.75; border-radius:2px; }
.meter-labels{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-top:8px; }

/* Signal chips */
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 14px; }
.chip{ padding:6px 12px; border-radius:99px; font-size:13px; font-weight:600; border:1px solid var(--line); background:var(--panel2); color:var(--text); }
.chip.high{ border-color:#FF4D6D77; color:#FF9DB0; }
.chip.mid{ border-color:#FFB02066; color:#FFD07A; }
.chip.info{ color:var(--muted); }
.section-label{ color:var(--muted); font-size:14px; font-weight:600; margin:18px 0 4px; }

/* Metrics */
div[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 18px; }
div[data-testid="stMetricLabel"] p{ color:var(--muted); }

.note{ color:var(--muted); font-size:13px; margin-top:28px; text-align:center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# MODEL (loaded once, cached)
# ============================================================
@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_model():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_PATH}\n"
            "Copy your trained 'distilbert_spam' folder into the 'models' folder, "
            "or run:  python train_distilbert.py"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    mdl.to(device).eval()
    return tok, mdl, device


try:
    tokenizer, model, DEVICE = load_model()
except Exception as e:
    st.error("Unable to load the DistilBERT model.")
    st.code(str(e))
    st.stop()


def _clean_inputs(enc):
    enc.pop("token_type_ids", None)
    enc.pop("overflow_to_sample_mapping", None)
    return enc


@st.cache_data(show_spinner=False, max_entries=256)
def spam_probability(text: str):
    """Spam probability for one message. Long emails are split into overlapping
    128-token windows (up to MAX_CHUNKS) and the probabilities are averaged,
    so the whole email is read, not just the first 128 tokens."""
    enc = tokenizer(
        text, truncation=True, max_length=MAX_LEN, stride=STRIDE,
        return_overflowing_tokens=True, padding=True, return_tensors="pt",
    )
    enc = _clean_inputs(enc)
    n_chunks = min(enc["input_ids"].shape[0], MAX_CHUNKS)
    enc = {k: v[:MAX_CHUNKS].to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits, dim=1)[:, SPAM_LABEL]
    return float(probs.mean().item()), int(n_chunks)


def batch_probabilities(texts, progress=None):
    """Fast batched scoring (first 128 tokens per message) for CSV files."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = [str(t) for t in texts[i:i + BATCH_SIZE]]
        enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in _clean_inputs(enc).items()}
        with torch.inference_mode():
            out += torch.softmax(model(**enc).logits, dim=1)[:, SPAM_LABEL].tolist()
        if progress:
            progress.progress(min(1.0, (i + BATCH_SIZE) / len(texts)))
    return out


# ============================================================
# HELPERS
# ============================================================
URGENCY = ["urgent", "immediately", "act now", "limited time", "expires", "last chance",
           "final notice", "hurry", "today only", "asap"]
PRIZE = ["free", "winner", "you won", "won ", "prize", "lottery", "reward", "claim", "cash", "bonus", "gift", "congratulations"]
CREDENTIAL = ["password", "verify", "suspended", "login", "otp", "kyc", "bank account", "confirm your", "security alert"]


def find_signals(text: str):
    low = text.lower()
    sig = []
    urls = re.findall(r"(https?://\S+|www\.\S+)", text)
    if urls:
        sig.append((f"{len(urls)} link{'s' if len(urls) > 1 else ''}", "mid"))
    if re.search(r"\b\+?\d[\d\s\-]{8,}\d\b", text):
        sig.append(("Phone number", "mid"))
    if re.search(r"[$£€₹]\s?\d|\b\d[\d,\.]*\s?(usd|inr|rs|dollars|pounds)\b", low):
        sig.append(("Money amount", "mid"))
    found = [w for w in URGENCY if w in low]
    if found:
        sig.append(("Urgency: " + found[0].strip(), "high"))
    found = [w for w in PRIZE if w in low]
    if found:
        sig.append(("Prize/offer wording: " + found[0].strip(), "high"))
    found = [w for w in CREDENTIAL if w in low]
    if found:
        sig.append(("Account/credential wording: " + found[0].strip(), "high"))
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.3:
        sig.append(("Lots of CAPITALS", "mid"))
    if text.count("!") >= 3:
        sig.append(("Many exclamation marks", "mid"))
    return sig


def verdict_html(p: float, thr: float) -> str:
    if p >= thr:
        cls, title, sub = "spam", "Spam detected", "This message looks like spam or a scam. Don't click links or reply."
    elif p >= max(thr - 0.25, 0.15):
        cls, title, sub = "warn", "Suspicious", "Some spam-like patterns were found. Check the sender before acting."
    else:
        cls, title, sub = "safe", "Looks legitimate", "No strong spam patterns were found."
    pct = p * 100
    return (
        f'<div class="verdict {cls}">'
        f'<div class="v-top"><div><div class="v-title">{title}</div><div class="v-sub">{sub}</div></div>'
        f'<div><div class="v-pct">{pct:.1f}<span>%</span></div><div class="v-cap">spam probability</div></div></div>'
        f'<div class="meter"><div class="meter-fill" style="width:{max(pct, 1.5):.1f}%"></div>'
        f'<div class="meter-tick" style="left:{thr * 100:.0f}%"></div></div>'
        f'<div class="meter-labels"><span>Legitimate</span><span>Decision threshold {thr * 100:.0f}%</span><span>Spam</span></div>'
        f'</div>'
    ), cls


def label_for(p: float, thr: float) -> str:
    if p >= thr:
        return "Spam"
    if p >= max(thr - 0.25, 0.15):
        return "Suspicious"
    return "Legitimate"


def set_example(name):
    st.session_state["email_text"] = EXAMPLES[name]
    st.session_state.pop("last", None)


def clear_text():
    st.session_state["email_text"] = ""
    st.session_state.pop("last", None)


if "history" not in st.session_state:
    st.session_state["history"] = []
if "email_text" not in st.session_state:
    st.session_state["email_text"] = ""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider(
        "Spam threshold", 0.30, 0.90, 0.50, 0.05,
        help="Messages at or above this spam probability are flagged as spam. "
             "Lower = stricter (catches more spam, more false alarms).",
    )
    st.markdown("---")
    st.markdown("### Model")
    st.caption("DistilBERT, fine-tuned for spam detection")
    st.caption(f"Running on: **{'GPU' if DEVICE == 'cuda' else 'CPU'}**")
    st.caption(f"Reads up to {MAX_CHUNKS} x {MAX_LEN} tokens per email")
    if st.button("Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="brand"><div class="brand-mark">🛡️</div><div class="brand-name">NeuroShield</div></div>'
    '<div class="brand-sub">Paste a message and find out if it is spam.</div>',
    unsafe_allow_html=True,
)

tab_analyze, tab_batch, tab_history, tab_about = st.tabs(["Analyze", "Check a CSV file", "History", "About"])

# ============================================================
# TAB 1: ANALYZE
# ============================================================
with tab_analyze:
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, key=f"ex_{name}", on_click=set_example, args=(name,), use_container_width=True)

    text = st.text_area(
        "Message", key="email_text", height=210,
        placeholder="Paste an email or text message here...",
    )

    b1, b2 = st.columns([4, 1])
    analyze = b1.button("Analyze message", type="primary", use_container_width=True)
    b2.button("Clear", on_click=clear_text, use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Paste a message first, then select Analyze message.")
            st.session_state.pop("last", None)
        else:
            t0 = time.perf_counter()
            with st.spinner("Analyzing..."):
                prob, n_chunks = spam_probability(text.strip())
            ms = (time.perf_counter() - t0) * 1000
            st.session_state["last"] = {"text": text.strip(), "prob": prob, "chunks": n_chunks, "ms": ms}
            st.session_state["history"].insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Message": text.strip()[:90].replace("\n", " ") + ("..." if len(text.strip()) > 90 else ""),
                "Spam probability %": round(prob * 100, 1),
                "Verdict": label_for(prob, threshold),
            })
            st.session_state["history"] = st.session_state["history"][:100]

    last = st.session_state.get("last")
    if last:
        card, cls = verdict_html(last["prob"], threshold)
        st.markdown(card, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Analysis time", f"{last['ms']:.0f} ms")
        m2.metric("Words", len(last["text"].split()))
        m3.metric("Sections read", last["chunks"])

        signals = find_signals(last["text"])
        st.markdown('<div class="section-label">Patterns found in the text</div>', unsafe_allow_html=True)
        if signals:
            chips = "".join(f'<span class="chip {sev}">{html.escape(lbl)}</span>' for lbl, sev in signals)
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chips"><span class="chip info">No common spam patterns found</span></div>',
                        unsafe_allow_html=True)
        st.caption("These patterns are simple text checks shown for context. The verdict comes from the DistilBERT model.")

# ============================================================
# TAB 2: CSV BATCH
# ============================================================
with tab_batch:
    st.write("Upload a CSV file and NeuroShield scores every row.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="latin-1")
        except Exception as e:
            st.error("Could not read this CSV file.")
            st.code(str(e))
            df = None
        if df is not None and len(df.columns):
            guess = next((c for c in df.columns if str(c).lower() in
                          ["text", "raw_text", "message", "email", "body", "content"]), df.columns[0])
            col_name = st.selectbox("Column that contains the messages", list(df.columns),
                                    index=list(df.columns).index(guess))
            st.caption(f"{len(df):,} rows found.")
            if len(df) > 5000:
                st.warning("Only the first 5,000 rows will be scored.")
            if st.button("Score all rows", type="primary"):
                work = df.head(5000).copy()
                work[col_name] = work[col_name].fillna("").astype(str)
                bar = st.progress(0.0)
                t0 = time.perf_counter()
                probs = batch_probabilities(work[col_name].tolist(), bar)
                bar.empty()
                work["spam_probability"] = [round(p, 4) for p in probs]
                work["verdict"] = [label_for(p, threshold) for p in probs]
                n_spam = int((work["verdict"] == "Spam").sum())
                s1, s2, s3 = st.columns(3)
                s1.metric("Messages scored", f"{len(work):,}")
                s2.metric("Flagged as spam", f"{n_spam:,}")
                s3.metric("Time", f"{time.perf_counter() - t0:.1f} s")
                st.dataframe(work, use_container_width=True, height=360)
                st.download_button("Download results (CSV)", work.to_csv(index=False).encode("utf-8"),
                                   "neuroshield_results.csv", "text/csv")

# ============================================================
# TAB 3: HISTORY
# ============================================================
with tab_history:
    if st.session_state["history"]:
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                           "neuroshield_history.csv", "text/csv")
    else:
        st.info("Nothing here yet. Analyze a message and it will appear in this list.")

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tab_about:
    st.markdown(
        """
**How it works.** NeuroShield uses DistilBERT, a compact language model, fine-tuned on labelled spam and
non-spam messages. It returns a spam probability between 0% and 100%. Long emails are split into overlapping
sections and the results are averaged.

**Threshold.** Use the slider in the sidebar to make the filter stricter or more relaxed.

**Limitations.**
- The model was trained mostly on short SMS-style messages, so long, formal emails can be less reliable.
- It does not check sender addresses, attachments, or whether a link is actually dangerous.
- Treat the result as a helpful second opinion, not a guarantee.
        """
    )

st.markdown('<div class="note">NeuroShield AI &nbsp;|&nbsp; Powered by DistilBERT</div>', unsafe_allow_html=True)

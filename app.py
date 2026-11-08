import os
import streamlit as st
import pandas as pd, numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from my_models import *                     # our own classes (needed to unpickle the models)

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon=":material/shopping_cart:",
                   layout="wide", initial_sidebar_state="expanded")

# ===================== Content =====================
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
SAMPLE_FILE = "test_data.csv"

ICONS = {   # inline SVG icons (no internet needed)
    "cart": '<circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/>',
    "cap": '<path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/><path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "layers": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
}
def ic(name, size=18):
    return (f"<svg class='ic' xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' "
            f"fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            f"{ICONS[name]}</svg>")

FEATURES = pd.DataFrame([
    ("Administrative", "Account / admin pages visited"),
    ("Administrative_Duration", "Seconds spent on admin pages"),
    ("Informational", "Info pages visited (about, help)"),
    ("Informational_Duration", "Seconds spent on info pages"),
    ("ProductRelated", "Product pages visited"),
    ("ProductRelated_Duration", "Seconds spent on product pages"),
    ("BounceRates", "Share of one-page visits"),
    ("ExitRates", "Share of page views that ended the session"),
    ("PageValues", "Average page value before a purchase"),
    ("SpecialDay", "Closeness to a special day (0 to 1)"),
    ("Month", "Month of visit (1 to 12)"),
    ("OperatingSystems / Browser / Region / TrafficType", "Coded IDs"),
    ("Weekend", "1 if weekend visit"),
    ("VisitorType_* (3 columns)", "New / Returning / Other (one-hot)"),
], columns=["Feature", "Meaning"])

MODEL_INFO = {
    "Logistic Regression": dict(idea="Weighted sum of features passed through a sigmoid to give a probability.",
        settings="lr 0.1, 2000 iterations", result="Acc 0.8932 | AUC 0.8777 | F1 0.5556 | MCC 0.5299"),
    "Decision Tree": dict(idea="Yes/no questions on features, split chosen by Gini impurity.",
        settings="max depth 6, min 5 samples", result="Acc 0.8765 | AUC 0.8859 | F1 0.5542 | MCC 0.4885"),
    "KNN": dict(idea="Looks at the 7 closest training sessions and takes a majority vote.",
        settings="k = 7, Euclidean distance", result="Acc 0.8815 | AUC 0.7981 | F1 0.4662 | MCC 0.4578"),
    "Naive Bayes": dict(idea="Bayes theorem with Gaussian likelihoods, assuming independent features.",
        settings="Gaussian, smoothing 1e-6", result="Acc 0.7546 | AUC 0.8137 | F1 0.4615 | MCC 0.3524"),
    "Support Vector Machine": dict(idea="Finds the widest-margin line between Buy and No Buy (hinge loss).",
        settings="linear, lambda 0.01, 2000 iterations", result="Acc 0.8948 | AUC 0.8431 | F1 0.5594 | MCC 0.5373"),
}

# ===================== Helpers =====================
@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(p) for n, p in MODEL_FILES.items()}
    return models, joblib.load("model/scaler.joblib"), joblib.load("model/feature_columns.joblib")

def predict_chunks(model, X, size=100):
    prob = np.concatenate([model.predict_proba(X[i:i + size]) for i in range(0, len(X), size)])
    pred = model.predict(X) if isinstance(model, MySVM) else (prob >= 0.5).astype(int)
    return pred, prob

@st.cache_data
def run_all(_models, X):
    return {n: predict_chunks(m, X) for n, m in _models.items()}

def report_table(y, pred):
    tn, fp, fn, tp = confusion(y, pred)
    rows = []
    for name, a, b, c, sup in [("No Buy (0)", tn, fn, fp, tn + fp), ("Buy (1)", tp, fp, fn, tp + fn)]:
        p = a / (a + b) if a + b else 0
        r = a / (a + c) if a + c else 0
        f = 2 * p * r / (p + r) if p + r else 0
        rows.append({"Class": name, "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f, 4), "Support": sup})
    return pd.DataFrame(rows)

def draw_cm(ax, y, pred, cmap, big=True):
    tn, fp, fn, tp = confusion(y, pred)
    cm = np.array([[tn, fp], [fn, tp]])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=20 if big else 11, fontweight="bold", color="#14231a")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Buy", "Buy"]); ax.set_yticklabels(["No Buy", "Buy"])
    ax.set_xlabel("Predicted", fontsize=13 if big else 9); ax.set_ylabel("Actual", fontsize=13 if big else 9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=4); ax.tick_params(which="minor", length=0)
OBS = {
    "Logistic Regression": (
        "Accuracy 0.8932 and AUC 0.8777, so it ranks buyers above non-buyers well. Precision is high (0.7843) "
        "with only 11 false alarms, but recall is just 0.4301: 53 of the 93 real buyers were missed. "
        "MCC 0.5299 is the second best. A reliable, conservative linear model."),
    "Decision Tree": (
        "Best AUC (0.8859) and it catches more buyers than Logistic Regression (recall 0.4946, 46 of 93). "
        "The price is 27 false alarms, so precision falls to 0.6301 and MCC to 0.4885. "
        "Depth 6 captures non-linear patterns but also adds some noise."),
    "KNN": (
        "Weakest ranking (AUC 0.7981) and lowest recall (0.3333): only 31 of 93 buyers found. "
        "Only about 15% of rows are buyers, so most of the 7 nearest neighbours are No Buy and the vote "
        "almost always says No Buy. It has the fewest false alarms (9), but misses the most buyers."),
    "Naive Bayes": (
        "Highest recall (0.6774, 63 of 93 buyers caught) but the lowest accuracy (0.7546), precision (0.3500) "
        "and MCC (0.3524) because of 117 false alarms. Page counts and durations are correlated, which breaks "
        "the independence assumption. Useful only when missing a buyer is costlier than a wasted offer."),
    "Support Vector Machine": (
        "Best Accuracy (0.8948), F1 (0.5594) and MCC (0.5373), with precision 0.8000 and only 10 false alarms. "
        "The wide-margin linear boundary generalises well on this scaled data. "
        "Recall (0.4301) is still modest, so many buyers are missed, a side effect of the 85/15 class imbalance."),
}
WINNER = ("Support Vector Machine. It leads on Accuracy, F1 and MCC, and MCC is the most trustworthy metric on "
          "imbalanced data. Logistic Regression is very close behind, and Decision Tree has the best AUC. "
          "If the goal is to reach as many buyers as possible and false alarms are cheap, Naive Bayes "
          "(highest recall) would be the better pick.")

def results_table():
    try:
        return pd.read_csv("model/results.csv")
    except Exception:
        return pd.DataFrame()

def html_table(headers, rows, win=None):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = ""
    for r in rows:
        cls = " class='win'" if win and r[0] == win else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f"<table class='rep'><tr>{h}</tr>{body}</table>"

models, scaler, feature_cols = load_artifacts()

# ===================== Sidebar: identity + settings =====================
st.sidebar.markdown(":material/school: **Machine Learning (S1-26_AIMLZG565)**")
st.sidebar.caption("Assignment 2  \nStudent ID: 2026AK05030  \nStudent Name: Mudigonda Neelima")
st.sidebar.title(":material/tune: Controls")

with st.sidebar.expander(":material/palette: Settings", expanded=False):
    mode = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)
    accent = st.color_picker("Accent colour", "#166534")
dark = mode == "Dark"

if dark:
    T = dict(bg="#0a1a10", card="#12301f", text="#f0fdf4", sub="#a7c4b2", line="#2f6b48")
else:
    T = dict(bg="#ffffff", card="#f3faf5", text="#14231a", sub="#5b6b61", line="#cfe5d6")

ACC_MAP = LinearSegmentedColormap.from_list("acc", ["#ffffff", "#86c9a0"])   # white to soft green
plt.rcParams.update({"figure.facecolor": T["card"], "axes.facecolor": T["card"],
                     "axes.edgecolor": T["line"], "axes.labelcolor": T["text"],
                     "xtick.color": T["text"], "ytick.color": T["text"], "text.color": T["text"],
                     "legend.facecolor": T["card"], "legend.edgecolor": T["line"], "grid.color": T["line"]})

# ===================== CSS =====================
CSS = """
<style>
:root{--bg:%BG%;--card:%CARD%;--text:%TEXT%;--sub:%SUB%;--line:%LINE%;--accent:%ACC%;}
.stApp{background:var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stCaptionContainer"]{color:var(--text);}
section[data-testid="stSidebar"]{background:var(--card);border-right:3px solid var(--accent);}
.ic{vertical-align:-3px;margin-right:8px;}
[data-baseweb="popover"] ul{background:var(--card) !important;}
[data-baseweb="popover"] li{color:var(--text) !important;}

/* hero: white top, wave, dark bottom */
.hero{background:#fff;border-radius:18px;overflow:hidden;margin-bottom:20px;box-shadow:0 8px 24px rgba(0,0,0,.14);}
.hero .top{padding:26px 34px 0 34px;background:#fff;}
.hero .top, .hero .top *{color:#14231a !important;}
.hero .course{display:inline-block;font-weight:700;margin-bottom:8px;padding:6px 16px;border-radius:10px;background:var(--accent);}
.hero .top .course, .hero .top .course *{color:#fff !important;}
.hero .student{font-weight:600;margin:4px 0 8px 2px;font-size:.98rem;}
.hero h1{margin:6px 0 0 0;font-size:2.2rem;letter-spacing:.5px;}
.hero p{margin:6px 0 0 0;font-size:1.02rem;max-width:760px;}
.hero .wave{display:block;width:100%;height:90px;margin-bottom:-1px;}
.hero .bottom{background:#1f2937;padding:0 34px 20px 34px;}
.hero .bottom, .hero .bottom *{color:#fff !important;}
.hero .pill{display:inline-block;padding:5px 14px;margin:3px 6px 3px 0;border-radius:999px;
  background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.45);font-size:.88rem;}

[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--accent);
  border-radius:12px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:1.6rem;font-weight:700;}

.card{border:1px solid var(--line);border-left:6px solid var(--accent);background:var(--card);
  border-radius:12px;padding:14px 18px;margin-bottom:14px;}
.card h4{margin:0 0 4px 0;}
.lbl{color:var(--accent) !important;}
.winner{border-left:6px solid var(--accent);background:var(--card);padding:12px 18px;border-radius:10px;margin-top:10px;}
.howto{border:2px dashed var(--accent);background:var(--card);border-radius:14px;padding:14px 22px;margin:6px 0 16px 0;}
.howto h4{margin:0 0 6px 0;}
.howto ol{margin:0 0 0 20px;padding:0;line-height:1.8;}
.howto .arrow{display:inline-block;padding:0 9px;border-radius:6px;background:var(--accent);color:#fff;font-weight:700;}

/* sidebar widgets: force readable colours in any base theme */
[data-testid="stDownloadButton"] button, [data-testid="stFileUploaderDropzone"] button{
  background:var(--accent) !important;border:none !important;}
[data-testid="stDownloadButton"] button *, [data-testid="stFileUploaderDropzone"] button *{
  color:#fff !important;fill:#fff !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--bg) !important;border:1px dashed var(--accent) !important;}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div{color:var(--sub) !important;}
[data-testid="stFileUploaderDropzone"] button span{color:#fff !important;}
[data-baseweb="select"] > div{background:var(--bg) !important;border:1px solid var(--accent) !important;}
[data-baseweb="select"] *{color:var(--text) !important;}
[data-baseweb="select"] svg{fill:var(--accent) !important;}
[data-baseweb="checkbox"] span:first-child{background:var(--bg) !important;border:2px solid var(--accent) !important;}
[data-baseweb="checkbox"]:has(input:checked) span:first-child{background:var(--accent) !important;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{color:var(--text);}
/* expander header: same colours in normal, hover, focus and open states */
[data-testid="stExpander"] details{border:1px solid var(--line) !important;border-radius:10px;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active, [data-testid="stExpander"] details[open] > summary{
  background:var(--card) !important;color:var(--text) !important;}
[data-testid="stExpander"] summary:hover{background:color-mix(in srgb, var(--accent) 22%, var(--card)) !important;}
[data-testid="stExpander"] summary *, [data-testid="stExpander"] summary:hover *,
[data-testid="stExpander"] summary:focus *{color:var(--text) !important;fill:var(--text) !important;}
[data-testid="stExpanderDetails"]{background:var(--card) !important;color:var(--text) !important;}
[data-testid="stColorPicker"] label{color:var(--text) !important;}
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"] *{
  color:var(--accent) !important;fill:var(--accent) !important;}
[data-testid="stSidebarContent"]{padding-bottom:80px;}

/* tabs: fill rises from the bottom on hover, solid fill when selected */
[data-baseweb="tab-list"]{gap:10px !important;border-bottom:none !important;padding-bottom:10px !important;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{
  display:none !important;height:0 !important;background:transparent !important;}
button[role="tab"]{
  height:auto !important;padding:10px 20px !important;border-radius:12px !important;
  border:2px solid var(--accent) !important;
  background-color:var(--card) !important;
  background-image:linear-gradient(var(--accent),var(--accent)) !important;
  background-repeat:no-repeat !important;background-position:bottom !important;
  background-size:100% 0% !important;
  transition:background-size .8s ease, transform .25s, box-shadow .3s !important;}
button[role="tab"]:hover{background-size:100% 100% !important;
  transform:translateY(-3px);box-shadow:0 6px 16px rgba(0,0,0,.18);}
button[role="tab"][aria-selected="true"]{
  background-color:var(--accent) !important;background-size:100% 100% !important;}
button[role="tab"] p, button[role="tab"] span{
  color:var(--text) !important;font-weight:600 !important;transition:color .4s .15s;}
button[role="tab"]:hover p, button[role="tab"]:hover span,
button[role="tab"][aria-selected="true"] p, button[role="tab"][aria-selected="true"] span{
  color:#ffffff !important;}

/* fixed footer */
.app-footer{position:fixed;left:0;right:0;bottom:0;z-index:1000000;display:flex;flex-wrap:wrap;
  justify-content:center;align-items:center;gap:6px 18px;padding:10px 16px;
  background:var(--accent);box-shadow:0 -4px 14px rgba(0,0,0,.18);font-size:.88rem;font-weight:600;}
.app-footer, .app-footer *{color:#fff !important;}
.app-footer .item{display:inline-flex;align-items:center;}
.app-footer .item .ic{margin-right:6px;}
.app-footer .sep{width:1px;height:16px;background:rgba(255,255,255,.55);}
.block-container{padding-bottom:80px !important;}
.rep{width:100%;border-collapse:collapse;margin:8px 0 18px 0;font-size:.95rem;}
.rep th{background:var(--accent);color:#fff !important;text-align:left;padding:10px 12px;border:1px solid var(--accent);}
.rep td{padding:10px 12px;border:1px solid var(--line);background:var(--card);vertical-align:top;}
.rep td:first-child{font-weight:700;white-space:nowrap;}
.rep tr.win td{background:var(--bg);border-left:5px solid var(--accent);}
%DARKFIX%
</style>
"""
darkfix = ('[data-testid="stDataFrame"]{filter:invert(.9) hue-rotate(180deg) saturate(.6);}'
           '[data-testid="stDataFrame"] canvas{border-radius:8px;}') if dark else ""
for k, v in {"%BG%": T["bg"], "%CARD%": T["card"], "%TEXT%": T["text"], "%SUB%": T["sub"],
             "%LINE%": T["line"], "%ACC%": accent, "%DARKFIX%": darkfix}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)

# ===================== Header banner =====================
st.markdown(f"""
<div class="hero">
<div class="top">
<div class="course">{ic('cap')}Machine Learning (S1-26_AIMLZG565) &nbsp;·&nbsp; Assignment 2</div>
<div class="student">{ic('user')}Student ID: 2026AK05030 &nbsp;·&nbsp; Student Name: Mudigonda Neelima</div>
<h1>{ic('cart', 34)}Online Shopper Purchase Predictor</h1>
<p>Will this website visitor buy something? Five classifiers built from scratch with NumPy. No retraining.</p>
</div>
<svg class="wave" viewBox="0 0 1440 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
<path d="M0,35 C400,130 950,130 1440,5 L1440,120 L0,120 Z" style="fill:{accent}"/>
<path d="M0,70 C420,140 980,140 1440,40 L1440,120 L0,120 Z" fill="#1f2937"/>
</svg>
<div class="bottom">
<span class="pill">{ic('target', 16)}Binary classification</span>
<span class="pill">{ic('layers', 16)}{len(feature_cols)} input features</span>
<span class="pill">{ic('flag', 16)}Target: class (0 = No Buy, 1 = Buy)</span>
<span class="pill">{ic('cpu', 16)}5 models, 6 metrics</span>
<span class="pill">{ic('db', 16)}UCI Online Shoppers Purchasing Intention</span>
</div>
</div>
""", unsafe_allow_html=True)

# ===================== Sidebar: data + model =====================
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 1: Get test data**")
if os.path.exists(SAMPLE_FILE):
    with open(SAMPLE_FILE, "rb") as f:
        st.sidebar.download_button("Download sample test_data.csv", f.read(), file_name="test_data.csv",
                                   mime="text/csv", icon=":material/download:")
file = st.sidebar.file_uploader("Step 2: Upload test CSV", type="csv")
use_sample = st.sidebar.checkbox("Or use bundled test_data.csv", value=False,
                                 disabled=not os.path.exists(SAMPLE_FILE))
choice = st.sidebar.selectbox("Step 3: Select model", list(MODEL_FILES.keys()))
st.sidebar.caption(f"CSV needs {len(feature_cols)} feature columns + target `class` (0 = No Buy, 1 = Buy).")

HOWTO = """
<div class='howto'>
<h4>How to use</h4>
<ol>
<li>Open the sidebar with the <span class='arrow'>&gt;</span> arrow at the top-left (skip if already open).</li>
<li>Click <b>Download sample test_data.csv</b>, then upload it under <b>Step 2</b> (or tick the bundled-data box).</li>
<li>Pick a model in <b>Step 3</b>.</li>
<li>Open the <b>Confusion Matrix</b> tab for metrics, or <b>Compare</b> to see all five models.</li>
</ol>
</div>
"""

source = file if file is not None else (SAMPLE_FILE if use_sample else None)
ready = source is not None
if ready:
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_cols + ["class"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        st.stop()
    X = scaler.transform(df[feature_cols].to_numpy(float))
    y = df["class"].to_numpy().astype(int)
    with st.spinner("Running all models..."):
        outputs = run_all(models, X)
    pred, prob = outputs[choice]
    m = all_metrics(y, pred, prob)
    tn, fp, fn, tp = confusion(y, pred)
    with st.expander("How to use", icon=":material/menu_book:"):
        st.markdown(HOWTO, unsafe_allow_html=True)
else:
    st.markdown(HOWTO, unsafe_allow_html=True)

tabs = st.tabs([":material/home: About", ":material/psychology: Models",
                ":material/grid_on: Confusion Matrix", ":material/emoji_events: Compare",
                ":material/insights: Performance"])
NEED = ":material/info:"

# ---- About ----
with tabs[0]:
    st.markdown("### Problem statement")
    st.write("Predict from a visitor's browsing behaviour whether the session ends in a purchase "
             "(**Buy = 1**) or not (**No Buy = 0**), so a shop can show offers at the right moment.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original sessions", "12,330"); c2.metric("Rows used", "3,000")
    c3.metric("Input features", len(feature_cols)); c4.metric("Buyers", "about 15%")
    st.caption("Source: UCI Online Shoppers Purchasing Intention. Duplicates removed, stratified 3,000-row sample, "
               "80/20 split (2,401 train / 599 test), standardised features, VisitorType one-hot encoded.")
    st.markdown("### Input features")
    st.dataframe(FEATURES, hide_index=True)

# ---- Models ----
with tabs[1]:
    st.markdown("### Five models built from scratch (NumPy, no scikit-learn)")
    cl, cr = st.columns(2)
    for k, (n, i) in enumerate(MODEL_INFO.items()):
        with (cl if k % 2 == 0 else cr):
            st.markdown(f"<div class='card'><h4>{ic('cpu', 20)}{n}</h4>{i['idea']}<br>"
                        f"<b class='lbl'>Settings:</b> {i['settings']}<br>"
                        f"<b class='lbl'>Test result:</b> {i['result']}</div>", unsafe_allow_html=True)

# ---- Confusion Matrix (main focus) ----
with tabs[2]:
    if not ready:
        st.warning("Provide test data from the sidebar to see results.", icon=NEED)
    else:
        st.markdown(f"### {choice}")
        for c, k in zip(st.columns(6), METRICS):
            c.metric(k, f"{m[k]:.4f}")
        left, right = st.columns([3, 2])
        with left:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            draw_cm(ax, y, pred, ACC_MAP, big=True)
            ax.set_title(f"Confusion matrix ({len(y)} test rows)", fontsize=14, fontweight="bold", pad=12)
            st.pyplot(fig)
        with right:
            st.markdown("#### Classification report")
            st.dataframe(report_table(y, pred), hide_index=True)
            st.markdown("#### How to read it")
            st.write(f"Buyers caught (TP): **{tp}**")
            st.write(f"Buyers missed (FN): **{fn}**")
            st.write(f"False alarms (FP): **{fp}**")
            st.write(f"Correct No Buy (TN): **{tn}**")
            out = df.copy()
            out["predicted_class"] = pred
            out["buy_probability"] = np.round(prob, 4)
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               file_name=f"predictions_{choice.replace(' ', '_').lower()}.csv",
                               mime="text/csv", icon=":material/download:")

# ---- Compare ----
with tabs[3]:
    if not ready:
        st.warning("Provide test data from the sidebar to compare models.", icon=NEED)
    else:
        rows = [{"ML Model Name": n, **{k: round(v, 4) for k, v in all_metrics(y, p_, pr_).items()}}
                for n, (p_, pr_) in outputs.items()]
        comp = pd.DataFrame(rows)
        st.markdown("#### Metrics table")
        st.dataframe(comp, hide_index=True)

        st.markdown("#### Confusion matrix of every model")
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
        for ax, (n, (p_, pr_)) in zip(axes, outputs.items()):
            draw_cm(ax, y, p_, ACC_MAP, big=False)
            ax.set_title(n, fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Metric chart")
        fig, ax = plt.subplots(figsize=(11, 4))
        width, xs = 0.15, np.arange(len(METRICS))
        shades = [1.0, 0.8, 0.62, 0.46, 0.3]
        for i, n in enumerate(comp["ML Model Name"]):
            ax.bar(xs + i * width, comp.loc[comp["ML Model Name"] == n, METRICS].to_numpy().ravel(),
                   width, label=n, color=to_rgba(accent, shades[i]))
        ax.set_xticks(xs + 2 * width); ax.set_xticklabels(METRICS); ax.set_ylim(0, 1)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False)
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)

        best = comp.sort_values("MCC", ascending=False).iloc[0]
        st.markdown(f"<div class='winner'><b>Best model by MCC:</b> {best['ML Model Name']} "
                    f"(MCC {best['MCC']}, F1 {best['F1']}, Accuracy {best['Accuracy']})</div>",
                    unsafe_allow_html=True)

# ---- Performance (Table structure in the assignment) ----
with tabs[4]:
    st.markdown("### Models used: comparison table")
    rt = results_table()
    if rt.empty:
        st.info("model/results.csv not found. Keep it inside the model folder.")
    else:
        rows = [[r["ML Model Name"]] + [f"{r[k]:.4f}" for k in METRICS] for _, r in rt.iterrows()]
        st.markdown(html_table(["ML Model Name"] + METRICS, rows), unsafe_allow_html=True)
        st.caption("Metrics calculated on the 599-row test set (80/20 stratified split).")

    st.markdown("### Observations on model performance")
    rows = [[n, o] for n, o in OBS.items()] + [["Overall winner for this dataset", WINNER]]
    st.markdown(html_table(["ML Model Name", "Observation about model performance"], rows,
                           win="Overall winner for this dataset"), unsafe_allow_html=True)


# ===================== Footer =====================
st.markdown(f"""<div class='app-footer'>
<span class='item'>{ic('cap', 16)}Machine Learning (S1-26_AIMLZG565)</span><span class='sep'></span>
<span class='item'>{ic('book', 16)}Assignment 2</span><span class='sep'></span>
<span class='item'>{ic('user', 16)}Student ID: 2026AK05030</span><span class='sep'></span>
<span class='item'>{ic('user', 16)}Student Name: Mudigonda Neelima</span>
</div>""", unsafe_allow_html=True)

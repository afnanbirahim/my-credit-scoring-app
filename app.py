import os
import json
import glob
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(page_title="Micro-Credit Default Scoring", page_icon="💳", layout="wide")

# =========================
# Constants
# =========================
APP_TITLE = "💳 Micro-Credit Default Scoring"
APP_SUB   = "Hybrid LR + XGBoost • Guarded Thresholds • Top-K Policy"

LR_PATH   = "model_lr_calibrated.joblib"
XGB_PATH  = "model_xgb_calibrated.joblib"
META_PATH = "metadata.json"
ZIP_PATH  = "models_bundle.zip"

# =========================
# Diagnostics sidebar
# =========================
with st.sidebar:
    try:
        import sys, sklearn, xgboost, numpy
        st.write("Env versions:", {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": numpy.__version__,
            "joblib": joblib.__version__,
        })
    except Exception:
        pass
    st.write("Root files:", glob.glob("*"))

# =========================
# Load helpers
# =========================
def _ensure_from_zip():
    """If missing and ZIP exists, unzip models."""
    need = [LR_PATH, XGB_PATH, META_PATH]
    if all(os.path.exists(p) for p in need):
        return
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(".")
        except Exception as e:
            st.error("Failed to unzip models_bundle.zip")
            st.exception(e)

def resolve_artifact_paths():
    """Find or upload model artifacts."""
    _ensure_from_zip()
    have_all = all(os.path.exists(p) for p in [LR_PATH, XGB_PATH, META_PATH])
    if have_all:
        return LR_PATH, XGB_PATH, META_PATH

    st.warning("Models not found in repo root. Upload the three artifacts below (one-time).")
    c1, c2, c3 = st.columns(3)
    with c1:
        lr_u  = st.file_uploader("model_lr_calibrated.joblib", type=["joblib"], key="u_lr")
    with c2:
        xgb_u = st.file_uploader("model_xgb_calibrated.joblib", type=["joblib"], key="u_xgb")
    with c3:
        meta_u= st.file_uploader("metadata.json", type=["json"], key="u_meta")

    if lr_u and xgb_u and meta_u:
        try:
            open(LR_PATH,  "wb").write(lr_u.read())
            open(XGB_PATH, "wb").write(xgb_u.read())
            open(META_PATH,"wb").write(meta_u.read())
            st.success("Artifacts saved to root. Click **Rerun** (top-right).")
        except Exception as e:
            st.error("Failed to save uploaded artifacts to root.")
            st.exception(e)
        st.stop()
    else:
        st.stop()

# =========================
# Cached model loader
# =========================
@st.cache_resource
def load_models(lr_path: str, xgb_path: str, meta_path: str):
    try:
        lr_m  = joblib.load(lr_path)
        xgb_m = joblib.load(xgb_path)
    except Exception as e:
        st.error("Failed to load/unpickle models. Likely version mismatch or corrupted/LFS file.")
        st.exception(e)
        st.stop()

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        st.error("Failed to read metadata.json from root.")
        st.exception(e)
        st.stop()

    for k in ["features", "thresholds", "topk_policy", "blend_weight_xgb"]:
        if k not in meta:
            st.error(f"metadata.json is missing the key: '{k}'.")
            st.stop()

    return lr_m, xgb_m, meta

# ============= Load models =============
lr_path, xgb_path, meta_path = resolve_artifact_paths()
lr_m, xgb_m, meta = load_models(lr_path, xgb_path, meta_path)

FEATURES     = meta["features"]
CAT_COLS     = meta.get("cat_cols", [])
NUM_COLS     = meta.get("num_cols", [])
W_XGB        = float(meta["blend_weight_xgb"])
THRESHOLD    = float(meta["thresholds"]["hybrid"])
REJECT_PCT   = float(meta["topk_policy"]["reject_pct"])
REVIEW_PCT   = float(meta["topk_policy"]["review_next_pct"])
REVIEW_FLOOR = max(0.6 * THRESHOLD, 0.05)

# Fix known numeric fields
NUMERIC_FORCE = {
    "How many years the member is staying at the area",
    "Number of children",
    "Number of children going to school",
    "Family Income in Taka",
    "Total Asset Value of Family in Taka",
    "Total Savings",
    "Loan Amount",
    "Installment Amount",
    "Interest Rate",
    "Number of Installment",
}
CAT_COLS = [c for c in CAT_COLS if c not in NUMERIC_FORCE]
for c in NUMERIC_FORCE:
    if c not in NUM_COLS and c in FEATURES:
        NUM_COLS.append(c)

# =========================
# Scoring helpers
# =========================
def p_default_hybrid(df_in: pd.DataFrame) -> np.ndarray:
    X = df_in.copy()
    for c in FEATURES:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURES]
    p_lr  = lr_m.predict_proba(X)[:, 1]
    p_xgb = xgb_m.predict_proba(X)[:, 1]
    return (1 - W_XGB) * p_lr + W_XGB * p_xgb

def decide_threshold(p: np.ndarray, t: float = THRESHOLD, review_floor: float = REVIEW_FLOOR) -> np.ndarray:
    d = np.full_like(p, "approve", dtype=object)
    d[p >= review_floor] = "review"
    d[p >= t] = "reject"
    return d

def apply_topk_policy(p: np.ndarray, reject_pct: float, review_pct: float) -> np.ndarray:
    n = len(p)
    order = np.argsort(-p)
    reject_k = max(1, int(np.floor(reject_pct * n)))
    review_k = max(0, int(np.floor(review_pct * n)))
    d = np.full(n, "approve", dtype=object)
    d[order[:reject_k]] = "reject"
    d[order[reject_k:reject_k+review_k]] = "review"
    return d

def topk_capture_stats(y_true: np.ndarray, p: np.ndarray, reject_pct: float, review_pct: float):
    n = len(p)
    order = np.argsort(-p)
    reject_k = max(1, int(np.floor(reject_pct * n)))
    review_k = max(0, int(np.floor(review_pct * n)))
    idx_sel = set(order[:reject_k + review_k])
    total_defaults = int(np.sum(y_true == 1))
    captured = int(sum((i in idx_sel) and (y_true[i] == 1) for i in range(n)))
    return {
        "n": n,
        "reject_k": reject_k,
        "review_k": review_k,
        "defaults_captured": captured,
        "total_defaults": total_defaults,
        "capture_rate": (captured / total_defaults) if total_defaults > 0 else 0.0
    }

# =========================
# UI START
# =========================
st.title(APP_TITLE)
st.caption(APP_SUB)

# ---- Explanation Section ----
with st.expander("ℹ️ Model & Decision Policy Explained", expanded=False):
    st.markdown("""
### 🧮 Model Overview
This scoring system predicts the probability that a borrower may **default** on repayment.
It uses a **hybrid model** blending Logistic Regression (interpretable) and XGBoost (non-linear power).

### ⚖️ Threshold-Based Decision
- **Approve** → if `p_default < 0.45`  
- **Review** → if `0.45 ≤ p_default < 0.75`  
- **Reject** → if `p_default ≥ 0.75`

These cut-offs (thresholds) come from trained model metadata and can be fine-tuned.

### 🔝 Top-K (Percentile-Based) Policy
- Rank all borrowers by predicted risk.
- Reject the top **K₁ %** (highest-risk).
- Review the next **K₂ %**.
- Approve the rest.

### ✅ Meaning of Decisions
| Decision | Interpretation | Action |
|-----------|----------------|--------|
| **Approve** | Strong repayment potential | Proceed to disbursement |
| **Review** | Medium risk — needs manual check | Field officer verification |
| **Reject** | High risk of default | Do not approve loan |

### 🔧 Top-K Slider Settings
| Control | Meaning | Effect |
|----------|----------|--------|
| **Reject top % (K₁)** | % of most risky borrowers auto-rejected | Higher → stricter approval |
| **Review next % (K₂)** | % of next risky borrowers for manual check | Higher → more human review |

**Example:** If K₁ = 5 % and K₂ = 10 %, out of 1000 borrowers → 50 reject, 100 review, 850 approve.
""")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Inference Settings")
    st.write("These mirror your trained metadata.")
    st.metric("Hybrid threshold (reject ≥ t)", f"{THRESHOLD:.3f}")
    st.metric("Review floor (≥ 0.6·t)", f"{REVIEW_FLOOR:.3f}")
    st.metric("Blend weight (XGB)", f"{W_XGB:.2f}")

    st.markdown("""
    <small>
    **Inference Settings Explained**

    - **Hybrid threshold (reject ≥ t):**  
      Borrowers with risk ≥ t are automatically rejected.

    - **Review floor (≥ 0.6·t):**  
      Borrowers with risk between this value and t are flagged for manual review.

    - **Blend weight (XGB):**  
      Weight of the XGBoost model in the hybrid prediction.  
      Example: 0.25 → 25 % XGB + 75 % Logistic Regression.
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Top-K Policy")
    rej = st.slider("Reject top % (K₁)", 1, 20, int(REJECT_PCT*100), 1)
    rev = st.slider("Review next % (K₂)", 0, 30, int(REVIEW_PCT*100), 1)
    rej_pct = rej / 100.0
    rev_pct = rev / 100.0

# ---- Tabs ----
tabs = st.tabs(["🔹 Single Borrower", "📂 Batch Scoring", "📈 Reports"])

# ---------- Tab 1: Single Borrower ----------
with tabs[0]:
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("Borrower Information")
        borrower = {}

        used_cats = [c for c in FEATURES if c in CAT_COLS]
        used_nums = [c for c in FEATURES if c in NUM_COLS]
        remaining = [c for c in FEATURES if c not in used_cats + used_nums]
        for c in remaining:
            if c in NUMERIC_FORCE:
                used_nums.append(c)
            elif any(x in c.lower() for x in ["yes","no","whether","own","aware","details","verified","remarks"]):
                used_cats.append(c)
            else:
                used_nums.append(c)

        for c in used_cats:
            borrower[c] = st.selectbox(c, ["Yes", "No"], index=0)

        for c in used_nums:
            borrower[c] = st.number_input(c, value=0.0, step=1.0, format="%.4f")

        if st.button("🔍 Predict", use_container_width=True):
            X_one = pd.DataFrame([borrower])
            p = float(p_default_hybrid(X_one)[0])
            decision_t = decide_threshold(np.array([p]))[0]
            decision_k = apply_topk_policy(np.array([p]), rej_pct, rev_pct)[0]

            with colR:
                st.subheader("Result")
                st.metric("Predicted Default Probability", f"{p:.3f}")
                st.metric("Threshold Policy", decision_t.capitalize())
                st.metric("Top-K Policy", decision_k.capitalize())

            with st.expander("Details"):
                st.json({
                    "p_default": p,
                    "p_repay": 1 - p,
                    "threshold": THRESHOLD,
                    "review_floor": REVIEW_FLOOR,
                    "blend_weight_xgb": W_XGB,
                    "features_order": FEATURES
                })

# ---------- Tab 2: Batch Scoring ----------
with tabs[1]:
    st.subheader("Upload borrower dataset (Excel or CSV)")
    up = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    if up:
        df_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.write("Preview:")
        st.dataframe(df_in.head())

        if st.button("⚡ Score All Rows", use_container_width=True):
            p = p_default_hybrid(df_in)
            df_out = df_in.copy()
            df_out["p_default"] = p
            df_out["decision_threshold"] = decide_threshold(p)
            df_out["decision_topk"] = apply_topk_policy(p, rej_pct, rev_pct)
            st.success("Scoring completed.")
            st.dataframe(df_out.head())
            st.download_button("⬇️ Download scored_borrowers.csv",
                               df_out.to_csv(index=False).encode("utf-8"),
                               "scored_borrowers.csv", "text/csv")

# ---------- Tab 3: Reports ----------
with tabs[2]:
    st.subheader("Evaluate capture on labeled data (optional)")
    st.caption("If your data includes **Advance or Due Amount**, defaults = 1 if > 0 else 0.")
    up_lab = st.file_uploader("Upload labeled file (Excel/CSV)", type=["xlsx","csv"], key="labeled")
    target_col = "Advance or Due Amount"

    if up_lab:
        df_lab = pd.read_csv(up_lab) if up_lab.name.lower().endswith(".csv") else pd.read_excel(up_lab)
        if target_col not in df_lab.columns:
            st.error(f"Target column '{target_col}' not found.")
        else:
            y = (df_lab[target_col] > 0).astype(int).to_numpy()
            p = p_default_hybrid(df_lab)
            stats = topk_capture_stats(y, p, rej_pct, rev_pct)

            st.dataframe(df_lab.head())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total rows", stats["n"])
            c2.metric("Total defaults", stats["total_defaults"])
            c3.metric("Captured (Top-K)", f"{stats['defaults_captured']} ({stats['capture_rate']*100:.1f}%)")

            df_lab["p_default"] = p
            df_lab["decision_threshold"] = decide_threshold(p)
            df_lab["decision_topk"] = apply_topk_policy(p, rej_pct, rev_pct)
            st.download_button("⬇️ Download scored_with_labels.csv",
                               df_lab.to_csv(index=False).encode("utf-8"),
                               "scored_with_labels.csv", "text/csv")

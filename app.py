import os
import io
import json
import glob
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# App constants
# =========================
APP_TITLE = "💳 Micro-Credit Default Scoring"
APP_SUB   = "Hybrid LR + XGBoost • Guarded Thresholds • Top-K Policy"

LR_PATH   = "artifacts/model_lr_calibrated.joblib"
XGB_PATH  = "artifacts/model_xgb_calibrated.joblib"
META_PATH = "artifacts/metadata.json"
ZIP_PATH  = "artifacts_bundle.zip"

# =========================
# Diagnostics (temporary)
# =========================
# Comment these out later if you want a cleaner sidebar
try:
    import sys, sklearn, xgboost, numpy
    st.sidebar.write("Env versions:", {
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    })
except Exception:
    pass
st.sidebar.write("Root files:", glob.glob("*"))
st.sidebar.write("Artifacts files:", glob.glob("artifacts/*"))

# =========================
# Resilient model loader
# =========================
@st.cache_resource
def load_artifacts():
    """
    Load models & metadata with 3 fallbacks:
      1) use artifacts/ if present
      2) else unzip artifacts_bundle.zip from repo root
      3) else ask user to upload the 3 files once and save them
    """
    need = [LR_PATH, XGB_PATH, META_PATH]

    # 1) If missing, try unzipping from artifacts_bundle.zip
    if not all(os.path.exists(p) for p in need) and os.path.exists(ZIP_PATH):
        os.makedirs("artifacts", exist_ok=True)
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(".")
        except Exception as e:
            st.error("Failed to unzip artifacts_bundle.zip")
            st.exception(e)
            st.stop()

    # 2) If still missing, let the user upload once
    if not all(os.path.exists(p) for p in need):
        st.warning("Models not found. Upload the three artifacts below to run the app (one-time).")
        c1, c2, c3 = st.columns(3)
        with c1:
            lr_u  = st.file_uploader("model_lr_calibrated.joblib", type=["joblib"], key="u_lr")
        with c2:
            xgb_u = st.file_uploader("model_xgb_calibrated.joblib", type=["joblib"], key="u_xgb")
        with c3:
            meta_u= st.file_uploader("metadata.json", type=["json"], key="u_meta")

        if lr_u and xgb_u and meta_u:
            os.makedirs("artifacts", exist_ok=True)
            try:
                open(LR_PATH,  "wb").write(lr_u.read())
                open(XGB_PATH, "wb").write(xgb_u.read())
                open(META_PATH,"wb").write(meta_u.read())
                st.success("Artifacts saved. Click **Rerun** (top-right).")
            except Exception as e:
                st.error("Failed to save uploaded artifacts.")
                st.exception(e)
            st.stop()
        else:
            st.stop()

    # 3) Load with error reporting
    try:
        lr_m  = joblib.load(LR_PATH)
        xgb_m = joblib.load(XGB_PATH)
    except Exception as e:
        st.error("Failed to load/unpickle models. This is usually a version mismatch or a corrupted/LFS file.")
        st.exception(e)
        st.stop()

    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    except Exception as e:
        st.error("Failed to read metadata.json.")
        st.exception(e)
        st.stop()

    # Validate minimal keys to avoid KeyError later
    for k in ["features", "thresholds", "topk_policy", "blend_weight_xgb"]:
        if k not in meta:
            st.error(f"metadata.json is missing the key: '{k}'.")
            st.stop()

    return lr_m, xgb_m, meta

# ============= Load artifacts =============
lr_m, xgb_m, meta = load_artifacts()

FEATURES     = meta["features"]
CAT_COLS     = meta.get("cat_cols", [])
NUM_COLS     = meta.get("num_cols", [])
W_XGB        = float(meta["blend_weight_xgb"])
THRESHOLD    = float(meta["thresholds"]["hybrid"])
REJECT_PCT   = float(meta["topk_policy"]["reject_pct"])
REVIEW_PCT   = float(meta["topk_policy"]["review_next_pct"])
REVIEW_FLOOR = max(0.6 * THRESHOLD, 0.05)

# =========================
# Helpers
# =========================
def p_default_hybrid(df_in: pd.DataFrame) -> np.ndarray:
    """
    Hybrid probability = (1-W)*LR + W*XGB
    Ensures all FEATURES exist; fills missing with NaN.
    """
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
    order = np.argsort(-p)  # descending risk
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
# UI
# =========================
st.set_page_config(page_title="Micro-Credit Default Scoring", page_icon="💳", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUB)

with st.sidebar:
    st.markdown("### ⚙️ Inference Settings")
    st.write("These mirror your trained metadata.")
    st.metric("Hybrid threshold (reject ≥ t)", f"{THRESHOLD:.3f}")
    st.metric("Review floor (≥ 0.6·t)", f"{REVIEW_FLOOR:.3f}")
    st.metric("Blend weight (XGB)", f"{W_XGB:.2f}")

    st.markdown("---")
    st.markdown("### 🧪 Top-K Policy")
    rej = st.slider("Reject top % (K₁)", min_value=1, max_value=20, value=int(REJECT_PCT*100), step=1)
    rev = st.slider("Review next % (K₂)", min_value=0, max_value=30, value=int(REVIEW_PCT*100), step=1)
    rej_pct = rej / 100.0
    rev_pct = rev / 100.0

tabs = st.tabs(["🔹 Single Borrower", "📂 Batch Scoring", "📈 Reports (optional)"])

# ---------- Tab 1: Single Borrower ----------
with tabs[0]:
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("Borrower Information")
        borrower = {}

        # Prefer metadata lists to render correct widgets
        used_cats = [c for c in FEATURES if c in CAT_COLS]
        used_nums = [c for c in FEATURES if c in NUM_COLS]

        # If metadata didn't capture all types, fall back on heuristics
        remaining = [c for c in FEATURES if c not in used_cats + used_nums]
        for c in remaining:
            if any(x in c.lower() for x in ["yes", "no", "whether", "own", "aware", "details", "verified", "remarks"]):
                used_cats.append(c)
            else:
                used_nums.append(c)

        # Render categorical first
        for c in used_cats:
            borrower[c] = st.selectbox(c, ["Yes", "No"], index=0)

        # Then numeric
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
                st.metric("Threshold Policy", decision_t)
                st.metric("Top-K Policy", decision_k)

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
    st.subheader("Upload borrowers (Excel or CSV)")
    up = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up)
        else:
            df_in = pd.read_excel(up)
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

            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored_borrowers.csv", data=csv,
                               file_name="scored_borrowers.csv", mime="text/csv")

# ---------- Tab 3: Reports ----------
with tabs[2]:
    st.subheader("Optional: Evaluate capture on a labeled batch")
    st.caption("If your uploaded data includes the target column **Advance or Due Amount**:\n"
               "- DEFAULT = 1 if value > 0 else 0")
    up_lab = st.file_uploader("Upload labeled file (Excel/CSV)", type=["xlsx","csv"], key="labeled")
    target_col = "Advance or Due Amount"

    if up_lab is not None:
        if up_lab.name.lower().endswith(".csv"):
            df_lab = pd.read_csv(up_lab)
        else:
            df_lab = pd.read_excel(up_lab)

        if target_col not in df_lab.columns:
            st.error(f"Target column '{target_col}' not found.")
        else:
            y = (df_lab[target_col] > 0).astype(int).to_numpy()
            p = p_default_hybrid(df_lab)
            stats = topk_capture_stats(y, p, rej_pct, rev_pct)

            st.write("Preview:")
            st.dataframe(df_lab.head())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total rows", stats["n"])
            col2.metric("Total defaults", stats["total_defaults"])
            col3.metric("Captured (Top-K)", f"{stats['defaults_captured']} "
                       f"({stats['capture_rate']*100:.1f}%)")

            # Add scored + decisions for download
            df_rep = df_lab.copy()
            df_rep["p_default"] = p
            df_rep["decision_threshold"] = decide_threshold(p)
            df_rep["decision_topk"] = apply_topk_policy(p, rej_pct, rev_pct)

            csv_rep = df_rep.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored_with_labels.csv", data=csv_rep,
                               file_name="scored_with_labels.csv", mime="text/csv")

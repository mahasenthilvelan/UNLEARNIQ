import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="UnlearnIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #050A14 !important;
    color: #E0EAF8;
    font-family: 'Rajdhani', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1280px; }

.splash-wrapper {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; height:80vh; text-align:center;
    animation: fadeIn 0.9s ease-in;
}
@keyframes fadeIn { from{opacity:0;transform:scale(0.88);} to{opacity:1;transform:scale(1);} }
.splash-logo {
    font-family:'Orbitron',monospace; font-size:5.5rem; font-weight:900;
    background:linear-gradient(135deg,#00F5FF,#0080FF,#8B5CF6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:0.06em; margin-bottom:0.4rem; line-height:1.1;
}
.splash-tagline {
    font-size:1.2rem; color:#6AABFF;
    letter-spacing:0.32em; text-transform:uppercase; margin-bottom:2rem;
}
.splash-ring {
    width:110px; height:110px; border:3px solid transparent;
    border-top-color:#00F5FF; border-right-color:#8B5CF6;
    border-radius:50%; animation:spin 1.1s linear infinite;
}
@keyframes spin { to{transform:rotate(360deg);} }

.section-title {
    font-family:'Orbitron',monospace; font-size:1.35rem; font-weight:700;
    color:#00F5FF; border-left:4px solid #0080FF;
    padding-left:0.8rem; margin:1.8rem 0 0.8rem; letter-spacing:0.04em;
}
.step-badge {
    display:inline-block;
    background:linear-gradient(135deg,#0080FF14,#8B5CF614);
    border:1px solid #0080FF44; border-radius:6px;
    padding:0.28rem 0.9rem; font-family:'Orbitron',monospace;
    font-size:0.7rem; color:#6AABFF; letter-spacing:0.15em;
}
.metric-card {
    background:linear-gradient(135deg,#0B1628,#0D1F3C);
    border:1px solid #1A3A6A; border-radius:12px;
    padding:1.1rem; text-align:center;
    transition:border-color 0.25s, transform 0.2s; margin-bottom:0.6rem;
}
.metric-card:hover { border-color:#00F5FF; transform:translateY(-2px); }
.metric-value { font-family:'Orbitron',monospace; font-size:1.65rem; font-weight:700; color:#00F5FF; }
.metric-label { font-size:0.75rem; color:#6AABFF; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.25rem; }
.metric-sub { font-size:0.68rem; color:#3A6A9A; margin-top:0.1rem; }

.model-card {
    background:linear-gradient(135deg,#0B1628,#0D1F3C);
    border:1px solid #1A3A6A; border-radius:12px;
    padding:1.3rem; text-align:center; margin-bottom:0.6rem;
    transition:border-color 0.25s, transform 0.2s;
}
.model-card:hover { border-color:#8B5CF6; transform:translateY(-2px); }
.model-card-winner { border-color:#00F5FF !important; background:linear-gradient(135deg,#0D2040,#0B2535) !important; }
.model-name { font-family:'Orbitron',monospace; font-size:0.72rem; color:#8B5CF6; letter-spacing:0.1em; margin-bottom:0.5rem; text-transform:uppercase; }
.model-acc { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; color:#00F5FF; }
.model-winner-badge {
    display:inline-block; margin-top:0.4rem;
    background:#00F5FF22; border:1px solid #00F5FF55;
    border-radius:4px; padding:0.15rem 0.6rem;
    font-size:0.68rem; color:#00F5FF; letter-spacing:0.1em;
}

.verdict-strong {
    background:linear-gradient(135deg,#00301A,#004D29); border:2px solid #00C853;
    border-radius:14px; padding:1.8rem; text-align:center;
    font-family:'Orbitron',monospace; font-size:1.45rem; color:#00E676;
}
.verdict-moderate {
    background:linear-gradient(135deg,#2D1F00,#4A3300); border:2px solid #FFB300;
    border-radius:14px; padding:1.8rem; text-align:center;
    font-family:'Orbitron',monospace; font-size:1.45rem; color:#FFD54F;
}
.verdict-weak {
    background:linear-gradient(135deg,#2D0000,#4A0000); border:2px solid #FF1744;
    border-radius:14px; padding:1.8rem; text-align:center;
    font-family:'Orbitron',monospace; font-size:1.45rem; color:#FF5252;
}

.upload-title {
    font-family:'Orbitron',monospace; font-size:2.5rem; font-weight:900;
    background:linear-gradient(135deg,#00F5FF,#0080FF,#8B5CF6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.4rem;
}
.upload-sub { color:#6AABFF; font-size:1rem; letter-spacing:0.14em; margin-bottom:0.5rem; }

.user-box {
    background:linear-gradient(135deg,#0B1628,#0D1F3C);
    border:1px solid #1A3A6A; border-radius:14px; padding:1.4rem; margin:1rem 0;
}
.user-box-title { font-family:'Orbitron',monospace; font-size:0.95rem; color:#00F5FF; margin-bottom:0.35rem; }

.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#00F5FF,#0080FF,#8B5CF6) !important; border-radius:4px;
}
.neon-divider {
    height:1px; margin:1.8rem 0;
    background:linear-gradient(90deg,transparent,#0080FF88,#00F5FF,#0080FF88,transparent);
}
.score-table { width:100%; border-collapse:collapse; }
.score-table tr { border-bottom:1px solid #1A3A6A; }
.score-table td { padding:0.45rem 0.7rem; font-size:0.88rem; }
.score-table td:first-child { color:#6AABFF; }
.score-table td:last-child { color:#00F5FF; font-family:'Orbitron',monospace; font-size:0.82rem; font-weight:700; text-align:right; }

span[data-baseweb="tag"] {
    background-color:#0080FF33 !important;
    border:1px solid #0080FF88 !important;
    color:#00F5FF !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
for _k, _v in {"phase":"splash","df_raw":None,"selected_users":[],"results":None,"available_users":[]}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def validate_and_load(df):
    if 'comment_text' in df.columns:
        text_col = 'comment_text'
    elif 'tweet' in df.columns:
        text_col = 'tweet'
    else:
        return None, None, False, "No text column found. Need `comment_text` or `tweet`."
    if 'toxic' in df.columns:
        label_col = 'toxic'
    elif 'class' in df.columns:
        label_col = 'class'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        return None, None, False, "No label column. Need `toxic`, `class`, or `label`."
    return text_col, label_col, True, f"text=`{text_col}` | label=`{label_col}`"

def build_binary_label(df, label_col):
    if label_col == 'class':
        return df[label_col].apply(lambda x: 0 if x == 2 else 1)
    return df[label_col].astype(int)


# ══════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════
def run_pipeline(df_raw, selected_users, pb, st_txt):

    # ── STEP 1: Prepare ──────────────────────────────
    st_txt.markdown('<div class="step-badge">STEP 1 — LOADING & PREPARING DATA</div>', unsafe_allow_html=True)
    pb.progress(4)

    df = df_raw.copy()
    text_col, label_col, _, _ = validate_and_load(df)
    df['label']      = build_binary_label(df, label_col)
    df               = df.rename(columns={text_col: 'Text'})
    df               = df[['Text', 'label']].dropna()
    df['clean_text'] = df['Text'].apply(clean_text)

    # Assign UserIds — use actual count so all users exist
    n_users = 100  # safe: 100 users always have rows on any size dataset
    df['UserId'] = ['user_' + str(i % n_users) for i in range(len(df))]

    # Balance
    sample_size = df['label'].value_counts().min()
    df = df.groupby('label').apply(
        lambda x: x.sample(sample_size, replace=True, random_state=42)
    ).reset_index(drop=True)

    # Re-assign UserIds after balancing so indices are clean
    df['UserId'] = ['user_' + str(i % n_users) for i in range(len(df))]

    pb.progress(10); time.sleep(0.15)

    # ── STEP 2: Vectorise + Baseline ─────────────────
    st_txt.markdown('<div class="step-badge">STEP 2 — VECTORISATION & BASELINE MODEL</div>', unsafe_allow_html=True)

    X, y = df['clean_text'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf      = TfidfVectorizer(max_features=1500)
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf  = tfidf.transform(X_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_tf, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test_tf))

    pb.progress(22); time.sleep(0.15)

    # ── STEP 3: Model Comparison ─────────────────────
    st_txt.markdown('<div class="step-badge">STEP 3 — MODEL COMPARISON (LR vs NB vs DT)</div>', unsafe_allow_html=True)

    model_zoo = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes":         MultinomialNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }
    model_results = {}
    for name, mdl in model_zoo.items():
        mdl.fit(X_train_tf, y_train)
        model_results[name] = round(accuracy_score(y_test, mdl.predict(X_test_tf)), 4)

    pb.progress(34); time.sleep(0.15)

    # ── STEP 4: Target user data ─────────────────────
    st_txt.markdown('<div class="step-badge">STEP 4 — COLLECTING TARGET USER DATA</div>', unsafe_allow_html=True)

    # Only keep selected users that actually exist in the (balanced) df
    valid_selected = [u for u in selected_users if u in df['UserId'].values]
    if not valid_selected:
        raise ValueError(f"None of the selected users exist in the dataset after balancing. Available: user_0 to user_{n_users-1}")

    user_mask     = df['UserId'].isin(valid_selected)
    user_data_all = df[user_mask]
    X_user        = user_data_all['clean_text']

    # Safety check
    if len(X_user) == 0:
        raise ValueError("Selected users have no data rows. Please select different users.")

    X_user_tf        = tfidf.transform(X_user)
    user_pred_before = baseline_model.predict(X_user_tf)
    user_prob_before = baseline_model.predict_proba(X_user_tf)[:, 1]

    pb.progress(42); time.sleep(0.15)

    # ── STEP 5: Unlearning ───────────────────────────
    st_txt.markdown('<div class="step-badge">STEP 5 — MACHINE UNLEARNING (RETRAIN)</div>', unsafe_allow_html=True)

    # Multi-criteria: remove only toxic rows of selected users
    user_data_toxic = df[(df['UserId'].isin(valid_selected)) & (df['label'] == 1)]
    remaining_data  = df.drop(user_data_toxic.index)

    X_r, y_r = remaining_data['clean_text'], remaining_data['label']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_r, y_r, test_size=0.2, stratify=y_r, random_state=42
    )

    tfidf_u = TfidfVectorizer(max_features=1500)
    X_tr_tf = tfidf_u.fit_transform(X_tr)
    X_te_tf = tfidf_u.transform(X_te)

    unlearn_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.05)
    unlearn_model.fit(X_tr_tf, y_tr)

    # Zero out user word weights
    user_words = set(" ".join(X_user).split())
    for word in user_words:
        if word in tfidf_u.vocabulary_:
            unlearn_model.coef_[0][tfidf_u.vocabulary_[word]] = 0

    X_user_after    = tfidf_u.transform(X_user)
    user_pred_after = unlearn_model.predict(X_user_after)
    user_prob_after = unlearn_model.predict_proba(X_user_after)[:, 1]

    pb.progress(55); time.sleep(0.15)

    # ── STEP 6: Metrics + MIA ────────────────────────
    st_txt.markdown('<div class="step-badge">STEP 6 — UNLEARNING METRICS + MIA ATTACK</div>', unsafe_allow_html=True)

    prediction_change = float(np.mean(user_pred_before != user_pred_after))
    confidence_drop   = float(np.mean(np.abs(user_prob_before - user_prob_after)))

    atk_X = np.concatenate([
        baseline_model.predict_proba(X_train_tf)[:, 1],
        baseline_model.predict_proba(X_test_tf)[:, 1]
    ]).reshape(-1, 1)
    atk_y = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])
    mia_before = accuracy_score(atk_y, LogisticRegression().fit(atk_X, atk_y).predict(atk_X))

    atk2_X = np.concatenate([
        unlearn_model.predict_proba(X_tr_tf)[:, 1],
        unlearn_model.predict_proba(X_te_tf)[:, 1]
    ]).reshape(-1, 1)
    atk2_y = np.concatenate([np.ones(len(X_tr)), np.zeros(len(X_te))])
    mia_after = accuracy_score(atk2_y, LogisticRegression().fit(atk2_X, atk2_y).predict(atk2_X))

    pb.progress(66); time.sleep(0.15)

    # ── STEP 7: Re-ID Attack ─────────────────────────
    st_txt.markdown('<div class="step-badge">STEP 7 — RE-IDENTIFICATION ATTACK</div>', unsafe_allow_html=True)

    prob_before_u    = baseline_model.predict_proba(tfidf.transform(X_user))[:, 1]
    prob_after_u     = unlearn_model.predict_proba(tfidf_u.transform(X_user))[:, 1]
    prob_after_noisy = np.clip(prob_after_u + np.random.laplace(0, 0.2, prob_after_u.shape), 0, 1)

    reid_X = np.clip(
        np.concatenate([prob_before_u, prob_after_noisy]).reshape(-1, 1),
        0.2, 0.8
    )
    reid_y = np.concatenate([np.ones(len(prob_before_u)), np.zeros(len(prob_after_noisy))])

    if len(np.unique(reid_y)) < 2 or len(reid_y) < 6:
        reid_auc = 0.5
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(reid_X, reid_y)
        reid_auc = float(roc_auc_score(reid_y, rf.predict_proba(reid_X)[:, 1]))

    pb.progress(76); time.sleep(0.15)

    # ── STEP 8: Feature Analysis ─────────────────────
    st_txt.markdown('<div class="step-badge">STEP 8 — FEATURE ANALYSIS</div>', unsafe_allow_html=True)

    imp_before = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'weight': baseline_model.coef_[0]})
    imp_before['abs'] = imp_before['weight'].abs()
    imp_before = imp_before.sort_values('abs', ascending=False)

    imp_after = pd.DataFrame({'word': tfidf_u.get_feature_names_out(), 'weight': unlearn_model.coef_[0]})
    imp_after['abs'] = imp_after['weight'].abs()
    imp_after = imp_after.sort_values('abs', ascending=False)

    residual_words  = user_words.intersection(set(imp_after.head(50)['word']))
    feature_overlap = len(set(imp_before.head(50)['word']) & set(imp_after.head(50)['word']))
    param_shift     = float(np.linalg.norm(baseline_model.coef_ - unlearn_model.coef_))

    pb.progress(86); time.sleep(0.15)

    # ── STEP 9: Final Scoring ────────────────────────
    st_txt.markdown('<div class="step-badge">STEP 9 — SCORING & VERDICT</div>', unsafe_allow_html=True)

    threshold_acc  = float(np.mean((prob_before_u > 0.5).astype(int) == np.ones(len(prob_before_u))))
    threshold_priv = 1.0 - abs(threshold_acc - 0.5) * 2
    auc_privacy    = 1.0 - abs(reid_auc - 0.5) * 2

    ensemble_score = (
        0.4 * auc_privacy +
        0.3 * threshold_priv +
        0.3 * max(0.0, mia_before - mia_after)
    )

    prob_after_noisy2 = np.clip(
        prob_after_u + np.random.laplace(0, 0.2, prob_after_u.shape), 0, 1
    )
    confidence_signal = float(np.mean(np.abs(prob_before_u - prob_after_noisy2)))

    final_score = (
        0.25 * prediction_change +
        0.15 * min(confidence_drop * 2, 1.0) +
        0.25 * max(0.0, mia_before - mia_after) +
        0.15 * auc_privacy +
        0.10 * threshold_priv +
        0.10 * min(param_shift, 1.0)
    )

    if final_score > 0.4:
        verdict = ("✅ STRONG PRIVACY", "strong")
    elif final_score > 0.3:
        verdict = ("⚠️ MODERATE PRIVACY", "moderate")
    else:
        verdict = ("❌ WEAK PRIVACY", "weak")

    pb.progress(100); time.sleep(0.2)
    st_txt.markdown('<div class="step-badge">✔ ALL STEPS COMPLETE</div>', unsafe_allow_html=True)

    return {
        "df": df,
        "n_users": n_users,
        "selected_users": valid_selected,
        "user_count": len(valid_selected),
        "user_sample_count": len(user_data_all),
        "toxic_removed": len(user_data_toxic),
        "model_results": model_results,
        "baseline_acc": baseline_acc,
        "prediction_change": prediction_change,
        "confidence_drop": confidence_drop,
        "mia_before": mia_before,
        "mia_after": mia_after,
        "reid_auc": reid_auc,
        "param_shift": param_shift,
        "feature_overlap": feature_overlap,
        "threshold_acc": threshold_acc,
        "threshold_priv": threshold_priv,
        "auc_privacy": auc_privacy,
        "ensemble_score": ensemble_score,
        "confidence_signal": confidence_signal,
        "final_score": final_score,
        "imp_before": imp_before.head(12),
        "imp_after": imp_after.head(12),
        "residual_words": list(residual_words)[:10],
        "residual_count": len(residual_words),
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════
# SPLASH
# ══════════════════════════════════════════════════════
if st.session_state.phase == "splash":
    st.markdown("""
    <div class="splash-wrapper">
        <div class="splash-logo">UnlearnIQ</div>
        <div class="splash-tagline">Machine Unlearning &amp; Privacy Intelligence</div>
        <div class="splash-ring"></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1.8)
    st.session_state.phase = "upload"
    st.rerun()


# ══════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════
elif st.session_state.phase == "upload":
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2.4, 1])
    with col:
        st.markdown('<div style="text-align:center"><div class="upload-title">🧠 UnlearnIQ</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center"><div class="upload-sub">Machine Unlearning &amp; Privacy Intelligence</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop your CSV file here", type=["csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            text_col, label_col, ok, msg = validate_and_load(df_raw)
            if not ok:
                st.error(f"❌ {msg}")
            else:
                st.success(f"✅ **{len(df_raw):,} rows** loaded — {msg}")
                show_col = text_col if text_col else df_raw.columns[0]
                st.dataframe(df_raw[[show_col, label_col]].head(5), use_container_width=True)
                st.session_state.df_raw = df_raw
                st.session_state.phase  = "user_select"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0B1628,#0D1F3C);border:1px solid #1A3A6A;
        border-radius:10px;padding:1rem 1.2rem;font-size:0.84rem;color:#6AABFF;line-height:1.9;">
        <b style="color:#00F5FF;font-family:Orbitron,monospace;font-size:0.78rem;">SUPPORTED FORMATS</b><br>
        • <b style="color:#E0EAF8;">Kaggle Toxic Comments</b> — <code>comment_text</code> + <code>toxic</code><br>
        • <b style="color:#E0EAF8;">Tweet dataset</b> — <code>tweet</code> + <code>class</code> &nbsp;(0,1=toxic | 2=clean)<br>
        • <b style="color:#E0EAF8;">Generic</b> — text column + <code>label</code> &nbsp;(0 / 1)
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# USER SELECT
# ══════════════════════════════════════════════════════
elif st.session_state.phase == "user_select":
    st.markdown(
        '<div style="text-align:center;font-family:Orbitron,monospace;font-size:2.1rem;'
        'font-weight:900;background:linear-gradient(135deg,#00F5FF,#0080FF,#8B5CF6);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding:0.6rem 0;">'
        '🧠 UnlearnIQ</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Select Users to Unlearn</div>', unsafe_allow_html=True)

    df_raw    = st.session_state.df_raw
    n_rows    = len(df_raw)
    N_USERS   = 100                                          # fixed — always user_0 … user_99
    all_users = ['user_' + str(i) for i in range(N_USERS)]  # exactly what pipeline assigns
    rows_per  = max(1, n_rows // N_USERS)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{n_rows:,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{N_USERS}</div><div class="metric-label">Unique Users</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">~{rows_per:,}</div><div class="metric-label">Rows / User</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#8B5CF6">1,500</div><div class="metric-label">TF-IDF Features</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="user-box">
        <div class="user-box-title">🎯 Choose users whose data will be erased from the model</div>
        <div style="color:#4A7AAA;font-size:0.86rem;margin-top:0.3rem;">
            Dataset is split into <b style="color:#00F5FF">100 synthetic users</b>
            (user_0 → user_99). Select one or more from the dropdown below.
            Only their <b style="color:#FF5252">toxic</b> rows are removed before retraining.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── clean multiselect — no free-text typing that causes ghost users ──
    prev        = [u for u in st.session_state.selected_users if u in all_users]
    default_sel = prev if prev else [all_users[0]]

    selected = st.multiselect(
        "Select users to unlearn  (user_0 – user_99):",
        options=all_users,          # fixed list — only valid users shown
        default=default_sel,
        help="Pick one or many from the list. These users' toxic rows will be removed from retraining."
    )

    if selected:
        approx = len(selected) * rows_per
        st.info(f"🗑  **{len(selected)} user(s) selected** — approx **{approx:,} rows** flagged for unlearning")

    # Show the full list so user knows what's available
    with st.expander("📋 View all available users (user_0 – user_99)"):
        cols = st.columns(10)
        for i, u in enumerate(all_users):
            cols[i % 10].markdown(
                f'<span style="color:{"#00F5FF" if u in selected else "#4A7AAA"};font-size:0.8rem;">{u}</span>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.phase = "upload"
            st.rerun()
    with colB:
        if st.button("🚀  Run UnlearnIQ Pipeline", use_container_width=True, disabled=(len(selected) == 0)):
            st.session_state.selected_users = selected
            st.session_state.phase          = "processing"
            st.rerun()


# ══════════════════════════════════════════════════════
# PROCESSING
# ══════════════════════════════════════════════════════
elif st.session_state.phase == "processing":
    st.markdown('<div class="section-title">⚙ Running Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#6AABFF;margin-bottom:1rem;font-size:0.9rem;">'
        f'Unlearning: <b style="color:#00F5FF">{", ".join(st.session_state.selected_users)}</b></div>',
        unsafe_allow_html=True
    )
    pb     = st.progress(0)
    st_txt = st.empty()

    try:
        results = run_pipeline(
            st.session_state.df_raw.copy(),
            st.session_state.selected_users,
            pb, st_txt
        )
        st.session_state.results = results
        st.session_state.phase   = "results"
        time.sleep(0.3)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        if st.button("← Go Back"):
            st.session_state.phase = "user_select"
            st.rerun()


# ══════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════
elif st.session_state.phase == "results":
    r  = st.session_state.results
    df = r["df"]

    # Header
    st.markdown(
        '<div style="text-align:center;font-family:Orbitron,monospace;font-size:2.6rem;'
        'font-weight:900;background:linear-gradient(135deg,#00F5FF,#0080FF,#8B5CF6);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;">'
        'UnlearnIQ</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center;color:#6AABFF;letter-spacing:0.22em;'
        'font-size:0.82rem;margin-bottom:0.6rem;">ANALYSIS COMPLETE</div>',
        unsafe_allow_html=True
    )
    users_str = ", ".join(r["selected_users"])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0D1F3C,#0B1628);border:1px solid #0080FF44;
    border-radius:10px;padding:0.75rem 1.2rem;margin-bottom:0.5rem;
    display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;">
        <span>🗑 <b style="color:#00F5FF">{r['user_count']} user(s) unlearned:</b>
        <span style="color:#8B9ABB"> {users_str}</span></span>
        <span style="color:#6AABFF">
            <b style="color:#00F5FF">{r['user_sample_count']:,}</b> total rows &nbsp;|&nbsp;
            <b style="color:#FF5252">{r['toxic_removed']:,}</b> toxic rows removed
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Dataset Overview ──────────────────────────────
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Balanced Samples</div></div>', unsafe_allow_html=True)
    with d2: st.markdown(f'<div class="metric-card"><div class="metric-value">{int(df["label"].sum()):,}</div><div class="metric-label">Toxic</div></div>', unsafe_allow_html=True)
    with d3: st.markdown(f'<div class="metric-card"><div class="metric-value">{int((df["label"]==0).sum()):,}</div><div class="metric-label">Non-Toxic</div></div>', unsafe_allow_html=True)
    with d4: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["baseline_acc"]:.3f}</div><div class="metric-label">Baseline Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────
    st.markdown('<div class="section-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
    best_model = max(r["model_results"], key=r["model_results"].get)
    mc1, mc2, mc3 = st.columns(3)
    for idx, (name, acc) in enumerate(r["model_results"].items()):
        is_w   = (name == best_model)
        badge  = '<div class="model-winner-badge">⭐ BEST MODEL</div>' if is_w else ""
        wcls   = "model-card-winner" if is_w else ""
        [mc1, mc2, mc3][idx].markdown(
            f'<div class="model-card {wcls}"><div class="model-name">{name}</div>'
            f'<div class="model-acc">{acc:.3f}</div>{badge}</div>',
            unsafe_allow_html=True
        )

    fig_mc = go.Figure(go.Bar(
        x=list(r["model_results"].keys()),
        y=list(r["model_results"].values()),
        marker=dict(color=list(r["model_results"].values()),
                    colorscale=[[0,"#1A3A6A"],[0.5,"#0080FF"],[1,"#00F5FF"]],
                    line=dict(color="#0080FF", width=1)),
        text=[f'{v:.3f}' for v in r["model_results"].values()],
        textposition='outside',
        textfont=dict(color="#00F5FF", family="Orbitron")
    ))
    fig_mc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0EAF8', height=260,
        yaxis=dict(range=[0,1.1], gridcolor='#1A3A6A', tickfont=dict(color='#6AABFF')),
        xaxis=dict(tickfont=dict(color='#6AABFF')),
        margin=dict(l=20,r=20,t=20,b=20), showlegend=False
    )
    st.plotly_chart(fig_mc, use_container_width=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Unlearning Metrics ────────────────────────────
    st.markdown('<div class="section-title">🗑 Unlearning Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["prediction_change"]:.3f}</div><div class="metric-label">Prediction Change</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["confidence_drop"]:.3f}</div><div class="metric-label">Confidence Drop</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["mia_before"]:.3f}</div><div class="metric-label">MIA Before</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["mia_after"]:.3f}</div><div class="metric-label">MIA After</div></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#8B5CF6">{r["param_shift"]:.3f}</div><div class="metric-label">Param Shift</div></div>', unsafe_allow_html=True)

    fig_mia = go.Figure()
    fig_mia.add_trace(go.Bar(name='Before', x=['MIA Score'], y=[r["mia_before"]], marker_color='#FF4D4D', width=0.25))
    fig_mia.add_trace(go.Bar(name='After',  x=['MIA Score'], y=[r["mia_after"]],  marker_color='#00F5FF', width=0.25))
    fig_mia.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0EAF8', height=220, barmode='group',
        margin=dict(l=20,r=20,t=10,b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#6AABFF'),
        yaxis=dict(range=[0,1], gridcolor='#1A3A6A'),
        xaxis=dict(gridcolor='#1A3A6A')
    )
    st.plotly_chart(fig_mia, use_container_width=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Attack Results ────────────────────────────────
    st.markdown('<div class="section-title">🔐 Attack Results</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["reid_auc"]:.3f}</div><div class="metric-label">Re-ID AUC</div><div class="metric-sub">RF Attack (Step 17)</div></div>', unsafe_allow_html=True)
    with a2: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["threshold_acc"]:.3f}</div><div class="metric-label">Threshold Accuracy</div><div class="metric-sub">Threshold Attack (Step 26)</div></div>', unsafe_allow_html=True)
    with a3: st.markdown(f'<div class="metric-card"><div class="metric-value">{r["ensemble_score"]:.4f}</div><div class="metric-label">Ensemble Score</div><div class="metric-sub">Combined (Step 27)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Feature Analysis ──────────────────────────────
    st.markdown('<div class="section-title">📌 Feature Analysis</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["feature_overlap"]}</div><div class="metric-label">Feature Overlap</div><div class="metric-sub">Top-50 shared before/after</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#FF5252">{r["residual_count"]}</div><div class="metric-label">Residual Words</div><div class="metric-sub">User influence still present</div></div>', unsafe_allow_html=True)
        if r["residual_words"]:
            st.markdown(
                f'<div style="background:#0D1628;border:1px solid #1A3A6A;border-radius:8px;'
                f'padding:0.8rem;font-size:0.8rem;color:#6AABFF;margin-top:0.5rem;">'
                f'<b style="color:#FF5252">Residual:</b> {", ".join(r["residual_words"])}</div>',
                unsafe_allow_html=True
            )
    with right:
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Before Unlearning**")
            fig_b = px.bar(r["imp_before"].head(10), x='weight', y='word', orientation='h',
                           color='weight', color_continuous_scale=['#FF4D4D','#FFB300','#00F5FF'],
                           labels={'weight':'Coeff','word':'Word'})
            fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 font_color='#E0EAF8', height=300, margin=dict(l=0,r=0,t=10,b=0),
                                 yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
            st.plotly_chart(fig_b, use_container_width=True)
        with fc2:
            st.markdown("**After Unlearning**")
            fig_a = px.bar(r["imp_after"].head(10), x='weight', y='word', orientation='h',
                           color='weight', color_continuous_scale=['#8B5CF6','#0080FF','#00F5FF'],
                           labels={'weight':'Coeff','word':'Word'})
            fig_a.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 font_color='#E0EAF8', height=300, margin=dict(l=0,r=0,t=10,b=0),
                                 yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
            st.plotly_chart(fig_a, use_container_width=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Final Score ───────────────────────────────────
    st.markdown('<div class="section-title">🏁 Final Enhanced Privacy Score  (Step 28)</div>', unsafe_allow_html=True)
    g_col, t_col = st.columns([1, 1])
    with g_col:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(r["final_score"], 4),
            delta={'reference':0.3,'increasing':{'color':'#00E676'},'decreasing':{'color':'#FF5252'}},
            title={'text':'Final Enhanced Score','font':{'color':'#6AABFF','family':'Orbitron','size':13}},
            gauge={
                'axis':{'range':[0,1],'tickcolor':'#6AABFF'},
                'bar':{'color':'#00F5FF'},
                'bgcolor':'#0B1628','bordercolor':'#1A3A6A',
                'steps':[
                    {'range':[0.0,0.3],'color':'#200A0A'},
                    {'range':[0.3,0.4],'color':'#1A1A0A'},
                    {'range':[0.4,1.0],'color':'#0A200A'},
                ],
                'threshold':{'line':{'color':'#FF4D4D','width':2},'value':0.3}
            },
            number={'font':{'color':'#00F5FF','family':'Orbitron'}}
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#E0EAF8', height=300)
        st.plotly_chart(fig_g, use_container_width=True)

    with t_col:
        st.markdown("<br>", unsafe_allow_html=True)
        mia_sig = max(0.0, r["mia_before"] - r["mia_after"])
        st.markdown(f"""
        <table class="score-table">
          <tr><td>Prediction Change ×0.25</td><td>{r['prediction_change']:.4f}</td></tr>
          <tr><td>Confidence Drop ×2 min(1) ×0.15</td><td>{min(r['confidence_drop']*2,1):.4f}</td></tr>
          <tr><td>MIA Signal ×0.25</td><td>{mia_sig:.4f}</td></tr>
          <tr><td>AUC Privacy ×0.15</td><td>{r['auc_privacy']:.4f}</td></tr>
          <tr><td>Threshold Privacy ×0.10</td><td>{r['threshold_priv']:.4f}</td></tr>
          <tr><td>Param Shift min(1) ×0.10</td><td>{min(r['param_shift'],1):.4f}</td></tr>
          <tr><td><b style="color:#00F5FF">FINAL SCORE</b></td>
              <td style="color:#00F5FF;font-size:1rem">{r['final_score']:.4f}</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    verdict_text, verdict_type = r["verdict"]
    st.markdown(f'<div class="verdict-{verdict_type}">{verdict_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    colX, colY = st.columns(2)
    with colX:
        if st.button("👤  Change Users & Re-run", use_container_width=True):
            st.session_state.phase = "user_select"
            st.rerun()
    with colY:
        if st.button("🔄  Upload New Dataset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UnlearnIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #050A14;
    color: #E0EAF8;
    font-family: 'Rajdhani', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }

.splash-wrapper {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 80vh; text-align: center;
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn { from {opacity:0;transform:scale(0.92);} to {opacity:1;transform:scale(1);} }
.splash-logo {
    font-family: 'Orbitron', monospace;
    font-size: 5rem; font-weight: 900;
    background: linear-gradient(135deg, #00F5FF, #0080FF, #8B5CF6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em; margin-bottom: 0.3rem;
}
.splash-tagline {
    font-size: 1.3rem; color: #6AABFF;
    letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 1.5rem;
}
.splash-ring {
    width: 120px; height: 120px;
    border: 3px solid transparent;
    border-top-color: #00F5FF; border-right-color: #0080FF;
    border-radius: 50%; animation: spin 1.2s linear infinite; margin-top: 1rem;
}
@keyframes spin { to { transform: rotate(360deg); } }

.section-title {
    font-family: 'Orbitron', monospace; font-size: 1.6rem; font-weight: 700;
    color: #00F5FF; border-left: 4px solid #0080FF;
    padding-left: 0.8rem; margin: 1.5rem 0 0.8rem; letter-spacing: 0.05em;
}
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0080FF22, #8B5CF622);
    border: 1px solid #0080FF66; border-radius: 6px;
    padding: 0.3rem 0.9rem; font-family: 'Orbitron', monospace;
    font-size: 0.75rem; color: #6AABFF; letter-spacing: 0.15em; margin-bottom: 0.4rem;
}
.metric-card {
    background: linear-gradient(135deg, #0B1628, #0D1F3C);
    border: 1px solid #1A3A6A; border-radius: 12px;
    padding: 1.2rem; text-align: center; transition: border-color 0.3s; margin-bottom: 0.5rem;
}
.metric-card:hover { border-color: #00F5FF; }
.metric-value { font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 700; color: #00F5FF; }
.metric-label { font-size: 0.82rem; color: #6AABFF; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem; }

.verdict-strong {
    background: linear-gradient(135deg, #00301A, #004D29); border: 2px solid #00C853;
    border-radius: 12px; padding: 1.5rem; text-align: center;
    font-family: 'Orbitron', monospace; font-size: 1.4rem; color: #00E676;
}
.verdict-moderate {
    background: linear-gradient(135deg, #2D1F00, #4A3300); border: 2px solid #FFB300;
    border-radius: 12px; padding: 1.5rem; text-align: center;
    font-family: 'Orbitron', monospace; font-size: 1.4rem; color: #FFD54F;
}
.verdict-weak {
    background: linear-gradient(135deg, #2D0000, #4A0000); border: 2px solid #FF1744;
    border-radius: 12px; padding: 1.5rem; text-align: center;
    font-family: 'Orbitron', monospace; font-size: 1.4rem; color: #FF5252;
}
.upload-wrapper {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 50vh; text-align: center;
}
.upload-title {
    font-family: 'Orbitron', monospace; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #00F5FF, #0080FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
}
.upload-sub { color: #6AABFF; font-size: 1.1rem; letter-spacing: 0.1em; margin-bottom: 2rem; }
.user-select-box {
    background: linear-gradient(135deg, #0B1628, #0D1F3C);
    border: 1px solid #1A3A6A; border-radius: 14px; padding: 1.5rem; margin: 1rem 0;
}
.user-select-title {
    font-family: 'Orbitron', monospace; font-size: 1.1rem;
    color: #00F5FF; margin-bottom: 0.5rem; letter-spacing: 0.08em;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00F5FF, #0080FF, #8B5CF6); border-radius: 4px;
}
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #0080FF, #00F5FF, #0080FF, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("phase", "splash"), ("results", None), ("df_raw", None), ("selected_users", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

TOXIC_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def make_label(row):
    return 1 if any(row.get(c, 0) == 1 for c in TOXIC_COLS) else 0

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(df_raw, selected_users, progress_bar, status_text):

    status_text.markdown('<div class="step-badge">STEP 1 — PREPARING DATASET</div>', unsafe_allow_html=True)
    progress_bar.progress(5)

    df = df_raw.copy()
    text_col = 'comment_text' if 'comment_text' in df.columns else 'tweet'

    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    elif 'class' in df.columns:
        df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)
    else:
        df['label'] = df.apply(make_label, axis=1)

    df = df.rename(columns={text_col: 'Text'})
    df = df[['Text', 'label']].dropna()
    df['clean_text'] = df['Text'].apply(clean_text)
    df['UserId'] = ['user_' + str(i % 100) for i in range(len(df))]

    sample_size = df['label'].value_counts().min()
    df = df.groupby('label').apply(
        lambda x: x.sample(sample_size, replace=True, random_state=42)
    ).reset_index(drop=True)

    progress_bar.progress(15)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 2 — TRAINING BASELINE MODEL</div>', unsafe_allow_html=True)
    X, y = df['clean_text'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(max_features=2000)
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf  = tfidf.transform(X_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_tf, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test_tf))
    progress_bar.progress(30)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 3 — COLLECTING TARGET USER DATA</div>', unsafe_allow_html=True)
    user_mask  = df['UserId'].isin(selected_users)
    user_data  = df[user_mask]
    X_user     = user_data['clean_text']
    X_user_tf  = tfidf.transform(X_user)
    user_pred_before = baseline_model.predict(X_user_tf)
    user_prob_before = baseline_model.predict_proba(X_user_tf)[:, 1]
    progress_bar.progress(45)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 4 — MACHINE UNLEARNING</div>', unsafe_allow_html=True)
    remaining = df[~user_mask]
    X_r, y_r  = remaining['clean_text'], remaining['label']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, stratify=y_r, random_state=42)

    tfidf_u = TfidfVectorizer(max_features=2000)
    X_tr_tf = tfidf_u.fit_transform(X_tr)
    X_te_tf = tfidf_u.transform(X_te)

    unlearn_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
    unlearn_model.fit(X_tr_tf, y_tr)
    unlearn_model.coef_ += np.random.normal(0, 0.02, unlearn_model.coef_.shape)

    X_user_after    = tfidf_u.transform(X_user)
    user_pred_after = unlearn_model.predict(X_user_after)
    user_prob_after = unlearn_model.predict_proba(X_user_after)[:, 1]
    progress_bar.progress(60)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 5 — COMPUTING METRICS</div>', unsafe_allow_html=True)
    prediction_change = np.mean(user_pred_before != user_pred_after)
    confidence_drop   = np.mean(np.abs(user_prob_before - user_prob_after))

    train_probs = baseline_model.predict_proba(X_train_tf)[:, 1]
    test_probs  = baseline_model.predict_proba(X_test_tf)[:, 1]
    atk_X = np.concatenate([train_probs, test_probs]).reshape(-1, 1)
    atk_y = np.concatenate([np.ones(len(train_probs)), np.zeros(len(test_probs))])
    atk   = LogisticRegression().fit(atk_X, atk_y)
    mia_before = accuracy_score(atk_y, atk.predict(atk_X))

    atk2_X = np.concatenate([unlearn_model.predict_proba(X_tr_tf)[:, 1],
                               unlearn_model.predict_proba(X_te_tf)[:, 1]]).reshape(-1, 1)
    atk2_y = np.concatenate([np.ones(len(X_tr_tf.toarray())), np.zeros(len(X_te_tf.toarray()))])
    atk2   = LogisticRegression().fit(atk2_X, atk2_y)
    mia_after = accuracy_score(atk2_y, atk2.predict(atk2_X))

    forgetting_score = (
        0.4 * prediction_change +
        0.3 * min(confidence_drop, 1) +
        0.3 * max(0, mia_before - mia_after)
    )
    progress_bar.progress(72)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 6 — FEATURE IMPORTANCE ANALYSIS</div>', unsafe_allow_html=True)
    importance_before = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'weight': baseline_model.coef_[0]})
    importance_before['abs'] = importance_before['weight'].abs()
    importance_before = importance_before.sort_values('abs', ascending=False)

    importance_after = pd.DataFrame({'word': tfidf_u.get_feature_names_out(), 'weight': unlearn_model.coef_[0]})
    importance_after['abs'] = importance_after['weight'].abs()
    importance_after = importance_after.sort_values('abs', ascending=False)

    user_words     = set(" ".join(X_user).split())
    residual_words = user_words.intersection(set(importance_after.head(50)['word']))

    vocab = tfidf_u.vocabulary_
    for word in user_words:
        if word in vocab:
            unlearn_model.coef_[0][vocab[word]] = 0

    progress_bar.progress(85)
    time.sleep(0.3)

    status_text.markdown('<div class="step-badge">STEP 7 — PRIVACY EVALUATION</div>', unsafe_allow_html=True)
    prob_before_u    = baseline_model.predict_proba(tfidf.transform(X_user))[:, 1]
    prob_after_u     = unlearn_model.predict_proba(tfidf_u.transform(X_user))[:, 1]
    prob_after_noisy = np.clip(prob_after_u + np.random.laplace(0, 0.1, prob_after_u.shape), 0, 1)

    attack_X2 = np.concatenate([prob_before_u, prob_after_noisy]).reshape(-1, 1)
    attack_y2 = np.concatenate([np.ones(len(prob_before_u)), np.zeros(len(prob_after_noisy))])

    if len(np.unique(attack_y2)) < 2 or len(attack_y2) < 4:
        reid_acc, reid_auc = 0.5, 0.5
    else:
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            attack_X2, attack_y2, test_size=0.3, stratify=attack_y2, random_state=42
        )
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_tr2, y_tr2)
        pred2    = rf.predict(X_te2)
        proba2   = rf.predict_proba(X_te2)[:, 1]
        reid_acc = accuracy_score(y_te2, pred2)
        reid_auc = roc_auc_score(y_te2, proba2) if len(np.unique(y_te2)) > 1 else 0.5

    auc_privacy         = 1 - abs(reid_auc - 0.5) * 2
    confidence_signal   = np.mean(np.abs(prob_before_u - prob_after_noisy))
    final_privacy_score = (
        0.4 * max(0, mia_before - mia_after) +
        0.3 * auc_privacy +
        0.2 * confidence_signal +
        0.1 * prediction_change
    )

    if final_privacy_score > 0.4:
        verdict = ("✅ STRONG PRIVACY", "strong")
    elif final_privacy_score > 0.25:
        verdict = ("⚠️ MODERATE PRIVACY", "moderate")
    else:
        verdict = ("❌ WEAK PRIVACY — LEAKAGE EXISTS", "weak")

    progress_bar.progress(100)
    status_text.markdown('<div class="step-badge">✔ PIPELINE COMPLETE</div>', unsafe_allow_html=True)

    return {
        "df": df,
        "selected_users": selected_users,
        "user_count": len(selected_users),
        "user_sample_count": len(user_data),
        "baseline_acc": baseline_acc,
        "prediction_change": prediction_change,
        "confidence_drop": confidence_drop,
        "mia_before": mia_before,
        "mia_after": mia_after,
        "forgetting_score": forgetting_score,
        "importance_before": importance_before.head(15),
        "importance_after": importance_after.head(15),
        "residual_words": list(residual_words)[:10],
        "residual_count": len(residual_words),
        "reid_acc": reid_acc,
        "reid_auc": reid_auc,
        "final_privacy_score": final_privacy_score,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
# SPLASH
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
elif st.session_state.phase == "upload":
    st.markdown("""
    <div class="upload-wrapper">
        <div class="upload-title">🧠 UnlearnIQ</div>
        <div class="upload-sub">Upload your labeled comment dataset to begin</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="Supports Kaggle toxic comment format (comment_text + toxic columns) or tweet/class format"
        )
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            has_text  = 'comment_text' in df_raw.columns or 'tweet' in df_raw.columns
            has_label = any(c in df_raw.columns for c in TOXIC_COLS + ['class', 'label'])

            if not (has_text and has_label):
                st.error("❌ CSV must have a text column (`comment_text` or `tweet`) and at least one label column (`toxic`, `class`, or `label`).")
            else:
                st.success(f"✅ Dataset loaded — {len(df_raw):,} rows detected")
                st.dataframe(df_raw.head(5), use_container_width=True)
                st.session_state.df_raw = df_raw
                st.session_state.phase  = "user_select"
                st.rerun()


# ─────────────────────────────────────────────
# USER SELECT
# ─────────────────────────────────────────────
elif st.session_state.phase == "user_select":
    st.markdown('<div style="text-align:center; font-family:Orbitron,monospace; font-size:2rem; font-weight:900; background:linear-gradient(135deg,#00F5FF,#0080FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem;">🧠 UnlearnIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Select Users to Unlearn</div>', unsafe_allow_html=True)

    df_raw      = st.session_state.df_raw
    unique_users = sorted(['user_' + str(i) for i in range(100)])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_raw):,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">100</div><div class="metric-label">Unique Users</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">~{len(df_raw)//100}</div><div class="metric-label">Rows / User</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="user-select-box">
        <div class="user-select-title">🎯 Choose one or more users whose data will be erased from the model</div>
        <div style="color:#6AABFF; font-size:0.9rem; margin-top:0.3rem;">Each user represents a synthetic partition of the dataset. Select multiple to perform batch unlearning.</div>
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Select users to unlearn:",
        options=unique_users,
        default=[unique_users[0]],
        help="Hold Ctrl/Cmd to select multiple users. All selected users' data will be removed from the retrained model."
    )

    if selected:
        st.info(f"🗑 **{len(selected)} user(s) selected:** {', '.join(selected)}  |  Approx. **{len(selected) * (len(df_raw)//100):,} rows** will be unlearned")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.phase = "upload"
            st.rerun()
    with col_b:
        run_disabled = len(selected) == 0
        if st.button("🚀  Run UnlearnIQ Pipeline", use_container_width=True, disabled=run_disabled):
            st.session_state.selected_users = selected
            st.session_state.phase          = "processing"
            st.rerun()


# ─────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────
elif st.session_state.phase == "processing":
    st.markdown('<div class="section-title">⚙ Running Pipeline</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text  = st.empty()

    results = run_pipeline(
        st.session_state.df_raw.copy(),
        st.session_state.selected_users,
        progress_bar,
        status_text
    )
    st.session_state.results = results
    st.session_state.phase   = "results"
    time.sleep(0.5)
    st.rerun()


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
elif st.session_state.phase == "results":
    r = st.session_state.results

    st.markdown('<div style="text-align:center; font-family:Orbitron,monospace; font-size:2.5rem; font-weight:900; background:linear-gradient(135deg,#00F5FF,#0080FF,#8B5CF6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.2rem;">UnlearnIQ</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#6AABFF; letter-spacing:0.2em; font-size:0.9rem; margin-bottom:0.5rem;">ANALYSIS COMPLETE</div>', unsafe_allow_html=True)

    users_str = ", ".join(r["selected_users"])
    st.markdown(f'<div style="text-align:center; background:linear-gradient(135deg,#0D1F3C,#0B1628); border:1px solid #0080FF55; border-radius:10px; padding:0.7rem; color:#6AABFF; font-size:0.95rem; margin-bottom:0.5rem;">🗑 <b style="color:#00F5FF">{r["user_count"]} user(s) unlearned:</b> {users_str} &nbsp;|&nbsp; <b style="color:#00F5FF">{r["user_sample_count"]:,}</b> samples removed</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Dataset Overview
    df = r["df"]
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["label"].sum():,}</div><div class="metric-label">Toxic</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{(df["label"]==0).sum():,}</div><div class="metric-label">Non-Toxic</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["user_count"]}</div><div class="metric-label">Users Unlearned</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Baseline
    st.markdown('<div class="section-title">🤖 Baseline Model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><div class="metric-value">{r["baseline_acc"]:.3f}</div><div class="metric-label">Baseline Accuracy (Before Unlearning)</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Unlearning Metrics
    st.markdown('<div class="section-title">🗑 Unlearning Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["prediction_change"]:.3f}</div><div class="metric-label">Prediction Change</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["confidence_drop"]:.3f}</div><div class="metric-label">Confidence Drop</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["mia_before"]:.3f}</div><div class="metric-label">MIA Before</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["mia_after"]:.3f}</div><div class="metric-label">MIA After</div></div>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(r["forgetting_score"], 4),
        title={'text': "Forgetting Score", 'font': {'color': '#6AABFF', 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': '#6AABFF'},
            'bar': {'color': '#00F5FF'},
            'bgcolor': '#0B1628', 'bordercolor': '#1A3A6A',
            'steps': [
                {'range': [0, 0.33], 'color': '#1A0A0A'},
                {'range': [0.33, 0.66], 'color': '#0A1A2A'},
                {'range': [0.66, 1.0], 'color': '#0A2A1A'},
            ],
            'threshold': {'line': {'color': '#FF4D4D', 'width': 2}, 'value': 0.5}
        },
        number={'font': {'color': '#00F5FF', 'family': 'Orbitron'}}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font_color='#E0EAF8', height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-title">📌 Feature Importance</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**Before Unlearning**")
        top_b = r["importance_before"].head(10)
        fig_b = px.bar(top_b, x='weight', y='word', orientation='h',
                       color='weight', color_continuous_scale=['#FF4D4D', '#FFB300', '#00F5FF'],
                       labels={'weight': 'Coefficient', 'word': 'Word'})
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font_color='#E0EAF8', height=350, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_b, use_container_width=True)
    with fc2:
        st.markdown("**After Unlearning**")
        top_a = r["importance_after"].head(10)
        fig_a = px.bar(top_a, x='weight', y='word', orientation='h',
                       color='weight', color_continuous_scale=['#8B5CF6', '#0080FF', '#00F5FF'],
                       labels={'weight': 'Coefficient', 'word': 'Word'})
        fig_a.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font_color='#E0EAF8', height=350, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_a, use_container_width=True)

    st.info(f"🔍 **Residual Influence Words:** {r['residual_count']} found — {', '.join(r['residual_words']) if r['residual_words'] else 'None'}")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Privacy Score
    st.markdown('<div class="section-title">🔒 Privacy Score</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["reid_acc"]:.3f}</div><div class="metric-label">Re-ID Accuracy</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{r["reid_auc"]:.3f}</div><div class="metric-label">Re-ID AUC</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#8B5CF6">{r["final_privacy_score"]:.4f}</div><div class="metric-label">Final Privacy Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    verdict_text, verdict_type = r["verdict"]
    st.markdown(f'<div class="verdict-{verdict_type}">{verdict_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        if st.button("👤  Change Users & Re-run", use_container_width=True):
            st.session_state.phase = "user_select"
            st.rerun()
    with col_y:
        if st.button("🔄  Upload New Dataset", use_container_width=True):
            for key in ["phase", "df_raw", "results", "selected_users"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

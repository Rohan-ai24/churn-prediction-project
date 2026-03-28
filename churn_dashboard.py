import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d0f1a;
    color: #e8e6df;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #12141f;
    border-right: 1px solid #1f2235;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2a2d45;
    border-radius: 12px;
    padding: 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12141f;
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7b7d9e;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #252849 !important;
    color: #a78bfa !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Selectbox, slider labels */
label { color: #9ca3af !important; font-size: 13px !important; }

/* Section headings */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; color: #e8e6df; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; color: #c4c2d4; }

/* Prediction result box */
.pred-box {
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 16px;
    font-family: 'DM Mono', monospace;
}
.pred-churn   { background: #2d1515; border: 1.5px solid #ef4444; color: #fca5a5; }
.pred-nochurn { background: #0f2d1a; border: 1.5px solid #22c55e; color: #86efac; }

/* Divider */
hr { border-color: #1f2235; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0f1a",
    "axes.facecolor":    "#12141f",
    "axes.edgecolor":    "#2a2d45",
    "axes.labelcolor":   "#9ca3af",
    "xtick.color":       "#9ca3af",
    "ytick.color":       "#9ca3af",
    "text.color":        "#e8e6df",
    "grid.color":        "#1f2235",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
})
ACCENT   = "#a78bfa"
DANGER   = "#ef4444"
SUCCESS  = "#22c55e"
NEUTRAL  = "#6b7280"

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df_clean = df.copy()
    df_clean.drop("customerID", axis=1, inplace=True)
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)
    df_clean["Churn"] = (df_clean["Churn"] == "Yes").astype(int)
    le = LabelEncoder()
    cat_cols = df_clean.select_dtypes(include="object").columns
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col])
    return df, df_clean

@st.cache_resource
def train_models(df_clean):
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sm)
    X_te_sc = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42)
    lr.fit(X_tr_sc, y_tr_sm)
    rf.fit(X_tr_sm, y_tr_sm)
    xgb_m.fit(X_tr_sm, y_tr_sm)

    models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_m}
    results = {}
    for name, mdl in models.items():
        if name == "Logistic Regression":
            preds = mdl.predict(X_te_sc)
            probs = mdl.predict_proba(X_te_sc)[:, 1]
        else:
            preds = mdl.predict(X_test)
            probs = mdl.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = {
            "preds": preds, "probs": probs,
            "auc":  roc_auc_score(y_test, probs),
            "report": report,
            "fpr_tpr": roc_curve(y_test, probs),
            "cm": confusion_matrix(y_test, preds),
        }
    feat_imp = pd.Series(xgb_m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return models, scaler, X_test, X_te_sc, y_test, results, feat_imp, X.columns.tolist()

# ── Sidebar — file upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Telco CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**About**")
    st.caption("Train 3 ML models on your Telco churn data, explore EDA, compare performance, and predict individual customers.")

if uploaded is None:
    st.markdown("# 📡 Customer Churn Predictor")
    st.info("👈 Upload your **Telco-Customer-Churn.csv** in the sidebar to begin.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw, df_clean = load_and_preprocess(uploaded)

with st.spinner("🔧 Training models — this takes ~20 seconds..."):
    models, scaler, X_test, X_te_sc, y_test, results, feat_imp, feature_cols = train_models(df_clean)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Customer Churn Predictor")
st.caption(f"Dataset: **{df_raw.shape[0]:,} customers · {df_raw.shape[1]} features**")

# ── KPI row ───────────────────────────────────────────────────────────────────
churn_rate = (df_raw["Churn"] == "Yes").mean()
best_auc   = max(v["auc"] for v in results.values())
best_model = max(results, key=lambda k: results[k]["auc"])
mc_churned = df_raw[df_raw["Churn"] == "Yes"]["MonthlyCharges"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",  f"{df_raw.shape[0]:,}")
c2.metric("Churn Rate",       f"{churn_rate:.1%}")
c3.metric("Best ROC-AUC",     f"{best_auc:.4f}", delta=best_model)
c4.metric("Avg Churner Bill", f"${mc_churned:.2f}/mo")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🏆 Model Performance", "🔍 Feature Importance", "🎯 Predict Customer"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Exploratory Data Analysis")

    # Churn distribution
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df_raw["Churn"].value_counts()
        bars = ax.bar(["No Churn", "Churn"], counts.values,
                      color=[SUCCESS, DANGER], edgecolor="none", width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                    f"{val:,}", ha="center", fontsize=11, color="#e8e6df")
        ax.set_title("Churn Distribution", pad=10)
        ax.set_ylabel("Customers")
        ax.grid(axis="y")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        cat_feature = st.selectbox("Churn rate by feature",
            ["Contract", "InternetService", "PaymentMethod",
             "gender", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rate = df_raw.groupby(cat_feature)["Churn"].apply(lambda x: (x == "Yes").mean())
        rate.sort_values().plot(kind="barh", ax=ax, color=ACCENT, edgecolor="none")
        ax.set_xlabel("Churn Rate")
        ax.set_title(f"Churn Rate by {cat_feature}", pad=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Numeric KDE
    st.markdown("#### Numeric feature distributions by churn")
    df_plot = df_raw.copy()
    df_plot["TotalCharges"] = pd.to_numeric(df_plot["TotalCharges"], errors="coerce")
    df_plot["TotalCharges"].fillna(df_plot["TotalCharges"].median(), inplace=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        for label, color in [("No", SUCCESS), ("Yes", DANGER)]:
            data = df_plot[df_plot["Churn"] == label][col].dropna()
            data.plot(kind="kde", ax=ax, label=label, color=color, linewidth=2)
        ax.set_title(col)
        ax.set_ylabel("")
        ax.legend(title="Churn", fontsize=9)
        ax.grid(True)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Comparison")

    # AUC summary cards
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        cols[i].metric(name, f"AUC {res['auc']:.4f}",
                       delta="Best ✓" if name == best_model else None)

    col_l, col_r = st.columns(2)

    # ROC curves
    with col_l:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        colors = [ACCENT, SUCCESS, DANGER]
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = res["fpr_tpr"]
            ax.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})", color=color, linewidth=2)
        ax.plot([0,1],[0,1], "--", color=NEUTRAL, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Confusion matrix for selected model
    with col_r:
        sel_model = st.selectbox("Confusion matrix for", list(results.keys()))
        cm = results[sel_model]["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["No Churn","Churn"],
                    yticklabels=["No Churn","Churn"],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {sel_model}")
        st.pyplot(fig)
        plt.close()

    # Classification reports
    st.markdown("#### Classification reports")
    rcols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        rep = res["report"]
        with rcols[i]:
            st.markdown(f"**{name}**")
            report_df = pd.DataFrame(rep).T.loc[["0","1","macro avg"], ["precision","recall","f1-score","support"]]
            report_df.index = ["No Churn","Churn","Macro avg"]
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Feature Importance — XGBoost")
    top_n = st.slider("Show top N features", 5, 20, 15)
    fi = feat_imp.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, top_n * 0.45 + 1.5))
    colors_fi = [ACCENT if i >= len(fi) - 3 else "#3b3f6e" for i in range(len(fi))]
    bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="none", height=0.65)
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color="#e8e6df")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(axis="x")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Insights")
    top3 = feat_imp.head(3).index.tolist()
    st.info(f"🔑 **Top drivers of churn:** `{'`, `'.join(top3)}`  \n"
            f"These features have the strongest influence on whether a customer churns.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predict Customer
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Predict Individual Customer Churn")
    st.caption("Fill in the customer details below to get a churn probability.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender          = st.selectbox("Gender", ["Male", "Female"])
            senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner         = st.selectbox("Partner", ["Yes", "No"])
            dependents      = st.selectbox("Dependents", ["Yes", "No"])

        with c2:
            st.markdown("**Services**")
            phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies= st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with c3:
            st.markdown("**Account**")
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment         = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0, 0.5)
            total_charges   = st.slider("Total Charges ($)", 0.0, 8700.0,
                                        float(tenure * monthly_charges), 10.0)
            pred_model      = st.selectbox("Model to use", list(models.keys()))

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        # Build raw input row
        input_raw = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protect,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        # Encode with same approach as training
        df_input = pd.DataFrame([input_raw])
        le2 = LabelEncoder()
        df_ref = df_raw.drop(columns=["customerID", "Churn"])
        for col in df_input.select_dtypes(include="object").columns:
            combined = pd.concat([df_ref[col], df_input[col]], ignore_index=True)
            le2.fit(combined)
            df_input[col] = le2.transform(df_input[col])

        df_input = df_input[feature_cols]

        mdl = models[pred_model]
        if pred_model == "Logistic Regression":
            prob = mdl.predict_proba(scaler.transform(df_input))[0, 1]
        else:
            prob = mdl.predict_proba(df_input)[0, 1]

        churned = prob >= 0.5
        label   = "⚠️ HIGH CHURN RISK" if churned else "✅ LOW CHURN RISK"
        css_cls = "pred-churn" if churned else "pred-nochurn"

        st.markdown(f"""
        <div class="pred-box {css_cls}">
            <div style="font-size:22px;font-weight:700;margin-bottom:8px">{label}</div>
            <div style="font-size:15px">Churn Probability: <strong>{prob:.1%}</strong></div>
            <div style="font-size:13px;margin-top:6px;opacity:0.8">Model: {pred_model}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 0.7))
        ax.barh([0], [1], color="#1f2235", height=0.5)
        ax.barh([0], [prob], color=DANGER if churned else SUCCESS, height=0.5)
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#0d0f1a")
        fig.patch.set_facecolor("#0d0f1a")
        ax.text(prob, 0, f" {prob:.1%}", va="center", color="#e8e6df", fontsize=11)
        st.pyplot(fig)
        plt.close()
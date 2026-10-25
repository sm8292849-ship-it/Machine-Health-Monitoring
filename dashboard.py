
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION AND DESIGN
# ============================================================

st.set_page_config(
    page_title="AEGIS | Machine Health Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#10233F"
BLUE = "#3478F6"
CYAN = "#28C7D9"
GREEN = "#20B486"
AMBER = "#F5A623"
RED = "#E45858"
MUTED = "#7C8AA5"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

.stApp {
    background: #F3F6FB;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #10233F;
}
[data-testid="stSidebar"] * {
    color: #EAF1FB;
}
.hero {
    padding: 32px;
    border-radius: 24px;
    margin-bottom: 24px;
    color: white;
    background: linear-gradient(120deg,#10233F,#174B7A,#267EA1);
    box-shadow: 0 15px 35px rgba(16,35,63,.15);
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 12px 0;
    color: white;
}
.hero p {
    max-width: 850px;
    color: #D3E2F2;
    font-size: 1rem;
    line-height: 1.7;
}
.chip {
    display: inline-block;
    padding: 7px 12px;
    margin: 8px 6px 0 0;
    border: 1px solid #FFFFFF45;
    border-radius: 30px;
    color: white;
    background: #FFFFFF12;
    font-size: .78rem;
}
.kicker {
    color: #3478F6;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
}
h2, h3 {
    color: #10233F;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E3EAF4;
    border-radius: 16px;
    padding: 17px;
    box-shadow: 0 4px 15px #10233F08;
}
div[data-testid="stMetricLabel"] p {
    color: #71809A;
    font-size: .8rem;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #10233F;
    font-weight: 800;
}
div[data-testid="stTabs"] button {
    font-weight: 700;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #E3EAF4;
    border-radius: 12px;
}
div.stButton > button,
div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 2.5rem;
}
</style>
""", unsafe_allow_html=True)


def header(kicker, title, subtitle=""):
    st.markdown(
        f'<div class="kicker">{kicker}</div><h2>{title}</h2>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def chart_style(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=NAVY),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8EDF5")
    fig.update_yaxes(showgrid=True, gridcolor="#E8EDF5")
    return fig


def classify_risk(probability, moderate, high):
    if probability >= high:
        return "HIGH RISK"
    if probability >= moderate:
        return "MODERATE"
    return "LOW RISK"


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"

st.markdown("""
<div class="hero">
    <div style="letter-spacing:3px;font-size:.75rem;color:#8FE5F1;font-weight:800">
        AEGIS • PREDICTIVE MAINTENANCE INTELLIGENCE
    </div>
    <h1>Machine Health<br>Monitoring Platform</h1>
    <p>
        Monitor engine health, estimate failure risk, forecast remaining
        useful life, detect unusual sensor patterns, and create maintenance
        priorities using machine learning.
    </p>
    <span class="chip">Random Forest ML</span>
    <span class="chip">Failure Prediction</span>
    <span class="chip">RUL Forecasting</span>
    <span class="chip">Anomaly Detection</span>
    <span class="chip">Maintenance Analytics</span>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info(
        "Place processed_data.csv inside a folder named data, "
        "next to dashboard.py."
    )
    st.stop()


@st.cache_data
def load_dataset(path):
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]
    return data


df = load_dataset(str(DATA_PATH))

required_columns = {
    "engine_id", "cycle", "RUL", "failure_label"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.write("Columns available:", list(df.columns))
    st.stop()

feature_columns = [
    c for c in df.columns
    if ("sensor" in c.lower() or "setting" in c.lower())
    and pd.api.types.is_numeric_dtype(df[c])
]

if not feature_columns:
    st.error(
        "No numeric sensor/setting features found in the CSV."
    )
    st.stop()

df = df.copy()

for col in [
    "engine_id", "cycle", "RUL", "failure_label"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=["engine_id", "cycle", "RUL", "failure_label"]
).copy()

df["engine_id"] = df["engine_id"].astype(int)
df["cycle"] = df["cycle"].astype(int)
df["failure_label"] = df["failure_label"].astype(int)

df = df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

feature_columns = [
    c for c in feature_columns
    if df[c].notna().any() and df[c].nunique(dropna=True) > 1
]

if not feature_columns:
    st.error("Sensor columns are empty or constant.")
    st.stop()

if df["failure_label"].nunique() < 2:
    st.error(
        "failure_label must contain both classes, normally 0 and 1, "
        "to train a failure classifier."
    )
    st.stop()

RUL_CAP = 125
df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)


# ============================================================
# 3. MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource(show_spinner="Training machine learning models...")
def train_models(data, features):
    work = data.copy()

    X = work[features]
    y_class = work["failure_label"].astype(int)
    y_rul = work["RUL_capped"].astype(float)
    groups = work["engine_id"]

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=features,
        index=X.index,
    )

    # Keep each engine in either training or testing split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_idx, test_idx = next(
        splitter.split(X_imp, y_class, groups=groups)
    )

    classifier_eval = RandomForestClassifier(
        n_estimators=140,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    classifier_eval.fit(
        X_imp.iloc[train_idx],
        y_class.iloc[train_idx],
    )

    class_predictions = classifier_eval.predict(
        X_imp.iloc[test_idx]
    )
    classes = list(classifier_eval.classes_)
    positive_col = classes.index(1) if 1 in classes else 0
    class_probabilities = classifier_eval.predict_proba(
        X_imp.iloc[test_idx]
    )[:, positive_col]

    y_test = y_class.iloc[test_idx]
    matrix = confusion_matrix(
        y_test, class_predictions, labels=[0, 1]
    )

    metrics_class = {
        "accuracy": accuracy_score(y_test, class_predictions),
        "precision": precision_score(
            y_test, class_predictions,
            pos_label=1, zero_division=0,
        ),
        "recall": recall_score(
            y_test, class_predictions,
            pos_label=1, zero_division=0,
        ),
        "f1": f1_score(
            y_test, class_predictions,
            pos_label=1, zero_division=0,
        ),
        "auc": (
            roc_auc_score(y_test, class_probabilities)
            if y_test.nunique() == 2 else np.nan
        ),
        "matrix": matrix,
        "test_engines": int(
            work.iloc[test_idx]["engine_id"].nunique()
        ),
    }

    # Separate engine-wise split for RUL regression.
    rul_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=21,
    )
    rul_train, rul_test = next(
        rul_split.split(X_imp, y_rul, groups=groups)
    )

    regressor_eval = RandomForestRegressor(
        n_estimators=140,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    regressor_eval.fit(
        X_imp.iloc[rul_train],
        y_rul.iloc[rul_train],
    )

    actual_rul = y_rul.iloc[rul_test].to_numpy()
    predicted_rul_test = regressor_eval.predict(
        X_imp.iloc[rul_test]
    )

    metrics_rul = {
        "mae": mean_absolute_error(
            actual_rul, predicted_rul_test
        ),
        "rmse": float(np.sqrt(mean_squared_error(
            actual_rul, predicted_rul_test
        ))),
        "r2": r2_score(actual_rul, predicted_rul_test),
        "actual": actual_rul,
        "predicted": predicted_rul_test,
    }

    # Final models score all records for dashboard exploration.
    classifier = RandomForestClassifier(
        n_estimators=140,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_imp, y_class)

    regressor = RandomForestRegressor(
        n_estimators=140,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_imp, y_rul)

    classes_final = list(classifier.classes_)
    positive_final = (
        classes_final.index(1) if 1 in classes_final else 0
    )

    scored = work.copy()
    scored["failure_probability"] = (
        classifier.predict_proba(X_imp)[:, positive_final] * 100
    )
    scored["predicted_rul"] = np.clip(
        regressor.predict(X_imp), 0, RUL_CAP
    )

    anomaly_model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=120,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    )
    anomaly_model.fit(X_imp)

    isolation_forest = anomaly_model.named_steps[
        "isolationforest"
    ]
    scaled_X = anomaly_model.named_steps[
        "standardscaler"
    ].transform(X_imp)

    scored["anomaly_score"] = (
        -isolation_forest.score_samples(scaled_X)
    )
    scored["is_anomaly"] = (
        isolation_forest.predict(scaled_X) == -1
    )

    importance = pd.Series(
        classifier.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    return scored, metrics_class, metrics_rul, importance


scored, class_metrics, rul_metrics, importance = train_models(
    df, feature_columns
)

engines = sorted(scored["engine_id"].unique().tolist())
max_cycle = int(scored["cycle"].max())

latest = (
    scored.sort_values("cycle")
    .groupby("engine_id", as_index=False)
    .tail(1)
    .copy()
)


# ============================================================
# 4. SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("# ⚙️ AEGIS")
    st.caption("MACHINE HEALTH INTELLIGENCE")
    st.divider()

    selected_engine = st.selectbox(
        "Select engine",
        engines,
    )

    engine_data = (
        scored[scored["engine_id"] == selected_engine]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    selected_cycle = st.slider(
        "Operating cycle",
        int(engine_data["cycle"].min()),
        int(engine_data["cycle"].max()),
        int(engine_data["cycle"].median()),
    )

    moderate_threshold = st.slider(
        "Moderate risk threshold (%)",
        10, 60, 30, 5,
    )
    high_threshold = st.slider(
        "High risk threshold (%)",
        35, 95, 60, 5,
    )

    if high_threshold <= moderate_threshold:
        st.warning(
            "High threshold should be above moderate threshold."
        )
        high_threshold = moderate_threshold + 5

    st.divider()
    st.caption("DATASET INFORMATION")
    st.write(f"Engines: {len(engines)}")
    st.write(f"Records: {len(scored):,}")
    st.write(f"Features: {len(feature_columns)}")
    st.caption(
        "Historical/demo data — not a live IoT connection."
    )


row = engine_data.loc[
    engine_data["cycle"] == selected_cycle
].iloc[0]

probability = float(row["failure_probability"])
predicted_rul = float(row["predicted_rul"])
health_score = float(np.clip(
    predicted_rul / RUL_CAP * 100, 0, 100
))
current_status = classify_risk(
    probability, moderate_threshold, high_threshold
)


# ============================================================
# 5. NAVIGATION
# ============================================================

tabs = st.tabs([
    "🏠 Fleet Overview",
    "🚨 Fleet Risk",
    "⚙️ Engine Health",
    "📈 Sensor Analysis",
    "🛠️ Maintenance Planner",
    "🧪 What-if Analysis",
    "🧠 Model Insights",
    "📋 Data Explorer",
    "▶ Live Simulation",
])


# ============================================================
# 6. FLEET OVERVIEW
# ============================================================

with tabs[0]:
    header(
        "EXECUTIVE COMMAND CENTER",
        "Fleet Overview",
        "Summary of the latest available record for every engine.",
    )

    latest["Risk status"] = latest[
        "failure_probability"
    ].apply(
        lambda p: classify_risk(
            p, moderate_threshold, high_threshold
        )
    )

    high_risk_count = int(
        (latest["Risk status"] == "HIGH RISK").sum()
    )
    moderate_count = int(
        (latest["Risk status"] == "MODERATE").sum()
    )
    anomaly_count = int(latest["is_anomaly"].sum())

    mean_health = np.clip(
        latest["predicted_rul"].mean() / RUL_CAP * 100,
        0, 100,
    )

    a, b, c, d, e = st.columns(5)
    a.metric("Engines monitored", f"{len(engines):,}")
    b.metric("Sensor records", f"{len(scored):,}")
    c.metric("High-risk engines", high_risk_count)
    d.metric("Anomaly flags", anomaly_count)
    e.metric("Average health index", f"{mean_health:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        header("FLEET STATUS", "Risk distribution")

        risk_counts = (
            latest["Risk status"]
            .value_counts()
            .reindex(
                ["LOW RISK", "MODERATE", "HIGH RISK"],
                fill_value=0,
            )
            .rename_axis("Risk status")
            .reset_index(name="Engines")
        )

        fig = px.pie(
            risk_counts,
            names="Risk status",
            values="Engines",
            hole=0.60,
            color="Risk status",
            color_discrete_map={
                "LOW RISK": GREEN,
                "MODERATE": AMBER,
                "HIGH RISK": RED,
            },
        )
        st.plotly_chart(
            chart_style(fig, 350),
            use_container_width=True,
        )

    with right:
        header("PRIORITY QUEUE", "Top 10 highest-risk engines")

        top10 = latest.sort_values(
            "failure_probability",
            ascending=False,
        ).head(10).copy()

        top10["Risk status"] = top10[
            "failure_probability"
        ].apply(
            lambda p: classify_risk(
                p, moderate_threshold, high_threshold
            )
        )

        view = top10[[
            "engine_id", "cycle", "Risk status",
            "failure_probability", "predicted_rul",
            "is_anomaly",
        ]].rename(columns={
            "engine_id": "Engine",
            "cycle": "Latest cycle",
            "failure_probability": "Failure probability (%)",
            "predicted_rul": "Predicted RUL",
            "is_anomaly": "Anomaly",
        })

        st.dataframe(
            view.style.format({
                "Failure probability (%)": "{:.1f}",
                "Predicted RUL": "{:.0f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download fleet snapshot CSV",
            csv_bytes(view),
            "fleet_snapshot.csv",
            "text/csv",
        )

    header(
        "FLEET INTELLIGENCE",
        "Engine risk comparison",
        "Each point represents the latest record for one engine.",
    )

    fig = px.scatter(
        latest,
        x="engine_id",
        y="failure_probability",
        color="failure_probability",
        color_continuous_scale=["#20B486", "#F5A623", "#E45858"],
        hover_data=["predicted_rul", "anomaly_score"],
        labels={
            "engine_id": "Engine ID",
            "failure_probability": "Failure probability (%)",
        },
    )
    fig.add_hline(
        y=moderate_threshold,
        line_dash="dot",
        line_color=AMBER,
    )
    fig.add_hline(
        y=high_threshold,
        line_dash="dash",
        line_color=RED,
    )
    st.plotly_chart(
        chart_style(fig, 360),
        use_container_width=True,
    )

    st.info(
        "Failure probability and health index are model estimates. "
        "They are not guarantees of future failure."
    )


# ============================================================
# 7. FLEET RISK REGISTER
# ============================================================

with tabs[1]:
    header(
        "RISK MANAGEMENT",
        "Fleet Risk Register",
        "Filter engines and prepare a maintenance review list.",
    )

    f1, f2 = st.columns(2)
    minimum_probability = f1.slider(
        "Minimum failure probability (%)",
        0, 100, 0, 5,
    )
    show_anomalies_only = f2.checkbox(
        "Show only engines with anomaly flags"
    )

    fleet = latest.copy()
    fleet["Risk status"] = fleet[
        "failure_probability"
    ].apply(
        lambda p: classify_risk(
            p, moderate_threshold, high_threshold
        )
    )

    fleet = fleet[
        fleet["failure_probability"] >= minimum_probability
    ]

    if show_anomalies_only:
        fleet = fleet[fleet["is_anomaly"]]

    fleet = fleet.sort_values(
        "failure_probability",
        ascending=False,
    )

    st.metric("Engines matching filters", len(fleet))

    risk_view = fleet[[
        "engine_id", "cycle", "Risk status",
        "failure_probability", "predicted_rul",
        "anomaly_score", "is_anomaly",
    ]].rename(columns={
        "engine_id": "Engine",
        "cycle": "Latest cycle",
        "failure_probability": "Failure probability (%)",
        "predicted_rul": "Predicted RUL (cycles)",
        "anomaly_score": "Anomaly score",
        "is_anomaly": "Anomaly flagged",
    })

    st.dataframe(
        risk_view.style.format({
            "Failure probability (%)": "{:.1f}",
            "Predicted RUL (cycles)": "{:.1f}",
            "Anomaly score": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered risk register",
        csv_bytes(risk_view),
        "risk_register.csv",
        "text/csv",
    )

    if len(fleet):
        fig = px.bar(
            fleet.sort_values("failure_probability"),
            x="failure_probability",
            y="engine_id",
            orientation="h",
            color="Risk status",
            color_discrete_map={
                "LOW RISK": GREEN,
                "MODERATE": AMBER,
                "HIGH RISK": RED,
            },
            labels={
                "failure_probability": "Failure probability (%)",
                "engine_id": "Engine ID",
            },
        )
        st.plotly_chart(
            chart_style(fig, max(350, min(650, len(fleet) * 24))),
            use_container_width=True,
        )


# ============================================================
# 8. ENGINE HEALTH AND DIGITAL TWIN
# ============================================================

with tabs[2]:
    header(
        "ENGINE DIGITAL TWIN",
        f"Engine {selected_engine} Health",
        "Explore historical cycles and model predictions for one engine.",
    )

    a, b, c, d = st.columns(4)
    a.metric("Failure probability", f"{probability:.1f}%")
    b.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")
    c.metric("Estimated health", f"{health_score:.1f}/100")
    d.metric(
        "Anomaly status",
        "Flagged" if bool(row["is_anomaly"]) else "Not flagged",
    )

    left, right = st.columns([1, 1.5])

    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                number={"suffix": " / 100"},
                title={"text": "Estimated Health Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": BLUE},
                    "steps": [
                        {"range": [0, 30], "color": "#FCE7E7"},
                        {"range": [30, 60], "color": "#FFF0D3"},
                        {"range": [60, 100], "color": "#DDF7ED"},
                    ],
                },
            )
        )
        st.plotly_chart(
            chart_style(gauge, 300),
            use_container_width=True,
        )

        st.write(f"**Risk classification:** {current_status}")
        st.write(f"**Selected cycle:** {selected_cycle}")

        if bool(row["is_anomaly"]):
            st.warning(
                "Unusual sensor pattern detected. "
                "This is an investigation signal, not proof of failure."
            )

    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=engine_data["cycle"],
            y=engine_data["failure_probability"],
            mode="lines",
            name="Failure probability",
            line=dict(color=BLUE, width=3),
        ))
        fig.add_vline(
            x=selected_cycle,
            line_dash="dash",
            line_color=AMBER,
        )
        fig.add_hline(
            y=moderate_threshold,
            line_dash="dot",
            line_color=AMBER,
        )
        fig.add_hline(
            y=high_threshold,
            line_dash="dash",
            line_color=RED,
        )
        fig.update_layout(
            xaxis_title="Operating cycle",
            yaxis_title="Failure probability (%)",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(
            chart_style(fig, 300),
            use_container_width=True,
        )

    header("RUL FORECAST", "Actual vs predicted remaining useful life")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=engine_data["cycle"],
        y=engine_data["RUL_capped"],
        mode="lines",
        name="Actual RUL (capped)",
        line=dict(color=MUTED, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=engine_data["cycle"],
        y=engine_data["predicted_rul"],
        mode="lines",
        name="Predicted RUL",
        line=dict(color=GREEN, width=3),
    ))
    fig.add_vline(
        x=selected_cycle,
        line_dash="dash",
        line_color=AMBER,
    )
    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color=RED,
    )
    fig.update_layout(
        xaxis_title="Operating cycle",
        yaxis_title="Remaining useful life (cycles)",
    )
    st.plotly_chart(
        chart_style(fig, 350),
        use_container_width=True,
    )

    st.markdown("#### Compare two engines")

    other_engines = [e for e in engines if e != selected_engine]

    if other_engines:
        second_engine = st.selectbox(
            "Select another engine",
            other_engines,
        )
        second_data = scored[
            scored["engine_id"] == second_engine
        ].sort_values("cycle")

        comparison = pd.concat([
            engine_data[["cycle", "failure_probability"]]
            .assign(Engine=f"Engine {selected_engine}"),
            second_data[["cycle", "failure_probability"]]
            .assign(Engine=f"Engine {second_engine}"),
        ])

        fig = px.line(
            comparison,
            x="cycle",
            y="failure_probability",
            color="Engine",
            labels={
                "cycle": "Operating cycle",
                "failure_probability": "Failure probability (%)",
            },
        )
        st.plotly_chart(
            chart_style(fig, 330),
            use_container_width=True,
        )


# ============================================================
# 9. SENSOR ANALYSIS AND ANOMALY DETECTION
# ============================================================

with tabs[3]:
    header(
        "SENSOR INTELLIGENCE",
        "Sensor Analysis",
        "Inspect readings, unusual records, correlations and feature importance.",
    )

    selected_features = st.multiselect(
        "Select sensor/setting columns",
        feature_columns,
        default=feature_columns[:min(5, len(feature_columns))],
    )

    if selected_features:
        melted = engine_data[
            ["cycle"] + selected_features
        ].melt(
            id_vars="cycle",
            var_name="Sensor",
            value_name="Reading",
        )

        fig = px.line(
            melted,
            x="cycle",
            y="Reading",
            color="Sensor",
        )
        fig.add_vline(
            x=selected_cycle,
            line_dash="dash",
            line_color=AMBER,
        )
        st.plotly_chart(
            chart_style(fig, 380),
            use_container_width=True,
        )

        current_row = engine_data[
            engine_data["cycle"] == selected_cycle
        ].iloc[0]

        summary = pd.DataFrame({
            "Sensor": selected_features,
            "Current reading": [
                current_row[s] for s in selected_features
            ],
            "Engine mean": [
                engine_data[s].mean() for s in selected_features
            ],
            "Engine standard deviation": [
                engine_data[s].std() for s in selected_features
            ],
            "Feature importance": [
                importance.get(s, 0) for s in selected_features
            ],
        })

        st.dataframe(
            summary.style.format({
                "Current reading": "{:.3f}",
                "Engine mean": "{:.3f}",
                "Engine standard deviation": "{:.3f}",
                "Feature importance": "{:.4f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        header("MODEL EXPLANATION", "Top feature importance")

        top_importance = importance.head(15).sort_values()

        fig = px.bar(
            x=top_importance.values,
            y=top_importance.index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=top_importance.values,
            color_continuous_scale=["#B9E7EB", BLUE],
        )
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            chart_style(fig, 420),
            use_container_width=True,
        )

    with right:
        header("ANOMALY WATCH", "Unusual sensor records")

        anomalies = latest[
            latest["is_anomaly"]
        ].sort_values(
            "anomaly_score",
            ascending=False,
        )

        st.metric(
            "Latest engine records flagged",
            len(anomalies),
        )

        anomaly_view = anomalies[[
            "engine_id", "cycle", "anomaly_score",
            "failure_probability", "predicted_rul",
        ]].rename(columns={
            "engine_id": "Engine",
            "cycle": "Cycle",
            "anomaly_score": "Anomaly score",
            "failure_probability": "Failure probability (%)",
            "predicted_rul": "Predicted RUL",
        })

        st.dataframe(
            anomaly_view,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("View sensor correlation heatmap"):
        corr_features = feature_columns[:30]
        corr = df[corr_features].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(
            chart_style(fig, 600),
            use_container_width=True,
        )

    st.caption(
        "Feature importance shows model reliance, not proof of a physical root cause. "
        "Isolation Forest detects unusual patterns, not confirmed machine faults."
    )


# ============================================================
# 10. MAINTENANCE PLANNER
# ============================================================

with tabs[4]:
    header(
        "MAINTENANCE WORKBENCH",
        "Maintenance Planner",
        "Prioritise engines for review using risk, predicted RUL and anomaly flags.",
    )

    maintenance = latest.copy()

    maintenance["Risk status"] = maintenance[
        "failure_probability"
    ].apply(
        lambda p: classify_risk(
            p, moderate_threshold, high_threshold
        )
    )

    def get_priority(record):
        if (
            record["Risk status"] == "HIGH RISK"
            or record["predicted_rul"] <= 15
        ):
            return "P1 - Review first"

        if (
            record["Risk status"] == "MODERATE"
            or record["predicted_rul"] <= 30
            or record["is_anomaly"]
        ):
            return "P2 - Inspect soon"

        return "P3 - Routine monitoring"

    maintenance["Priority"] = maintenance.apply(
        get_priority, axis=1
    )

    maintenance["Suggested action"] = maintenance[
        "Priority"
    ].map({
        "P1 - Review first":
            "Review model output and arrange qualified inspection",
        "P2 - Inspect soon":
            "Review sensor trends and plan inspection",
        "P3 - Routine monitoring":
            "Continue scheduled monitoring",
    })

    maintenance = maintenance.sort_values(
        ["Priority", "failure_probability"],
        ascending=[True, False],
    )

    a, b, c = st.columns(3)
    a.metric(
        "P1 - Review first",
        int((maintenance["Priority"] == "P1 - Review first").sum()),
    )
    b.metric(
        "P2 - Inspect soon",
        int((maintenance["Priority"] == "P2 - Inspect soon").sum()),
    )
    c.metric(
        "P3 - Routine",
        int((maintenance["Priority"] == "P3 - Routine").sum()),
    )

    maintenance_view = maintenance[[
        "engine_id", "cycle", "Priority",
        "failure_probability", "predicted_rul",
        "is_anomaly", "Suggested action",
    ]].rename(columns={
        "engine_id": "Engine",
        "cycle": "Latest cycle",
        "failure_probability": "Failure probability (%)",
        "predicted_rul": "Predicted RUL (cycles)",
        "is_anomaly": "Anomaly flagged",
    })

    st.dataframe(
        maintenance_view.style.format({
            "Failure probability (%)": "{:.1f}",
            "Predicted RUL (cycles)": "{:.1f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download maintenance action report",
        csv_bytes(maintenance_view),
        "maintenance_action_report.csv",
        "text/csv",
    )

    st.info(
        "Maintenance priorities are educational model-based suggestions. "
        "Real equipment must be assessed by qualified personnel."
    )


# ============================================================
# 11. WHAT-IF ANALYSIS
# ============================================================

with tabs[5]:
    header(
        "SCENARIO LAB",
        "What-if Risk Analysis",
        "Explore how changing alert thresholds affects the number of engines flagged.",
    )

    a, b = st.columns(2)

    scenario_moderate = a.slider(
        "Scenario moderate threshold (%)",
        10, 70, moderate_threshold, 5,
        key="scenario_moderate",
    )

    scenario_high = b.slider(
        "Scenario high-risk threshold (%)",
        35, 95,
        max(high_threshold, scenario_moderate + 5),
        5,
        key="scenario_high",
    )

    if scenario_high <= scenario_moderate:
        st.warning(
            "Choose a high-risk threshold greater than the moderate threshold."
        )
        scenario_high = scenario_moderate + 5

    scenario = latest.copy()
    scenario["Scenario status"] = scenario[
        "failure_probability"
    ].apply(
        lambda p: classify_risk(
            p, scenario_moderate, scenario_high
        )
    )

    counts = scenario["Scenario status"].value_counts()

    a, b, c = st.columns(3)
    a.metric("Low risk", int(counts.get("LOW RISK", 0)))
    b.metric("Moderate", int(counts.get("MODERATE", 0)))
    c.metric("High risk", int(counts.get("HIGH RISK", 0)))

    chart_data = (
        scenario["Scenario status"]
        .value_counts()
        .reindex(
            ["LOW RISK", "MODERATE", "HIGH RISK"],
            fill_value=0,
        )
        .rename_axis("Status")
        .reset_index(name="Engines")
    )

    fig = px.bar(
        chart_data,
        x="Status",
        y="Engines",
        color="Status",
        text="Engines",
        color_discrete_map={
            "LOW RISK": GREEN,
            "MODERATE": AMBER,
            "HIGH RISK": RED,
        },
    )

    st.plotly_chart(
        chart_style(fig, 330),
        use_container_width=True,
    )

    st.dataframe(
        scenario[[
            "engine_id", "failure_probability",
            "predicted_rul", "Scenario status",
        ]].sort_values(
            "failure_probability", ascending=False
        ).rename(columns={
            "engine_id": "Engine",
            "failure_probability": "Failure probability (%)",
            "predicted_rul": "Predicted RUL",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "This changes alert thresholds only; it does not retrain the model "
        "or change the underlying sensor readings."
    )


# ============================================================
# 12. MODEL INSIGHTS AND VALIDATION
# ============================================================

with tabs[6]:
    header(
        "MODEL GOVERNANCE",
        "Model Insights & Validation",
        "Classification metrics, RUL prediction errors and feature importance.",
    )

    st.info(
        "Evaluation uses held-out engines to reduce leakage between "
        "training and testing. Final dashboard scores use models trained "
        "on all available historical records."
    )

    st.subheader("Failure prediction classifier")

    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{class_metrics['accuracy'] * 100:.1f}%")
    b.metric("Precision", f"{class_metrics['precision']:.3f}")
    c.metric("Recall", f"{class_metrics['recall']:.3f}")
    d.metric("F1 score", f"{class_metrics['f1']:.3f}")
    e.metric(
        "ROC AUC",
        (
            f"{class_metrics['auc']:.3f}"
            if np.isfinite(class_metrics["auc"])
            else "N/A"
        ),
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Confusion matrix")

        matrix = class_metrics["matrix"]

        fig = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted safe", "Predicted failure"],
            y=["Actual safe", "Actual failure"],
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            chart_style(fig, 330),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Most influential features")

        top = importance.head(12).sort_values()

        fig = px.bar(
            x=top.values,
            y=top.index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=top.values,
            color_continuous_scale=["#B9E7EB", BLUE],
        )
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            chart_style(fig, 330),
            use_container_width=True,
        )

    st.subheader("Remaining Useful Life regression")

    a, b, c = st.columns(3)
    a.metric("MAE", f"{rul_metrics['mae']:.2f} cycles")
    b.metric("RMSE", f"{rul_metrics['rmse']:.2f} cycles")
    c.metric("R² score", f"{rul_metrics['r2']:.3f}")

    predictions = pd.DataFrame({
        "Actual RUL": rul_metrics["actual"],
        "Predicted RUL": rul_metrics["predicted"],
    })

    fig = px.scatter(
        predictions,
        x="Actual RUL",
        y="Predicted RUL",
        opacity=0.5,
    )

    fig.add_trace(go.Scatter(
        x=[0, RUL_CAP],
        y=[0, RUL_CAP],
        mode="lines",
        name="Ideal prediction",
        line=dict(color=RED, dash="dash"),
    ))

    st.plotly_chart(
        chart_style(fig, 360),
        use_container_width=True,
    )

    with st.expander("Metric definitions and limitations"):
        st.markdown("""
        - **Accuracy:** fraction of test records classified correctly.
        - **Precision:** proportion of predicted failures that were labelled failures.
        - **Recall:** proportion of labelled failures detected by the model.
        - **F1:** balance between precision and recall.
        - **ROC AUC:** ranking quality when both classes occur in test data.
        - **MAE:** average absolute RUL prediction error.
        - **RMSE:** prediction error with a larger penalty for large errors.
        - **R²:** regression score relative to a mean-value baseline.
        - **RUL cap:** early-life RUL values are capped at 125 cycles.
        - **Feature importance:** model reliance, not a causal explanation.
        - **Anomaly detection:** unusual records, not confirmed physical faults.
        """)


# ============================================================
# 13. DATA EXPLORER
# ============================================================

with tabs[7]:
    header(
        "DATA QUALITY",
        "Data Explorer",
        "Inspect records, data types, missing values and feature coverage.",
    )

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Unique engines", df["engine_id"].nunique())
    d.metric("Sensor/setting features", len(feature_columns))

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[col].dtype) for col in df.columns],
        "Missing values": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "Unique values": [
            int(df[col].nunique(dropna=True)) for col in df.columns
        ],
    })

    st.subheader("Dataset quality report")
    st.dataframe(
        quality,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Dataset preview")
    st.dataframe(
        df.head(100),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download first 100 records",
        csv_bytes(df.head(100)),
        "dataset_preview.csv",
        "text/csv",
    )

    st.download_button(
        "Download complete processed dataset",
        csv_bytes(df),
        "processed_data_export.csv",
        "text/csv",
    )


# ============================================================
# 14. LIVE-STYLE HISTORICAL SIMULATION
# ============================================================

with tabs[8]:
    header(
        "HISTORICAL REPLAY",
        "Live Simulation",
        "Replay historical engine cycles and observe changing risk and RUL estimates.",
    )

    st.warning(
        "This is a replay of stored dataset records, not a live IoT sensor connection."
    )

    a, b = st.columns(2)

    simulation_engine = a.selectbox(
        "Engine to simulate",
        engines,
        index=engines.index(selected_engine),
    )

    simulation_speed = b.select_slider(
        "Simulation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
    )

    simulation_data = scored[
        scored["engine_id"] == simulation_engine
    ].sort_values("cycle").reset_index(drop=True)

    if st.button(
        "▶ Start historical replay",
        type="primary",
    ):
        progress = st.progress(0)
        metrics_slot = st.empty()
        chart_slot = st.empty()
        log_slot = st.empty()

        sample_count = min(35, len(simulation_data))
        indices = np.unique(
            np.linspace(
                0,
                len(simulation_data) - 1,
                sample_count,
                dtype=int,
            )
        )

        replay_log = []
        delay = {
            "Slow": 0.20,
            "Normal": 0.08,
            "Fast": 0.01,
        }[simulation_speed]

        for step, idx in enumerate(indices, start=1):
            current = simulation_data.iloc[idx]

            p = float(current["failure_probability"])
            rul = float(current["predicted_rul"])
            health = np.clip(rul / RUL_CAP * 100, 0, 100)

            status = classify_risk(
                p, moderate_threshold, high_threshold
            )

            with metrics_slot.container():
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("Cycle", int(current["cycle"]))
                x2.metric("Failure probability", f"{p:.1f}%")
                x3.metric("Predicted RUL", f"{rul:.1f} cycles")
                x4.metric("Health index", f"{health:.1f}/100")
                st.write(f"**Current risk status:** {status}")

            history = simulation_data.iloc[:idx + 1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history["cycle"],
                y=history["failure_probability"],
                mode="lines",
                name="Failure probability",
                line=dict(color=BLUE, width=3),
            ))
            fig.add_hline(
                y=moderate_threshold,
                line_dash="dot",
                line_color=AMBER,
            )
            fig.add_hline(
                y=high_threshold,
                line_dash="dash",
                line_color=RED,
            )
            fig.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Failure probability (%)",
                yaxis_range=[0, 100],
            )

            chart_slot.plotly_chart(
                chart_style(fig, 330),
                use_container_width=True,
            )

            replay_log.append({
                "Cycle": int(current["cycle"]),
                "Failure probability (%)": round(p, 2),
                "Predicted RUL": round(rul, 2),
                "Health index": round(float(health), 2),
                "Risk status": status,
                "Sensor anomaly": bool(current["is_anomaly"]),
            })

            log_slot.dataframe(
                pd.DataFrame(replay_log).tail(10),
                hide_index=True,
                use_container_width=True,
            )

            progress.progress(
                step / len(indices),
                text=f"Replaying cycle {int(current['cycle'])}...",
            )

            time.sleep(delay)

        st.success("Historical replay completed.")

        st.download_button(
            "Download simulation log",
            csv_bytes(pd.DataFrame(replay_log)),
            f"engine_{simulation_engine}_replay.csv",
            "text/csv",
        )

    else:
        st.info(
            "Choose an engine and click Start historical replay "
            "to watch its historical risk estimates."
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="display:flex;justify-content:space-between;gap:12px;
flex-wrap:wrap;color:#7C8AA5;font-size:.8rem">
    <span><b style="color:#10233F">AEGIS</b> · Machine Health Intelligence</span>
    <span>Academic prototype · Historical dataset · Not a live industrial control system</span>
</div>
""", unsafe_allow_html=True)

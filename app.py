import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Budget Sales Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    border: 1px solid rgba(99,179,237,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { font-size: 2rem; margin-bottom: 0.3rem; font-weight: 800; }
.hero p  { font-size: 1rem; opacity: 0.8; margin: 0; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    color: white;
}
.kpi .label { font-size: 0.75rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 1px; }
.kpi .value { font-size: 1.8rem; font-weight: 800; color: #63b3ed; margin: 0.15rem 0; }
.kpi .sub   { font-size: 0.75rem; opacity: 0.55; }

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #065f46, #064e3b);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    margin: 1rem 0;
}
.pred-box .amount { font-size: 3rem; font-weight: 900; color: #34d399; }
.pred-box .unit   { font-size: 1rem; opacity: 0.7; }

/* Warning / info badges */
.badge-warn { background:#78350f; border:1px solid #d97706; border-radius:8px; padding:0.6rem 1rem; color:#fcd34d; font-size:0.85rem; margin:0.4rem 0; }
.badge-good { background:#064e3b; border:1px solid #34d399; border-radius:8px; padding:0.6rem 1rem; color:#34d399; font-size:0.85rem; margin:0.4rem 0; }
.badge-info { background:#1e3a5f; border:1px solid #63b3ed; border-radius:8px; padding:0.6rem 1rem; color:#93c5fd; font-size:0.85rem; margin:0.4rem 0; }

/* Section headers */
.section-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin: 1.2rem 0 0.6rem; border-left: 3px solid #63b3ed; padding-left: 0.7rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# ── Load & train model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("advertising.csv")
    X = df[["TV", "Radio", "Newspaper"]].values
    y = df["Sales"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test  = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred = model.predict(X_poly_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":  mean_absolute_error(y_test, y_pred),
    }

    # cross-val
    X_full = poly.transform(X)
    cv_scores = cross_val_score(model, X_full, y, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = cv_scores.mean()
    metrics["cv_r2_std"]  = cv_scores.std()

    return model, poly, df, metrics, y_test, y_pred

model, poly, df, metrics, y_test, y_pred = load_model()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Budget Inputs")
    st.markdown("Adjust spending (in $ thousands) per channel:")

    tv_budget = st.slider(
        "📺 TV Budget ($K)", 0.0, 300.0,
        value=150.0, step=0.5,
        help="Typical range: $0 – $296K"
    )
    radio_budget = st.slider(
        "📻 Radio Budget ($K)", 0.0, 50.0,
        value=25.0, step=0.5,
        help="Typical range: $0 – $50K"
    )
    newspaper_budget = st.slider(
        "📰 Newspaper Budget ($K)", 0.0, 115.0,
        value=30.0, step=0.5,
        help="Typical range: $0 – $114K"
    )

    total_budget = tv_budget + radio_budget + newspaper_budget
    st.markdown(f"**Total Budget:** `${total_budget:.1f}K`")
    st.markdown("---")
    st.markdown("#### 📊 View Options")
    show_data      = st.checkbox("Show raw dataset",    value=False)
    show_residuals = st.checkbox("Show residual plot",  value=False)
    show_cv        = st.checkbox("Show CV details",     value=False)

# ── Prediction ────────────────────────────────────────────────────────────────
input_arr   = np.array([[tv_budget, radio_budget, newspaper_budget]])
input_poly  = poly.transform(input_arr)
pred_sales  = model.predict(input_poly)[0]
pred_sales  = max(0, pred_sales)

# Diminishing returns thresholds (75th percentile of training data)
TV_THRESH   = df["TV"].quantile(0.75)          # ~218
RADIO_THRESH = df["Radio"].quantile(0.75)      # ~36.5
NEWS_THRESH  = df["Newspaper"].quantile(0.75)  # ~45

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📈 Advertising Sales Predictor</h1>
  <p>Polynomial Regression (Degree 2) · Captures non-linear returns · 200 market observations</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Model R²</div><div class="value">{metrics['r2']:.3f}</div><div class="sub">Test set</div></div>
  <div class="kpi"><div class="label">RMSE</div><div class="value">{metrics['rmse']:.2f}K</div><div class="sub">Sales units</div></div>
  <div class="kpi"><div class="label">MAE</div><div class="value">{metrics['mae']:.2f}K</div><div class="sub">Avg error</div></div>
  <div class="kpi"><div class="label">CV R² (5-fold)</div><div class="value">{metrics['cv_r2_mean']:.3f}</div><div class="sub">±{metrics['cv_r2_std']:.3f}</div></div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    # Prediction result
    st.markdown(f"""
    <div class="pred-box">
      <div class="unit">Predicted Sales</div>
      <div class="amount">{pred_sales:.2f}K</div>
      <div class="unit">units (thousands)</div>
    </div>
    """, unsafe_allow_html=True)

    # Budget allocation donut
    fig_donut = go.Figure(go.Pie(
        labels=["TV", "Radio", "Newspaper"],
        values=[tv_budget, radio_budget, newspaper_budget],
        hole=0.55,
        marker_colors=["#63b3ed", "#34d399", "#f6ad55"],
        textfont_size=12,
    ))
    fig_donut.update_layout(
        title="Budget Allocation",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Diminishing returns warnings
    st.markdown('<div class="section-title">⚠️ Spending Signals</div>', unsafe_allow_html=True)

    any_warning = False
    if tv_budget > TV_THRESH:
        roi_est = pred_sales / tv_budget if tv_budget > 0 else 0
        st.markdown(f'<div class="badge-warn">📺 TV at ${tv_budget:.0f}K exceeds 75th pct (${TV_THRESH:.0f}K) — likely diminishing returns. ROI ≈ {roi_est:.2f}</div>', unsafe_allow_html=True)
        any_warning = True
    else:
        st.markdown(f'<div class="badge-good">📺 TV spend is within high-ROI zone (≤${TV_THRESH:.0f}K)</div>', unsafe_allow_html=True)

    if radio_budget > RADIO_THRESH:
        st.markdown(f'<div class="badge-warn">📻 Radio at ${radio_budget:.0f}K exceeds 75th pct (${RADIO_THRESH:.0f}K) — diminishing returns</div>', unsafe_allow_html=True)
        any_warning = True
    else:
        st.markdown(f'<div class="badge-good">📻 Radio spend is within high-ROI zone (≤${RADIO_THRESH:.0f}K)</div>', unsafe_allow_html=True)

    if newspaper_budget > NEWS_THRESH:
        st.markdown(f'<div class="badge-warn">📰 Newspaper at ${newspaper_budget:.0f}K exceeds 75th pct (${NEWS_THRESH:.0f}K) — weak channel</div>', unsafe_allow_html=True)
        any_warning = True
    else:
        st.markdown(f'<div class="badge-info">📰 Newspaper shows weak correlation with sales</div>', unsafe_allow_html=True)

    if not any_warning:
        eff = (pred_sales / total_budget * 1000) if total_budget > 0 else 0
        st.markdown(f'<div class="badge-good">✅ Budget mix looks efficient — {eff:.2f} sales units per $1K spent</div>', unsafe_allow_html=True)

with col_right:
    # ── Chart 1: Diminishing returns curves ──────────────────────────────────
    st.markdown('<div class="section-title">📉 Diminishing Returns Curves</div>', unsafe_allow_html=True)

    tv_range    = np.linspace(0, 300, 200)
    radio_range = np.linspace(0, 50, 200)
    news_range  = np.linspace(0, 115, 200)

    med_r = df["Radio"].median()
    med_n = df["Newspaper"].median()
    med_t = df["TV"].median()

    pred_tv    = model.predict(poly.transform(np.column_stack([tv_range,   np.full(200, med_r), np.full(200, med_n)])))
    pred_radio = model.predict(poly.transform(np.column_stack([np.full(200, med_t), radio_range, np.full(200, med_n)])))
    pred_news  = model.predict(poly.transform(np.column_stack([np.full(200, med_t), np.full(200, med_r), news_range])))

    fig_dr = make_subplots(
        rows=1, cols=3,
        subplot_titles=["TV Budget", "Radio Budget", "Newspaper Budget"],
    )

    def add_curve(fig, x_range, y_pred, current, threshold, color, col):
        fig.add_trace(go.Scatter(
            x=x_range, y=y_pred, mode="lines",
            line=dict(color=color, width=3),
            name=["TV","Radio","Newspaper"][col-1],
            showlegend=False,
        ), row=1, col=col)
        # Current budget marker
        idx = np.argmin(np.abs(x_range - current))
        fig.add_trace(go.Scatter(
            x=[current], y=[y_pred[idx]], mode="markers",
            marker=dict(color="white", size=12, symbol="star", line=dict(color=color, width=2)),
            showlegend=False,
        ), row=1, col=col)
        # Threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="#f6ad55",
                      line_width=1.5, row=1, col=col)

    add_curve(fig_dr, tv_range,    pred_tv,    tv_budget,        TV_THRESH,    "#63b3ed", 1)
    add_curve(fig_dr, radio_range, pred_radio, radio_budget,     RADIO_THRESH, "#34d399", 2)
    add_curve(fig_dr, news_range,  pred_news,  newspaper_budget, NEWS_THRESH,  "#f6ad55", 3)

    fig_dr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig_dr.update_xaxes(gridcolor="rgba(255,255,255,0.07)", title_text="Budget ($K)")
    fig_dr.update_yaxes(gridcolor="rgba(255,255,255,0.07)", title_text="Predicted Sales")
    st.plotly_chart(fig_dr, use_container_width=True)

    # ── Chart 2: Actual vs Predicted ─────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(color="#63b3ed", size=8, opacity=0.8, line=dict(color="#1e3a5f", width=1)),
        name="Predictions",
    ))
    lim = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    fig_avp.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="#34d399", dash="dash", width=2),
        name="Perfect fit",
    ))
    fig_avp.update_layout(
        xaxis_title="Actual Sales", yaxis_title="Predicted Sales",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white",
        height=290,
        margin=dict(t=10, b=20, l=20, r=20),
        legend=dict(orientation="h", y=1.05),
    )
    fig_avp.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig_avp.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig_avp, use_container_width=True)

# ── Optional panels ───────────────────────────────────────────────────────────
if show_residuals:
    st.markdown('<div class="section-title">📐 Residual Analysis</div>', unsafe_allow_html=True)
    residuals = y_test - y_pred
    fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs Fitted", "Residual Distribution"])
    fig_res.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color="#f6ad55", size=7, opacity=0.7),
        showlegend=False,
    ), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="#ef4444", row=1, col=1)
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=20, marker_color="#63b3ed",
        opacity=0.8, showlegend=False,
    ), row=1, col=2)
    fig_res.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white", height=300, margin=dict(t=40, b=20, l=20, r=20),
    )
    fig_res.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig_res.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig_res, use_container_width=True)

if show_cv:
    st.markdown('<div class="section-title">🔁 Cross-Validation (5-Fold R² Scores)</div>', unsafe_allow_html=True)
    X_full  = poly.transform(df[["TV", "Radio", "Newspaper"]].values)
    cv_scores = cross_val_score(model, X_full, df["Sales"].values, cv=5, scoring="r2")
    fig_cv = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],
        y=cv_scores,
        marker_color=["#34d399" if s > 0.9 else "#63b3ed" for s in cv_scores],
        text=[f"{s:.4f}" for s in cv_scores], textposition="outside",
    ))
    fig_cv.update_layout(
        yaxis=dict(range=[0.8, 1.0], gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white", height=300, margin=dict(t=10, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

if show_data:
    st.markdown('<div class="section-title">📋 Raw Dataset (200 Observations)</div>', unsafe_allow_html=True)
    st.dataframe(
        df.style.background_gradient(cmap="Blues", subset=["TV", "Radio", "Newspaper"])
              .background_gradient(cmap="Greens", subset=["Sales"]),
        height=350, use_container_width=True,
    )

# ── Feature importance (coefficient magnitudes) ───────────────────────────────
st.markdown('<div class="section-title">🔬 Polynomial Feature Importances (|Coefficient|)</div>', unsafe_allow_html=True)
feat_names = poly.get_feature_names_out(["TV", "Radio", "Newspaper"])
coefs      = np.abs(model.coef_)
top_idx    = np.argsort(coefs)[-12:][::-1]

fig_imp = go.Figure(go.Bar(
    x=coefs[top_idx],
    y=[feat_names[i] for i in top_idx],
    orientation="h",
    marker=dict(
        color=coefs[top_idx],
        colorscale=[[0, "#1e3a5f"], [0.5, "#63b3ed"], [1, "#34d399"]],
    ),
))
fig_imp.update_layout(
    xaxis_title="|Coefficient|",
    yaxis=dict(autorange="reversed"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font_color="white",
    height=340,
    margin=dict(t=10, b=20, l=130, r=20),
)
fig_imp.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
st.plotly_chart(fig_imp, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='opacity:0.4;font-size:0.8rem;'>Polynomial Regression Degree-2 · Advertising Dataset · Built with Streamlit & scikit-learn</center>",
    unsafe_allow_html=True
)

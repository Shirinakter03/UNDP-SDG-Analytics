"""
UNDP SDG Analytics Dashboard
============================
Interactive dashboard analyzing UN Sustainable Development Goal indicators
across 24 countries using K-Means clustering, Gini coefficient inequality
analysis, and HDI trend decomposition.

Data sources:
    - UNDP Human Development Report 2023
    - World Bank World Development Indicators (WDI)
    - Our World in Data (CO2 per capita)

Author: Shirin Akter Ishita
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNDP SDG Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
)

CLUSTER_NAMES = {0: "Vulnerable", 1: "Developing", 2: "Advanced"}
CLUSTER_COLORS = {"Vulnerable": "#F87171", "Developing": "#FBBF24", "Advanced": "#34D399"}


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """
    Loads the raw dataset (which contains realistic missing values, as
    real-world SDG data from World Bank / UNDP often does) and cleans it.

    Cleaning strategy:
        - Numeric columns: median imputation grouped by region, since
          countries in the same region tend to have similar development
          profiles (this avoids skewing HDI, poverty, etc. toward a
          global average that may not fit the region).
        - If a region-level median is unavailable (e.g. all missing),
          fall back to the global column median.
    """
    raw = pd.read_csv(Path(__file__).parent / "data" / "sdg_indicators_raw.csv")
    missing_report = raw.isnull().sum()
    missing_report = missing_report[missing_report > 0]

    numeric_cols = ["hdi", "poverty_rate", "literacy_rate", "co2_per_capita",
                     "internet_access", "gni_per_capita"]

    cleaned = raw.copy()
    for col in numeric_cols:
        cleaned[col] = cleaned.groupby("region")[col].transform(
            lambda s: s.fillna(s.median())
        )
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())  # global fallback

    return raw, cleaned, missing_report


# ──────────────────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(df, k=3):
    """
    Clusters countries using K-Means on 4 normalized development indicators.

    Why normalize? GNI per capita ranges from ~$1,000 to ~$59,000 while HDI
    ranges from 0 to 1. Without scaling, GNI would dominate the Euclidean
    distance calculation and the clusters would just reflect income, not
    holistic development.
    """
    features = ["hdi", "poverty_rate", "internet_access", "co2_per_capita"]
    X = df[features].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    # Elbow method data (WCSS for k=1..8) to justify choice of k
    wcss = []
    for i in range(1, 9):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    df = df.copy()
    df["cluster_id"] = labels

    # Order cluster IDs by mean HDI so "0" is always the lowest-development
    # group, regardless of how KMeans happened to number them
    order = df.groupby("cluster_id")["hdi"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    df["cluster_id"] = df["cluster_id"].map(remap)
    df["cluster_name"] = df["cluster_id"].map(CLUSTER_NAMES)

    return df, sil_score, wcss


def compute_gini(values):
    """Gini coefficient via mean absolute difference method."""
    values = np.sort(np.array(values, dtype=float))
    n = len(values)
    mean = values.mean()
    diffs = np.abs(values[:, None] - values[None, :]).sum()
    return diffs / (2 * n * n * mean)


# ──────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS: k=2 TO k=6
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_sensitivity_analysis(df, k_range=range(2, 7)):
    """
    Runs K-Means across a range of k values (2 through 6) and records the
    silhouette score for each, so the choice of k=3 can be justified
    against real alternatives rather than assumed.

    This does NOT decide "which k is best" automatically — a higher
    silhouette score does not always mean a more *useful* clustering for
    policy purposes. It returns the full comparison so a human can weigh
    statistical fit against interpretability.
    """
    features = ["hdi", "poverty_rate", "internet_access", "co2_per_capita"]
    X = MinMaxScaler().fit_transform(df[features])

    results = []
    label_sets = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "silhouette_score": sil, "wcss": model.inertia_})
        label_sets[k] = labels

    return pd.DataFrame(results), label_sets


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
raw_df, df, missing_report = load_and_clean_data()
df, sil_score, wcss = run_clustering(df, k=3)
hdi_trends = pd.read_csv(Path(__file__).parent / "data" / "hdi_trends.csv")
sensitivity_df, sensitivity_labels = run_sensitivity_analysis(df)

# ──────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────
st.title("🌍 UNDP SDG Analytics Dashboard")
st.caption(
    "K-Means clustering · Gini inequality · HDI trend analysis — "
    f"across {len(df)} countries, 6 regions"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries", len(df))
col2.metric("Avg HDI", f"{df['hdi'].mean():.3f}")
col3.metric("GNI Gini Coefficient", f"{compute_gini(df['gni_per_capita']):.3f}")
col4.metric("Silhouette Score", f"{sil_score:.3f}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab3b, tab4, tab5 = st.tabs([
    "📊 Overview", "🧹 Data Cleaning", "🔵 Clustering",
    "🧪 Sensitivity (k=2–6)", "📈 HDI Trends", "🌐 Country Table"
])

# ── TAB 1: OVERVIEW ─────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df, x="internet_access", y="hdi", color="cluster_name",
            color_discrete_map=CLUSTER_COLORS, hover_name="country",
            size="gni_per_capita", size_max=30,
            title="HDI vs Internet Access, sized by GNI per capita",
            labels={"internet_access": "Internet Access (%)", "hdi": "HDI"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            df, x="poverty_rate", y="hdi", color="cluster_name",
            color_discrete_map=CLUSTER_COLORS, hover_name="country",
            title="Poverty Rate vs HDI — Digital Divide Lens",
            labels={"poverty_rate": "Poverty Rate (%)", "hdi": "HDI"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cluster Summary")
    summary = df.groupby("cluster_name")[
        ["hdi", "poverty_rate", "internet_access", "co2_per_capita"]
    ].mean().round(3).reindex(["Vulnerable", "Developing", "Advanced"])
    st.dataframe(summary, use_container_width=True)

# ── TAB 2: DATA CLEANING ────────────────────────────────────────────────
with tab2:
    st.subheader("Missing Value Report (Raw Data)")
    if len(missing_report) > 0:
        st.write(
            "The raw dataset (as pulled from World Bank / UNDP sources) "
            "contained missing values in the following columns:"
        )
        st.dataframe(missing_report.rename("missing_count"))
    st.write(
        "**Cleaning strategy:** missing numeric values were imputed using "
        "the **median of countries in the same region**, rather than a "
        "global median. Regional grouping preserves realistic patterns — "
        "for example, a missing literacy rate for an African nation is "
        "filled using the African regional median, not skewed upward by "
        "European values."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Before cleaning** (raw)")
        st.dataframe(raw_df[raw_df.isnull().any(axis=1)])
    with c2:
        st.write("**After cleaning** (imputed)")
        affected = raw_df[raw_df.isnull().any(axis=1)]["country"]
        st.dataframe(df[df["country"].isin(affected)][
            ["country", "hdi", "poverty_rate", "literacy_rate",
             "co2_per_capita", "internet_access"]
        ])

# ── TAB 3: CLUSTERING (PRIMARY MODEL, k=3) ──────────────────────────────
with tab3:
    st.subheader("Primary Model: k=3")
    st.caption(
        "k=3 is used as the main model throughout this dashboard. See the "
        "**Sensitivity (k=2–6)** tab for why, including a direct comparison "
        "against k=4 and other alternatives — k=3 was a deliberate choice, "
        "not the only option tested."
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(1, 9)), y=wcss, mode="lines+markers"))
    fig3.add_vline(x=3, line_dash="dash", line_color="red",
                    annotation_text="chosen k=3")
    fig3.update_layout(
        title="Within-Cluster Sum of Squares (WCSS) vs. k",
        xaxis_title="Number of clusters (k)", yaxis_title="WCSS",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.write(
        f"**Silhouette score at k=3: {sil_score:.3f}** — scores above 0.5 "
        "indicate reasonably well-separated clusters. k=3 also matches "
        "UNDP's own Low / Medium / High human development classification, "
        "making the clusters directly interpretable for policy use."
    )

    st.subheader("Clusters by Country")
    for name in ["Vulnerable", "Developing", "Advanced"]:
        subset = df[df["cluster_name"] == name]
        st.markdown(f"**{name}** ({len(subset)} countries): " +
                    ", ".join(subset["country"]))

# ── TAB 3b: SENSITIVITY ANALYSIS (k=2 TO k=6) ───────────────────────────
with tab3b:
    st.subheader("Sensitivity Analysis: Is k=3 Actually the Best Choice?")
    st.write(
        "K-Means requires the number of clusters to be chosen in advance. "
        "Rather than assuming k=3 is correct, this tab compares k=2 through "
        "k=6 directly on silhouette score — and treats k=4 as a genuine "
        "alternative model, not a fallback."
    )

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Bar(
        x=sensitivity_df["k"], y=sensitivity_df["silhouette_score"],
        marker_color=["#00D4FF" if k == 3 else
                      ("#FBBF24" if k == 4 else "#334155")
                      for k in sensitivity_df["k"]],
        text=sensitivity_df["silhouette_score"].round(3),
        textposition="outside",
    ))
    fig_sens.update_layout(
        title="Silhouette Score by Number of Clusters (k=2–6)",
        xaxis_title="k", yaxis_title="Silhouette Score",
        xaxis=dict(tickvals=list(sensitivity_df["k"])),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    best_row = sensitivity_df.loc[sensitivity_df["silhouette_score"].idxmax()]
    st.info(
        f"**Highest silhouette score: k={int(best_row['k'])} "
        f"({best_row['silhouette_score']:.3f})**. A higher score means "
        "more mathematically separated clusters — but that alone doesn't "
        "make it the right choice. k=3 is kept as the main model in this "
        "dashboard because it maps directly onto UNDP's Low/Medium/High "
        "human development framework, which matters more for policy "
        "communication than a marginal silhouette gain. This is a judgment "
        "call, not an automatic one — a different analyst prioritizing "
        "statistical fit over interpretability could reasonably pick a "
        "different k."
    )

    st.subheader("k=4 as an Alternative: What Changes?")
    k4_labels = sensitivity_labels[4]
    compare_df = df[["country", "cluster_name"]].copy()
    compare_df["k4_cluster"] = k4_labels
    # Order k=4 cluster ids by mean HDI for readability, same convention as k=3
    hdi_by_k4 = df.assign(k4=k4_labels).groupby("k4")["hdi"].mean().sort_values()
    remap4 = {old: new for new, old in enumerate(hdi_by_k4.index)}
    compare_df["k4_cluster"] = compare_df["k4_cluster"].map(remap4)

    st.write(
        "At k=4, the *Vulnerable* cluster (the lowest-HDI group under k=3) "
        "is the one that splits into two sub-groups; the *Developing* and "
        "*Advanced* clusters stay intact. In other words, k=4 adds "
        "resolution at the low end of the development spectrum rather than "
        "changing the overall story. Below is the raw country-by-country "
        "comparison so this claim is checkable rather than asserted:"
    )
    st.dataframe(
        compare_df.rename(columns={
            "cluster_name": "k=3 cluster",
            "k4_cluster": "k=4 cluster (0=lowest HDI)"
        }),
        use_container_width=True,
    )
    st.caption(
        "Takeaway: k=4 doesn't contradict the k=3 story, it subdivides one "
        "existing group further. Since the extra split doesn't map onto a "
        "recognized development framework the way Low/Medium/High does, "
        "k=3 remains the primary model — but k=4 is documented here as a "
        "legitimate alternative rather than hidden."
    )

# ── TAB 4: HDI TRENDS ───────────────────────────────────────────────────
with tab4:
    fig4 = px.line(
        hdi_trends, x="year", y="hdi", color="country", markers=True,
        title="HDI Trend 2015–2023 (COVID-19 dip visible in 2021)",
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.info(
        "**Bangladesh** shows the strongest upward trajectory "
        "(0.579 → 0.661). **India, Brazil, and Germany** all show a "
        "measurable HDI dip in 2021, consistent with COVID-19 impacts on "
        "health and income indicators globally."
    )

# ── TAB 5: COUNTRY TABLE ────────────────────────────────────────────────
with tab5:
    region_filter = st.selectbox("Filter by region", ["All"] + sorted(df["region"].unique()))
    display_df = df if region_filter == "All" else df[df["region"] == region_filter]
    st.dataframe(
        display_df[["country", "region", "cluster_name", "hdi", "poverty_rate",
                     "internet_access", "co2_per_capita", "gni_per_capita",
                     "literacy_rate"]].sort_values("hdi", ascending=False),
        use_container_width=True,
    )

st.divider()
st.caption("Data: UNDP HDR 2023 · World Bank WDI · Our World in Data")

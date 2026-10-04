"""
charts.py
---------
All Plotly visualisation functions used by the Streamlit dashboard.
Each function is self-contained and returns a go.Figure object.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BG       = "#0A0F1E"
SURFACE  = "#111827"
CARD     = "#1A2235"
BORDER   = "#1E3A5F"
TEXT     = "#E2E8F0"
MUTED    = "#64748B"

CLUSTER_COLORS = {
    "Vulnerable": "#F87171",
    "Developing": "#FBBF24",
    "Advanced":   "#34D399",
}

_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def scatter_plot(df: pd.DataFrame, x: str, y: str,
                 x_label: str, y_label: str) -> go.Figure:
    """
    Scatter plot coloured by cluster.
    Hover shows country, region, and both axis values.
    """
    fig = go.Figure()
    for cluster, group in df.groupby("cluster"):
        fig.add_trace(go.Scatter(
            x=group[x],
            y=group[y],
            mode="markers",
            name=cluster,
            marker=dict(
                color=CLUSTER_COLORS.get(cluster, "#888"),
                size=10,
                opacity=0.85,
                line=dict(width=1, color="#ffffff22"),
            ),
            customdata=np.stack([group["country"],
                                 group["region"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Region: %{customdata[1]}<br>"
                f"{x_label}: %{{x}}<br>"
                f"{y_label}: %{{y}}<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_LAYOUT,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def gni_bar(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: GNI per capita by country, coloured by cluster.
    Sorted ascending for easy reading.
    """
    df_sorted = df.sort_values("gni_per_capita")
    colors = df_sorted["cluster"].map(CLUSTER_COLORS)

    fig = go.Figure(go.Bar(
        x=df_sorted["gni_per_capita"],
        y=df_sorted["country"],
        orientation="h",
        marker_color=colors,
        customdata=np.stack([df_sorted["cluster"],
                             df_sorted["region"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "GNI/capita: $%{x:,.0f}<br>"
            "Cluster: %{customdata[0]}<br>"
            "Region: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_LAYOUT,
        xaxis_title="GNI per Capita (PPP, int'l $)",
        height=600,
    )
    fig.update_xaxes(gridcolor=BORDER)
    return fig


def hdi_trend(df_trends: pd.DataFrame,
              countries: list) -> go.Figure:
    """
    Multi-line HDI trend chart (2015–2023).
    Annotates the 2021 COVID-19 dip with a vertical band.
    """
    fig = go.Figure()

    # COVID shading
    fig.add_vrect(x0=2020, x1=2022,
                  fillcolor="#F87171", opacity=0.08,
                  line_width=0, annotation_text="COVID-19",
                  annotation_position="top left",
                  annotation_font_color="#F87171")

    palette = px.colors.qualitative.Set2
    for i, country in enumerate(countries):
        data = df_trends[df_trends["country"] == country].sort_values("year")
        fig.add_trace(go.Scatter(
            x=data["year"],
            y=data["hdi"],
            mode="lines+markers",
            name=country,
            line=dict(color=palette[i % len(palette)], width=2.5),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{country}</b><br>"
                "Year: %{x}<br>"
                "HDI: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT,
        xaxis_title="Year",
        yaxis_title="Human Development Index (HDI)",
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
        xaxis=dict(tickvals=[2015, 2017, 2019, 2021, 2023]),
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def elbow_plot(wcss_dict: dict) -> go.Figure:
    """
    Elbow Method plot — WCSS vs number of clusters k.
    Used to validate the choice of k=3.
    """
    ks   = list(wcss_dict.keys())
    vals = list(wcss_dict.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=vals,
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2.5),
        marker=dict(size=9, color="#00D4FF"),
        hovertemplate="k=%{x}<br>WCSS=%{y:.4f}<extra></extra>",
    ))
    # Highlight chosen k
    chosen_idx = ks.index(3) if 3 in ks else 0
    fig.add_trace(go.Scatter(
        x=[ks[chosen_idx]], y=[vals[chosen_idx]],
        mode="markers",
        marker=dict(size=14, color="#34D399",
                    symbol="star", line=dict(width=1, color="white")),
        name="Chosen k=3",
        hovertemplate="Chosen k=3<br>WCSS=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Within-Cluster Sum of Squares",
        showlegend=True,
        xaxis=dict(tickvals=ks),
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def radar_chart(summary: pd.DataFrame) -> go.Figure:
    """
    Radar (spider) chart comparing normalised mean indicators
    across the three clusters.
    """
    categories = ["HDI", "Poverty\n(inv)", "Internet", "Literacy", "GNI\n(norm)"]
    fig = go.Figure()

    for _, row in summary.iterrows():
        cluster = row["cluster"]
        # Invert poverty so higher = better on all axes
        poverty_inv = 1 - (row["poverty_rate"] / 100)
        gni_norm    = row["gni_per_capita"] / 60000

        values = [
            row["hdi"],
            poverty_inv,
            row["internet_access"] / 100,
            row["literacy_rate"]   / 100,
            gni_norm,
        ]
        values += [values[0]]  # close the polygon
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values, theta=cats,
            fill="toself",
            name=cluster,
            line_color=CLUSTER_COLORS.get(cluster, "#888"),
            fillcolor=CLUSTER_COLORS.get(cluster, "#888"),
            opacity=0.3,
        ))

    fig.update_layout(
        **_LAYOUT,
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=BORDER, color=MUTED),
            angularaxis=dict(gridcolor=BORDER, color=TEXT),
        ),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
    )
    return fig

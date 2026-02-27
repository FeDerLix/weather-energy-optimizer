"""
Chart utilities for the Weather & Energy Optimizer.

Provides reusable Plotly chart builders with a consistent visual style.
All charts use a dark theme to match the Streamlit app design.

Chart types available:
    - Line charts (time series)
    - Bar charts (comparisons)
    - Scatter plots (correlations)
    - Heatmaps (correlation matrices)
    - Gauge charts (KPI-style indicators)
    - Prediction vs actual plots (model evaluation)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Shared theme configuration
# ---------------------------------------------------------------------------

# Consistent color palette used across all charts
COLORS = {
    "primary": "#4CAF50",       # Green – primary accent
    "secondary": "#2196F3",     # Blue – secondary data
    "warning": "#FF9800",       # Orange – warnings/attention
    "danger": "#F44336",        # Red – negative indicators
    "info": "#00BCD4",          # Cyan – informational
    "energy": "#FFD700",        # Gold – energy consumption
    "temperature": "#FF6B6B",   # Coral – temperature
    "background": "#0e1117",    # Dark background
    "paper": "#1a1f2e",         # Card background
    "text": "#fafafa",          # Light text
}

# Color sequence for multi-series charts
COLOR_SEQUENCE = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#00BCD4", "#FFD700"]

# Standard layout to apply to all charts for consistency
BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
    ),
)


def _apply_base_layout(fig):
    """Apply the shared theme layout to a Plotly figure."""
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Line charts
# ---------------------------------------------------------------------------

def line_chart(df, x, y, title="", color=None, labels=None, y_label=""):
    """
    Create a line chart for time-series data.

    Args:
        df (pd.DataFrame): Data to plot
        x (str): Column name for x-axis (typically 'date')
        y (str or list): Column name(s) for y-axis
        title (str): Chart title
        color (str, optional): Column to use for color grouping
        labels (dict, optional): Axis label overrides
        y_label (str): Y-axis label

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.line(
        df, x=x, y=y, title=title, color=color, labels=labels,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(line=dict(width=2))
    if y_label:
        fig.update_yaxes(title_text=y_label)
    return _apply_base_layout(fig)


def dual_axis_line_chart(df, x, y1, y2, title="", y1_label="", y2_label="",
                         y1_color=None, y2_color=None):
    """
    Create a line chart with two y-axes for comparing different scales.
    Useful for overlaying temperature and energy consumption.

    Args:
        df (pd.DataFrame): Data to plot
        x (str): Column for x-axis
        y1, y2 (str): Column names for left and right y-axes
        title (str): Chart title
        y1_label, y2_label (str): Axis labels
        y1_color, y2_color (str): Line colors

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Left y-axis trace
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y1],
        name=y1_label or y1,
        line=dict(color=y1_color or COLORS["temperature"], width=2),
        yaxis="y1",
    ))

    # Right y-axis trace
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y2],
        name=y2_label or y2,
        line=dict(color=y2_color or COLORS["energy"], width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(title=y1_label, side="left", showgrid=False),
        yaxis2=dict(title=y2_label, side="right", overlaying="y", showgrid=False),
    )
    return _apply_base_layout(fig)


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def bar_chart(df, x, y, title="", color=None, labels=None, horizontal=False):
    """
    Create a bar chart for categorical comparisons.

    Args:
        df (pd.DataFrame): Data to plot
        x, y (str): Column names for axes
        title (str): Chart title
        color (str, optional): Column for color grouping
        labels (dict, optional): Axis label overrides
        horizontal (bool): If True, create horizontal bars

    Returns:
        plotly.graph_objects.Figure
    """
    if horizontal:
        fig = px.bar(df, x=y, y=x, title=title, color=color, labels=labels,
                     orientation="h", color_discrete_sequence=COLOR_SEQUENCE)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color, labels=labels,
                     color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_traces(marker_line_width=0)
    return _apply_base_layout(fig)


def feature_importance_chart(importance_df, title="Feature Importance"):
    """
    Create a horizontal bar chart for ML model feature importances.

    Args:
        importance_df (pd.DataFrame): With columns ['feature', 'importance']
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        color="importance",
        color_continuous_scale=["#1a1f2e", "#4CAF50"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return _apply_base_layout(fig)


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def scatter_chart(df, x, y, title="", color=None, labels=None, trendline=None):
    """
    Create a scatter plot for exploring correlations.

    Args:
        df (pd.DataFrame): Data to plot
        x, y (str): Column names for axes
        title (str): Chart title
        color (str, optional): Column for color grouping
        labels (dict, optional): Axis label overrides
        trendline (str, optional): 'ols' for ordinary least squares trendline

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.scatter(
        df, x=x, y=y, title=title, color=color, labels=labels,
        trendline=trendline, color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.6,
    )
    return _apply_base_layout(fig)


def prediction_vs_actual_chart(y_true, y_pred, title="Predicted vs Actual Energy (kWh)"):
    """
    Create a scatter plot comparing model predictions to actual values.
    Includes a diagonal reference line (perfect prediction).

    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Scatter points: predicted vs actual
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=COLORS["primary"], opacity=0.6, size=8),
        name="Predictions",
    ))

    # Perfect-prediction diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["warning"], dash="dash", width=2),
        name="Perfect prediction",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Actual Energy (kWh)",
        yaxis_title="Predicted Energy (kWh)",
    )
    return _apply_base_layout(fig)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def correlation_heatmap(df, columns=None, title="Feature Correlation Matrix"):
    """
    Create a correlation heatmap for numerical columns.

    Args:
        df (pd.DataFrame): Data with numerical columns
        columns (list, optional): Specific columns to include
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure
    """
    if columns:
        corr = df[columns].corr()
    else:
        corr = df.select_dtypes(include=[np.number]).corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, "#F44336"], [0.5, "#1a1f2e"], [1, "#4CAF50"]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(title=title, height=500)
    return _apply_base_layout(fig)


# ---------------------------------------------------------------------------
# Gauge / KPI charts
# ---------------------------------------------------------------------------

def gauge_chart(value, title="", min_val=0, max_val=100, suffix=""):
    """
    Create a gauge chart as a visual KPI indicator.

    Args:
        value (float): Current value to display
        title (str): Gauge title
        min_val, max_val (float): Range of the gauge
        suffix (str): Unit suffix (e.g., '%', 'kWh')

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": COLORS["primary"]},
            "bgcolor": COLORS["paper"],
            "steps": [
                {"range": [min_val, max_val * 0.33], "color": "#1b5e20"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#f57f17"},
                {"range": [max_val * 0.66, max_val], "color": "#c62828"},
            ],
        },
    ))
    fig.update_layout(height=250)
    return _apply_base_layout(fig)

"""
Chart Factory Module.

Provides reusable Plotly chart generation adhering to the Rin Mitra Theme.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class ChartFactory:
    """Generates standardized Plotly charts."""

    def __init__(self):
        # Rin Mitra Theme colors and layout settings
        self.primary_color = "#1E88E5"
        self.secondary_color = "#43A047"
        self.danger_color = "#E53935"
        self.warning_color = "#FDD835"
        self.bg_color = "rgba(0,0,0,0)"
        self.font_family = "Inter, Roboto, sans-serif"
        self.color_sequence = [self.primary_color, self.secondary_color, "#8E24AA", "#FFB300", "#00ACC1"]
        
        self.layout_defaults = dict(
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(family=self.font_family, color="#333333"),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="closest"
        )

    def _apply_theme(self, fig: go.Figure, title: str) -> go.Figure:
        """Apply the global theme to a Plotly figure."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family=self.font_family)),
            **self.layout_defaults
        )
        return fig

    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, title: str, 
                         orientation: str = 'v', color: Optional[str] = None) -> go.Figure:
        """Create a standard bar chart."""
        try:
            fig = px.bar(df, x=x, y=y, orientation=orientation, color=color,
                         color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create bar chart '{title}': {e}")
            return go.Figure()

    def create_line_chart(self, df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
        """Create a standard line chart."""
        try:
            fig = px.line(df, x=x, y=y, color=color, color_discrete_sequence=self.color_sequence)
            fig.update_traces(mode='lines+markers')
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create line chart '{title}': {e}")
            return go.Figure()

    def create_area_chart(self, df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
        """Create a standard area chart."""
        try:
            fig = px.area(df, x=x, y=y, color=color, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create area chart '{title}': {e}")
            return go.Figure()

    def create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, title: str, 
                            color: Optional[str] = None, size: Optional[str] = None) -> go.Figure:
        """Create a standard scatter plot."""
        try:
            fig = px.scatter(df, x=x, y=y, color=color, size=size, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create scatter plot '{title}': {e}")
            return go.Figure()

    def create_histogram(self, df: pd.DataFrame, x: str, title: str, color: Optional[str] = None, nbins: int = 20) -> go.Figure:
        """Create a standard histogram."""
        try:
            fig = px.histogram(df, x=x, color=color, nbins=nbins, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create histogram '{title}': {e}")
            return go.Figure()

    def create_pie_chart(self, df: pd.DataFrame, names: str, values: str, title: str, hole: float = 0.0) -> go.Figure:
        """Create a standard pie or donut chart."""
        try:
            fig = px.pie(df, names=names, values=values, hole=hole, color_discrete_sequence=self.color_sequence)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create pie chart '{title}': {e}")
            return go.Figure()

    def create_box_plot(self, df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
        """Create a standard box plot."""
        try:
            fig = px.box(df, x=x, y=y, color=color, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create box plot '{title}': {e}")
            return go.Figure()

    def create_violin_plot(self, df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
        """Create a standard violin plot."""
        try:
            fig = px.violin(df, x=x, y=y, color=color, box=True, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create violin plot '{title}': {e}")
            return go.Figure()

    def create_treemap(self, df: pd.DataFrame, path: List[str], values: str, title: str) -> go.Figure:
        """Create a standard treemap."""
        try:
            fig = px.treemap(df, path=path, values=values, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create treemap '{title}': {e}")
            return go.Figure()

    def create_sunburst(self, df: pd.DataFrame, path: List[str], values: str, title: str) -> go.Figure:
        """Create a standard sunburst chart."""
        try:
            fig = px.sunburst(df, path=path, values=values, color_discrete_sequence=self.color_sequence)
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create sunburst '{title}': {e}")
            return go.Figure()

    def create_heatmap(self, z: List[List[float]], x: List[str], y: List[str], title: str) -> go.Figure:
        """Create a standard heatmap."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=z, x=x, y=y, colorscale='Blues'
            ))
            return self._apply_theme(fig, title)
        except Exception as e:
            logger.error(f"Failed to create heatmap '{title}': {e}")
            return go.Figure()

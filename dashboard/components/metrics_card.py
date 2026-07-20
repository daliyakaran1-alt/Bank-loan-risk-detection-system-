import streamlit as st
from src.analytics.analytics_models import KPI

def render_metric_card(kpi: KPI):
    """Render a KPI dataclass as a premium UI card."""
    color = "var(--primary)" if kpi.is_good is None else ("#43A047" if kpi.is_good else "#E53935")
    
    st.markdown(f"""
    <div class="finsight-card">
        <div class="finsight-metric-label">{kpi.name}</div>
        <div class="finsight-metric-value" style="color: {color};">{kpi.formatted_value}</div>
    </div>
    """, unsafe_allow_html=True)

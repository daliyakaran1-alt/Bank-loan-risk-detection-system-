import streamlit as st
from src.analytics.analytics_models import ChartData

def render_chart_card(chart_data: ChartData, use_container_width: bool = True):
    """Render a Plotly figure inside a premium UI card."""
    st.markdown(f"""
    <div class="finsight-card" style="padding-bottom: 0;">
        <h4 style="margin-top: 0; margin-bottom: 1rem; color: var(--text-dark);">{chart_data.title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if chart_data.figure:
        st.plotly_chart(chart_data.figure, use_container_width=use_container_width)
    else:
        st.warning("Chart data unavailable.")

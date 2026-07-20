import streamlit as st
from src.analytics.analytics_models import BusinessInsight

def render_insight_card(insight: BusinessInsight):
    """Render a BusinessInsight with semantic styling."""
    
    # Map priority/type to colors/icons
    bg_color = "var(--bg-main)"
    border_color = "var(--text-muted)"
    icon = "💡"
    
    if insight.insight_type == "warning" or insight.priority == "high":
        bg_color = "#FFF4F4"
        border_color = "#E53935"
        icon = "⚠️"
    elif insight.insight_type == "recommendation":
        bg_color = "#F4FCF5"
        border_color = "#43A047"
        icon = "📈"
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color}; 
        border-left: 4px solid {border_color}; 
        padding: 1rem; 
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 0.75rem; font-size: 1.2rem;">{icon}</div>
            <div style="color: var(--text-dark); font-size: 0.95rem; line-height: 1.5;">
                {insight.content}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st

def page_header(title: str, description: str = ""):
    """Render a standard page header."""
    st.markdown(f"""
    <div style="padding-bottom: 1.5rem; border-bottom: 1px solid #E0E5ED; margin-bottom: 2rem;">
        <h1 style="color: var(--primary); margin: 0; font-size: 2.2rem;">{title}</h1>
        <p style="color: var(--text-muted); margin: 0.5rem 0 0 0; font-size: 1.1rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

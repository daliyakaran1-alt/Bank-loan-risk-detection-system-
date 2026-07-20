import streamlit as st
from src.services.auth_service import logout_user, get_current_role

def render_sidebar():
    with st.sidebar:
        import base64
        import os
        logo_path = os.path.join("assets", "images", "logo.png")
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <img src="data:image/png;base64,{logo_b64}" style="height: 70px; width: auto; max-width: 100%; object-fit: contain; margin-bottom: 5px;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: white; margin: 0; font-weight: 800;">FinSight</h2>
                <p style="color: var(--accent); font-size: 0.8rem; margin: 0; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">Enterprise Platform</p>
            </div>
            """, unsafe_allow_html=True)
        
        # User info
        role = get_current_role()
        st.markdown(f"<div style='text-align:center; color: #ccc; font-size: 0.9rem; margin-bottom: 1rem;'>User: {st.session_state.get('username')} <br>Role: <b>{role}</b></div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Navigation Groups
        st.markdown("<h4 style='color: white; margin-bottom: 0.5rem;'>Dashboards</h4>", unsafe_allow_html=True)
        dashboards = {
            "Executive Dashboard": "executive_dashboard",
            "Customer Analytics": "customer_analytics",
            "Loan Analytics": "loan_analytics",
            "Risk Analytics": "risk_analytics",
            "Financial Analytics": "financial_analytics",
            "Demographic Analytics": "demographic_analytics"
        }
        for label, page_id in dashboards.items():
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("<h4 style='color: white; margin-top: 1rem; margin-bottom: 0.5rem;'>Data & Discovery</h4>", unsafe_allow_html=True)
        discovery = {
            "Global Search": "global_search",
            "Prediction History": "history",
            "Exploratory Data Analysis": "eda",
            "Customer Segmentation": "customer_segmentation"
        }
        for label, page_id in discovery.items():
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("<h4 style='color: white; margin-top: 1rem; margin-bottom: 0.5rem;'>Credit Risk</h4>", unsafe_allow_html=True)
        risk = {
            "Risk Scoring Engine": "risk_scoring",
            "Explainable AI (SHAP)": "explainable_ai",
            "Report Management": "reports"
        }
        for label, page_id in risk.items():
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()
                
        if role == "Administrator":
            st.markdown("<h4 style='color: white; margin-top: 1rem; margin-bottom: 0.5rem;'>System</h4>", unsafe_allow_html=True)
            if st.button("Administration", key="nav_admin", use_container_width=True):
                st.session_state.current_page = "administration"
                st.rerun()

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin-top: 2rem;'>", unsafe_allow_html=True)
        
        if st.button("Logout", key="nav_logout", use_container_width=True):
            logout_user()
            st.session_state.current_page = 'landing'
            st.rerun()

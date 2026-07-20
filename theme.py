"""
UI design system and theming engine.

Defines color palettes, typography, and visual consistency rules
for the FinSight platform, ensuring a professional enterprise look.
"""

import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return ""

def apply_theme():
    """
    Applies the custom enterprise theme and CSS overrides to the Streamlit app.
    """
    st.markdown("""
        <style>
        /* FinSight Enterprise Theme */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        :root {
            --primary: #082567;
            --secondary: #FDB913;
            --bg-main: #F8FAFC;
            --bg-card: #FFFFFF;
            --sidebar: #0A192F;
            --accent: #2563EB;
            --text-dark: #1E293B; /* Near Black */
            --text-muted: #64748B;
            --border: #F1F5F9; /* Very Light Grey */
            --danger: #EF4444; /* Professional Red */
            --success: #10B981; /* Professional Green */
            --warning: #F59E0B; /* Amber */
            
            --radius: 12px;
            --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.05);
            --shadow-premium: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif !important;
            color: var(--text-dark);
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 700 !important;
            color: var(--primary) !important;
            letter-spacing: -0.025em;
        }

        /* Hide Streamlit Defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Main Container */
        .stApp {
            background-color: var(--bg-main);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar) !important;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        /* Top Padding override */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 1400px;
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Poppins', sans-serif !important;
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow);
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #0A3080;
            color: white;
            border: none;
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Primary CTA Button utility */
        .btn-premium {
            background: linear-gradient(135deg, var(--primary) 0%, #1a365d 100%);
            color: white !important;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .btn-premium:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-premium);
        }
        
        .btn-secondary {
            background: white;
            color: var(--primary) !important;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        /* Text Input */
        .stTextInput > div > div > input {
            font-family: 'Poppins', sans-serif !important;
            border-radius: 8px;
            border: 1.5px solid var(--border);
            padding: 0.75rem;
            box-shadow: none;
            transition: all 0.2s ease;
            background-color: white;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(8, 37, 103, 0.08);
        }

        /* Custom Cards */
        .finsight-card {
            background-color: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .finsight-card:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-4px);
        }

        /* Glassmorphism */
        .glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.1);
        }

        .finsight-metric-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .finsight-metric-value {
            font-size: 2.2rem;
            color: var(--primary);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        /* Animations */
        @keyframes slideUpFadeIn {
            0% { transform: translateY(30px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        .animate-up {
            animation: slideUpFadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
            100% { transform: translateY(0px); }
        }
        .animate-float {
            animation: float 6s ease-in-out infinite;
        }

        /* Responsive utilities */
        .d-flex { display: flex; }
        .flex-column { flex-direction: column; }
        .justify-content-between { justify-content: space-between; }
        .align-items-center { align-items: center; }
        .gap-3 { gap: 1rem; }
        .gap-4 { gap: 1.5rem; }
        .mt-4 { margin-top: 1.5rem; }
        .mb-4 { margin-bottom: 1.5rem; }
        .text-center { text-align: center; }
        </style>
    """, unsafe_allow_html=True)


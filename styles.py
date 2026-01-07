import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
        /* Global Font & Colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Card Styling */
        div.css-1r6slb0, .stCard {
            background-color: #1e2329;
            border: 1px solid #2d333b;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #4da6ff;
        }
        
        /* Custom Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0e1117;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
                width: auto;
            padding: 0 16px;
            white-space: pre-wrap;
            background-color: #1e2329;
            border-radius: 2px;
            color: #b0b0b0;
            border: 1px solid #2d333b;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563eb;
            color: white;
            border-color: #2563eb;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #2d333b;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

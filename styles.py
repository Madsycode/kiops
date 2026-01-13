import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
        /* Global Font & Colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0e1117;
            color: #e0e0e0;
        }

        /* Code Blocks */
        code {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Card Styling */
        div.stCard {
            background-color: #1e2329;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }

        /* Headers */
        h1, h2, h3 {
            color: #fafafa;
            font-weight: 600;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 4px;
            background-color: #21262d;
            border: 1px solid #30363d;
            color: #8b949e;
            padding: 0 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f6feb;
            color: white;
            border-color: #1f6feb;
        }

        /* Metrics & Status */
        div[data-testid="stStatusWidget"] {
            background-color: #1e2329;
            border: 1px solid #30363d;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
        }

        /* Graphviz Override */
        svg {
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)
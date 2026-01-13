import os
import graphviz
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Local Imports
from models import RichMLAppProfile
from styles import inject_custom_css
from containerize import DockerExecutionEngine
from generative import query_ai_json, query_ai_text

# ==========================================
# 1. SETUP & STATE
# ==========================================

st.set_page_config(
    layout="wide",
    page_icon="📡",
    page_title="MLOps Platform",
    initial_sidebar_state="expanded"
)

# Apply CSS
inject_custom_css()

# Initialize Session State
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'docker' not in st.session_state:
    st.session_state.docker = DockerExecutionEngine()

# ==========================================
# 2. SIDEBAR (CONTROLS)
# ==========================================
with st.sidebar:
    st.markdown(
        """<div style="display: flex; justify-content: center;">
            <img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="120"/>
            <h1>MLOps Platform</h1>
            <!--<h4>v2.2.0 · Modular Architecture</h4>-->
        </div>""",
        unsafe_allow_html=True
    ) 
    st.divider()
    
    provider = st.radio("AI Provider", ["Google Gemini", "Ollama (Local)"])
    api_key, base_url, model_choice = None, None, None
    
    if provider == "Google Gemini":
        api_key = st.text_input("API Key", type="password", value=os.getenv("GEMINI_API_KEY"))
        model_choice = st.selectbox("Model", ["gemini-flash-latest", "gemini-3-pro-preview"])
    else:
        base_url = st.text_input("Ollama URL", "http://localhost:11434/v1")
        model_choice = st.selectbox("Model", ["gpt-oss:latest", "LLama3.2:1B"])
        st.warning("⚠️ Ensure Ollama is running locally.")

# ==========================================
# 3. MAIN UI
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "🚀 Onboarding",
    "🐳 Synthesis",
    "💬 Chats",
])

# --- TAB 1: ONBORDING ---
with tab1:
    col_input, col_view = st.columns(2)
    with col_input:
        st.subheader("Intent Definition")
        intent = st.text_area("Requirement", height=150, value="Create a 'Network Load Predictor' app. It needs to read channel statistics every 10ms and adjust the precoding matrix. Latency is critical (must be under 5ms). It needs to retrain every night using a central parameter server.  This is a core network function, so it needs privileged access.")            
        if st.button("Generate BOM", type="primary"):
            with st.spinner("Compiling BOM..."):
                data = query_ai_json(provider, api_key, base_url, model_choice, intent)                
                if isinstance(data, dict) and "error" not in data:
                    st.session_state.profile = RichMLAppProfile(**data)
                    # show descriptor json content
                    with st.expander("Raw Descriptor (JSON)", expanded=True):
                        st.json(st.session_state.profile.model_dump())
                    st.success("Profile Generated!")
                else:
                    st.error(f"Generation Failed: {data}")          

    with col_view:
        st.subheader("Architecture Preview")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Architecture Preview")

        if st.session_state.profile:
            profile = st.session_state.profile
          
            g = graphviz.Digraph()
            g.attr(rankdir="LR", fontname="Inter")
            g.node("App", profile.name, shape="box", style="rounded,filled", fillcolor="#e0f2fe")

            for o in profile.observables:
                g.node(o.name, o.name, shape="ellipse")
                g.edge(o.name, "App")

            for a in profile.actions:
                g.node(a.name, a.name, shape="component")
                g.edge("App", a.name)

            if profile.training.required:
                g.node('Train', label="Training Loop", fillcolor='#FFF9C4', style='dashed,filled')
                g.edge('App', 'Train', style='dashed')
                g.edge('Train', 'App', style='dashed')

            st.graphviz_chart(g, use_container_width=True)            
        else:
            st.info("Generate an app to preview architecture.")

        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: DEPLOY ---
with tab2: 
    st.subheader("🐳 Containerized AI Pipeline")

    # Status Check
    dex = st.session_state.get('docker', DockerExecutionEngine())
    if dex.is_available():
        st.success("🟢 Docker Daemon: CONNECTED")
    else:
        st.error("🔴 Docker Daemon: NOT FOUND. Please start Docker Desktop.")
        st.stop()
    
    if not dex.is_available():
        st.error("Docker Daemon not found. Start Docker Desktop.")
        st.stop()

    if not st.session_state.profile:
        st.warning("⚠️ Please generate an App Descriptor in Tab 1 first.")
        st.stop()

    profile = st.session_state.profile

    # Pipeline Visualization
    col_pipe_1, col_pipe_2 = st.columns(2)
    with col_pipe_1:
        st.markdown(f"**🎯 Training Container:** `{profile.training.resource.container}`")
    with col_pipe_2:
        st.markdown(f"**🚀 Deployment Container:** `{profile.inference.resource.container}`")
    
    st.divider()

    # ==========================================
    # PHASE 1: BUILD & TRAIN
    # ==========================================
    st.markdown("### 🛠️ Phase 1: Build & Train")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.info("Step A: Generate Artifacts")
        if st.button("Generate Training Code & Dockerfile"):
            with st.spinner("AI is coding..."):
                # 1. Python Script
                prompt_code = (
                    f"Output ONLY (no extra code or comments) a Python script 'train.py' for app '{profile.name}'. "
                    f"Logic: {profile.description}. "
                    "MUST DO: 1. Generate synthetic data.\n"
                    "MUST DO: 2. Train a dummy sklearn model.\n"
                    "MUST DO: 3. Save the 'train.py' script to '/context/train.py' (CRITICAL).\n"
                    "MUST DO: 4. Save the model to '/data/model.pkl' (CRITICAL).\n"
                    "MUST DO: 5. Print 'Step X/X' logs."
                )
                st.session_state.train_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_code)
                
                # 2. Dockerfile
                prompt_docker = (
                    f"Output ONLY (no extra code or comments) a Dockerfile to run the 'train.py' python script. "
                    "Base image: python:3.11-slim. no requirements.txt file allowed. "
                    "Install all dependencies (scikit-learn, joblib, pandas, etc). "
                    "WORKDIR /app."
                )
                st.session_state.dockerfile = query_ai_text(provider, api_key, base_url, model_choice, prompt_docker)
                
                # Clean Markdown
                st.session_state.dockerfile = st.session_state.dockerfile.replace("```dockerfile", "").replace("```", "")
                st.session_state.train_script = st.session_state.train_script.replace("```python", "").replace("```", "")

        # Display Artifacts        
        if 'dockerfile' in st.session_state:
            with st.expander("View Dockerfile"):               
                st.code(st.session_state.dockerfile, language="dockerfile")

        # Display Training Script
        if 'train_script' in st.session_state:
            with st.expander("View Training Script"):               
                st.code(st.session_state.train_script, language="python")
    
    with c2:
        st.info("Step B: Execution")
        if 'dockerfile' in st.session_state:
            if st.button("Build Image & Run Training"):
                # Build
                with st.status("Building Container Image...") as status:
                    res = dex.build_custom_image(st.session_state.dockerfile, "kiops-custom:latest")
                    if "Failed" in res:
                        status.update(label="Build Failed", state="error")
                        st.error(res)
                        st.stop()
                    status.update(label="Image Built Successfully!", state="complete")
                
                # Run
                container = dex.run_container("kiops-custom:latest", st.session_state.train_script, 
                "train.py", f"kiops-train-{profile.training.resource.container}")

                if container:
                    st.session_state.train_container = container
                    st.toast("Training Started!")

            # Training Logs
            if 'train_container' in st.session_state:
                st.caption("Training Logs:")
                st.empty()
                logs_ph = st.empty()
                logs_ph.code(dex.get_logs(st.session_state.train_container))
            st.divider()

    # ==========================================
    # PHASE 2: DEPLOY
    # ==========================================
    st.markdown("### 🚀 Phase 2: Deploy to Target")
    
    if st.button("Deploy Model"):
        with st.spinner(f"Deploying to {profile.inference.resource.container}..."):
            
            # 1. Generate Inference Script
            prompt_serve = (
                f"Write ONLY a Python script (no extra code or comments) 'serve.py' that loads a model from '/data/model.pkl'. "
                "Simulate an inference loop: Load model, enter while True loop, "
                "print 'Serving Request ID: x', sleep 1 second. "
                "Handle FileNotFoundError gracefully."
            )
            serve_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_serve)
            serve_script = serve_script.replace("```python", "").replace("```", "")
            
            # 2. Run Serving Container (Mocking a remote node)
            # We reuse the same image for simplicity, but use the new script
            container = dex.run_container("kiops-custom:latest", serve_script, "serve.py", 
                f"kiops-deploy-{profile.inference.resource.container}", mode="serving")
            
            if container:
                st.session_state.deploy_container = container
                st.success(f"Model Deployed to **{profile.inference.resource.container}** (Simulated)")

    # Deployment Logs
    if 'deploy_container' in st.session_state:
        st.caption("Live Inference Logs:")
        st.empty()
        logs_ph = st.empty()
        logs_ph.code(dex.get_logs(st.session_state.deploy_container))

# --- TAB 3: CHAT ---
with tab3:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])
    if q := st.chat_input("Ask about the network..."):
        st.session_state.messages.append({"role": "user", "content": q})
        ans = query_ai_text(provider, api_key, base_url, model_choice, f"Context: {st.session_state.profile}. Q: {q}")
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()
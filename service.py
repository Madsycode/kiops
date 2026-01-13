import os
import graphviz
import streamlit as st
from dotenv import load_dotenv

# Load Env
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
    page_title="6G MLOps Platform",
    initial_sidebar_state="expanded"
)

# Apply CSS
inject_custom_css()

# Initialize State
if 'profile' not in st.session_state: st.session_state.profile = None
if 'messages' not in st.session_state: st.session_state.messages = []
if 'docker' not in st.session_state: st.session_state.docker = DockerExecutionEngine()

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        """<div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 3rem;">📡</div>
            <h2 style="margin:0;">Network Ops</h2>
            <div style="font-size: 0.8rem; color: #666;">GenAI Orchestrator v2.0</div>
        </div>""",
        unsafe_allow_html=True
    ) 
    st.divider()
    
    provider = st.radio("AI Provider", ["Google Gemini", "Ollama (Local)"])
    api_key, base_url, model_choice = None, None, None
    
    if provider == "Google Gemini":
        api_key = st.text_input("API Key", type="password", value=os.getenv("GEMINI_API_KEY"))
        model_choice = st.selectbox("Model", ["gemini-3-flash-preview", "gemini-3-pro-preview"])
    else:
        base_url = st.text_input("Ollama URL", "http://localhost:11434/v1")
        model_choice = st.selectbox("Model", ["llama3.2:1b", "mistral", "qwen2.5:0.5b"])
        st.info("⚠️ Ensure Ollama is running locally.")
    
    st.divider()
    
    # Docker Status Indicator
    dex = st.session_state.docker
    if dex.is_available():
        st.success("Docker: Online")
    else:
        st.error("Docker: Offline")

# ==========================================
# 3. MAIN UI
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "🧩 Onboarding",
    "🏭 Synthesis Factory",
    "💬 Context Chat",
])

# --- TAB 1: ONBOARDING ---
with tab1:
    col_input, col_view = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("### 1. Define Intent")
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        intent = st.text_area(
            "What do you want to build?", 
            height=150, 
            value="Create a 'Beam-Predictor'. It analyzes signal-to-noise ratio (SNR) every 5ms from the base station and predicts the optimal beam index. It requires GPU acceleration and needs to retrain weekly on aggregated logs.",
            help="Describe inputs, outputs, constraints, and resources."
        )
        
        btn_gen = st.button("🚀 Generate Architecture Profile", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if btn_gen:
            with st.spinner("AI is analyzing requirements..."):
                data = query_ai_json(provider, api_key, base_url, model_choice, intent)                
                if isinstance(data, dict) and "error" not in data:
                    st.session_state.profile = RichMLAppProfile(**data)
                else:
                    st.error(f"Generation Failed: {data.get('error')}")

    with col_view:
        st.markdown("### 2. Architecture Blueprint")
        if st.session_state.profile:
            profile = st.session_state.profile
            
            # --- Better Visuals for Graphviz ---
            g = graphviz.Digraph()
            g.attr(rankdir="LR", splines="ortho", bgcolor="transparent")
            g.attr('node', fontname="Inter", fontsize="10", style="filled", shape="box")
            g.attr('edge', color="#58a6ff", penwidth="1.2")

            # Main App
            g.node("App", f"{profile.name}\n(Controller)", fillcolor="#1f6feb", fontcolor="white", shape="component")

            # Observables
            with g.subgraph(name="cluster_input") as c:
                c.attr(label="Data Plane", color="#30363d", fontcolor="#8b949e")
                for o in profile.observables:
                    c.node(o.name, f"{o.name}\n({o.frequency})", fillcolor="#238636", fontcolor="white", shape="ellipse")
                    g.edge(o.name, "App")

            # Actions
            with g.subgraph(name="cluster_action") as c:
                c.attr(label="Control Plane", color="#30363d", fontcolor="#8b949e")
                for a in profile.actions:
                    c.node(a.name, f"{a.name}\nTarget: {a.target}", fillcolor="#da3633", fontcolor="white", shape="box")
                    g.edge("App", a.name)

            # Resources
            if profile.training.required:
                r = profile.training.resource
                info = f"TRAINING LOOP\nRes: {r.memory}\nAcc: {r.accelerator}"
                g.node('Train', info, fillcolor='#d29922', fontcolor='black', style='dashed,filled')
                g.edge('App', 'Train', style='dashed', dir="both")

            st.graphviz_chart(g, use_container_width=True)
            
            with st.expander("📄 View JSON Descriptor"):
                st.json(st.session_state.profile.model_dump())
        else:
            st.info("👋 Generate a profile to see the architecture.")

# --- TAB 2: SYNTHESIS ---
with tab2:
    if not st.session_state.profile:
        st.warning("⚠️ Please generate a profile in the 'Onboarding' tab first.")
        st.stop()
        
    profile = st.session_state.profile
    st.subheader(f"🛠️ Fabrication: {profile.name}")
    
    col_code, col_run = st.columns([1, 1])
    
    # ----------------- CODE GENERATION -----------------
    with col_code:
        st.markdown("**Step A: Blueprint Code**")
        if st.button("📝 Generate Source Code"):
            with st.status("Writing Software...", expanded=True) as status:
                
                status.write("Generating 'train.py'...")
                prompt_code = (
                    f"Output ONLY (NO extra chracters, comments, etc.) a Python script 'train.py' for: {profile.description}. "
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Use 'sklearn' to create a dummy model.\n"
                    "2. Generate synthetic data based on the description.\n"
                    "3. Save the model to '/data/model.pkl'.\n"
                    "4. Print clear logs like 'Epoch 1/10', 'Loss: 0.x'.\n"
                    "5. Output ONLY code."
                )
                st.session_state.train_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_code)
                
                status.write("Generating Dockerfile...")
                prompt_docker = (
                    "Output ONLY (NO extra chracters, comments, etc.) a Dockerfile. Base: python:3.10-slim. "
                    "Run pip install scikit-learn pandas joblib. "
                    "Do NOT copy files (files are mounted at /app). "
                    "WORKDIR /app. "
                    "Output ONLY the Dockerfile content."
                )
                st.session_state.dockerfile = query_ai_text(provider, api_key, base_url, model_choice, prompt_docker)
                status.update(label="Artifacts Generated!", state="complete", expanded=False)

        if 'train_script' in st.session_state:
            with st.expander("View train.py", expanded=False):
                st.code(st.session_state.train_script, language='python')
            with st.expander("View Dockerfile", expanded=False):
                st.code(st.session_state.dockerfile, language='dockerfile')

    # ----------------- CONTAINER RUN -----------------
    with col_run:
        st.markdown("**Step B: Infrastructure Execution**")
        
        if 'dockerfile' in st.session_state and dex.is_available():
            if st.button("🏗️ Build & Train Model"):
                
                log_placeholder = st.empty()
                
                # 1. Build
                with st.status("Building Container Image...") as s:
                    res = dex.build_custom_image(st.session_state.dockerfile, "kiops-custom:latest")
                    if "Failed" in res or "Error" in res:
                        s.update(label="Build Failed", state="error")
                        st.error(res)
                        st.stop()
                    else:
                        s.update(label="Image Built Successfully", state="complete")
                
                # 2. Train
                with st.spinner("Initializing Training Container..."):
                    container = dex.run_container(
                        "kiops-custom:latest", 
                        st.session_state.train_script, 
                        "train.py", 
                        f"kiops-train-{profile.name.lower().replace(' ', '-')}"
                    )

                if container:
                    st.session_state.train_container = container
                    st.success("Training Completed!")
                    
                    # Show Logs
                    logs = dex.get_logs(container)
                    st.code(logs, language="text")

        elif not dex.is_available():
            st.warning("Docker is not available. Start Docker Desktop to run code.")

    st.divider()
    
    # ----------------- DEPLOYMENT -----------------
    st.subheader("🚀 Deployment")
    if st.button("Deploy Inference Node"):
        if dex.is_available():
            with st.spinner("Deploying..."):
                prompt_serve = (
                    "Write a 'serve.py' script. Load model from '/data/model.pkl'. "
                    "Infinite loop checking for input. Print 'Serving prediction...'. "
                    "Sleep 1s between checks. Output ONLY code."
                )
                serve_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_serve)
                
                container = dex.run_container(
                    "kiops-custom:latest", 
                    serve_script, 
                    "serve.py", 
                    f"kiops-deploy-{profile.name.lower()}", 
                    mode="serving"
                )
                
                if container:
                    st.session_state.deploy_container = container
                    st.success(f"Deployed to **{profile.inference.resource.container}**")

    if 'deploy_container' in st.session_state:
        st.caption("Live Inference Logs (Last 10 lines):")
        logs = dex.get_logs(st.session_state.deploy_container)
        st.code("\n".join(logs.splitlines()[-10:]))

# --- TAB 3: CHAT ---
with tab3:
    st.markdown("### Contextual Assistant")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])
    
    if q := st.chat_input("Ask about the generated model or architecture..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.write(q)
        
        # Context building
        ctx = f"Current App Profile: {st.session_state.profile.model_dump() if st.session_state.profile else 'None'}."
        ans = query_ai_text(provider, api_key, base_url, model_choice, f"Context: {ctx}\nUser Question: {q}")
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.write(ans)
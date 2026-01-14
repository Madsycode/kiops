import os
import time
import json
import graphviz
import requests
import streamlit as st
from dotenv import load_dotenv

# Local Imports
from models import RichMLAppProfile
from styles import inject_custom_css
from containerize import DockerExecutionEngine
from generative import query_ai_json, query_ai_text

# Load Env
load_dotenv()

# ==========================================
# 1. SETUP & STATE
# ==========================================

st.set_page_config(layout="wide", page_icon="📡", page_title="6G MLOps Platform", initial_sidebar_state="expanded")

# Apply CSS
inject_custom_css()

# Initialize State
if 'descriptor' not in st.session_state: st.session_state.descriptor = None
if 'messages' not in st.session_state: st.session_state.messages = []
if 'docker' not in st.session_state: st.session_state.docker = DockerExecutionEngine()

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        """<div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 3rem;">📡</div>
            <h2 style="margin:0;">6G MLOps Platform</h2>
            <div style="font-size: 0.8rem; color: #666;">Agentic MLOps Executive v1.0</div>
        </div>""",
        unsafe_allow_html=True
    ) 
    st.divider()
    
    provider = st.radio("Provider", ["Google Gemini", "Ollama (Local)"])
    api_key, base_url, model_choice = None, None, None
    
    if provider == "Google Gemini":
        model_choice = st.selectbox("Model", ["gemini-3-flash-preview", "gemini-3-pro-preview"])
        api_key = st.text_input("API Key", type="password", value=os.getenv("GEMINI_API_KEY"))
    else:
        model_choice = st.selectbox("Model", ["gemma3:27b", "gpt-oss:latest"])
        base_url = st.text_input("Ollama URL", "http://localhost:11434/v1")
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
    "🧩 Descriptor",
    "🏭 Factory",
    "💬 Chat",
])

# --- TAB 1: ONBOARDING ---
with tab1:
    col_input, col_view = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("### 1. Define Intent")
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        intent = st.text_area("What do you want to build?", height=150, 
            value= f"Create a 'Beam-Predictor' app. It analyzes the signal-to-noise ratio (SNR) every 50ms from the base station 'bs01' "
            "and predicts the optimal beam index. It requires GPU acceleration and needs to retrain daily on aggregated logs. "
            "The trained model must be deployed to the edge node 'edge-box-05' and inference is perform on port '5000' with the api endpoint '/predict'.",

            help="Describe inputs, outputs, constraints, and resources."
        )
        
        btn_gen = st.button("🚀 Generate Profile (BOM)", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if btn_gen:
            with st.spinner("AI is analyzing requirements..."):
                data = query_ai_json(provider, api_key, base_url, model_choice, intent)                
                if isinstance(data, dict) and "error" not in data:
                    st.session_state.descriptor = RichMLAppProfile(**data)
                else:
                    st.error(f"Generation Failed: {data.get('error')}")

    with col_view:
        st.markdown("### 2. Architecture Blueprint")
        if st.session_state.descriptor:
            descriptor = st.session_state.descriptor
            
            # --- Better Visuals for Graphviz ---
            g = graphviz.Digraph()
            g.attr(rankdir="LR", splines="ortho", bgcolor="transparent")
            g.attr('node', fontname="Inter", fontsize="10", style="filled", shape="box")
            g.attr('edge', color="#58a6ff", penwidth="1.2")

            # Main App
            g.node("App", f"{descriptor.name}\n(Controller)", fillcolor="#1f6feb", fontcolor="white", shape="component")

            # Observables
            with g.subgraph(name="cluster_input") as c:
                c.attr(label="Data Plane", color="#30363d", fontcolor="#8b949e")
                for o in descriptor.observables:
                    c.node(o.name, f"{o.name}\n({o.frequency})", fillcolor="#238636", fontcolor="white", shape="ellipse")
                    g.edge(o.name, "App")

            # Actions
            with g.subgraph(name="cluster_action") as c:
                c.attr(label="Control Plane", color="#30363d", fontcolor="#8b949e")
                for a in descriptor.actions:
                    c.node(a.name, f"{a.name}\nTarget: {a.target}", fillcolor="#da3633", fontcolor="white", shape="box")
                    g.edge("App", a.name)

            # Resources
            if descriptor.training.required:
                r = descriptor.training.resource
                info = f"TRAINING LOOP\nRes: {r.memory}\nAcc: {r.accelerator}"
                g.node('Train', info, fillcolor='#d29922', fontcolor='black', style='dashed,filled')
                g.edge('App', 'Train', style='dashed', dir="both")

            st.graphviz_chart(g, use_container_width=True)
            
            with st.expander("📄 View JSON Descriptor"):
                st.json(st.session_state.descriptor.model_dump())
        else:
            st.info("👋 Generate a descriptor to see the architecture.")

# --- TAB 2: SYNTHESIS ---
with tab2:
    if not st.session_state.descriptor:
        st.warning("⚠️ Please generate a descriptor in the 'Onboarding' tab first.")
        st.stop()
        
    descriptor = st.session_state.descriptor
    st.subheader(f"🛠️ Fabrication: {descriptor.name}")
    
    # --- VISUAL TOPOLOGY ---
    st.markdown("### 📦 Deployment Topology")
    t1, t2, t3 = st.columns(3)
    t1.info(f"**Build Host:**\n\n`kiops-train-{descriptor.training.resource.container}`")
    t2.success(f"**Target Node:**\n\n`{descriptor.inference.resource.container}`")
    t3.warning(f"**Service Port:**\n\n`localhost:5000`")
    st.divider()
    # -----------------------

    col_code, col_run = st.columns([1, 1])
    
    # ----------------- CODE GENERATION -----------------
    with col_code:
        st.markdown("**Step A: Blueprint Code**")
        if st.button("📝 Generate Source Code"):
            with st.status("Writing Software...", expanded=True) as status:
                
                status.write("Generating 'train.py'...")
                prompt_code = (
                    f"Output ONLY (NO extra chracters, comments, etc.) the Python code for: {descriptor.description}. "
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Use 'sklearn' to create a dummy model.\n"
                    "2. Generate synthetic data based on the description.\n"
                    "3. Save the model to '/data/model.pkl'.\n"
                    "4. Print clear logs like 'Epoch 1/10', 'Loss: 0.x'.\n"
                    "5. Do NOT wrap in markdown."
                )
                st.session_state.train_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_code)
                
                status.write("Generating Dockerfile...")
                prompt_docker = (
                    "Output ONLY (NO extra chracters, comments, etc.) a Dockerfile. Base: python:3.10-slim. "
                    "Run pip install scikit-learn pandas joblib flask. "  # ADDED FLASK
                    "Do NOT copy files (files are mounted at /app). "
                    "WORKDIR /app. "
                    "Output ONLY the Dockerfile content."
                    "Do NOT wrap in markdown."
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
                        f"kiops-train-{descriptor.name.lower().replace(' ', '-')}"
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
            with st.spinner(f"Deploying to {descriptor.inference.resource.container}..."):
                # 1. Generate Script
                prompt_serve = (
                    "Output ONLY (NO extra chracters, comments, etc.) a 'serve.py' script using Flask. "
                    f"1. Load model from '/data/model.pkl'. "
                    "2. Create POST /predict. "
                    "3. CRITICAL: Extract input list from JSON key 'input'. Example: `features = request.json['input']`. "
                    "4. CRITICAL: Convert to 2D array for sklearn: `prediction = model.predict([features])`. "
                    "5. Return JSON: `{'prediction': result.tolist()}`. "
                    "6. Run app(host='0.0.0.0', port=5000)."
                    "7 Do NOT wrap in markdown."
                )

                st.session_state.serve_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_serve)                
                with st.expander("View train.py", expanded=False):
                    st.code(st.session_state.serve_script, language='python')

                # 2. Run Container
                container = dex.run_container(
                    "kiops-custom:latest", 
                    st.session_state.serve_script, 
                    "serve.py", 
                    descriptor.inference.resource.container, 
                    mode="serving",
                    ports={'5000/tcp': 5000}
                )
                
                if container:
                    # 3. TRANSFER MODEL ARTIFACT (Fix applied here)
                    if 'train_container' in st.session_state:
                        with st.spinner("📦 Transferring Model Artifacts..."):
                            src_name = st.session_state.train_container.name
                            dst_name = container.name
                            # Copy from Train -> Deploy
                            res = dex.copy_model(src_name, "/data/model.pkl", dst_name, "/data/")
                            if res is True:
                                st.toast(f"Model transferred to {dst_name}", icon="📦")
                            else:
                                st.warning(f"Copy Warning: {res}")

                    st.session_state.deploy_container = container
                    st.success(f"Service active at http://localhost:5000 on node **{descriptor.inference.resource.container}**")


 # --- INTERACTIVE TESTING ---
    if 'deploy_container' in st.session_state:
        st.divider()
        st.markdown("#### 🧪 Live Inference Lab")
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.caption("📡 Container Logs")
            # Auto-refresh logs button
            if st.button("🔄 Refresh Logs", type="secondary", use_container_width=True):
                pass 
            st.code(dex.get_logs(st.session_state.deploy_container), language="text")
        
        with c2:
            st.caption("⚡ Test Client")
            
            # 1. AI Data Generator
            if st.button("🎲 Generate Synthetic Input"):
                with st.spinner("AI is generating context-aware test data..."):
                    # Construct prompt based on the specific App Profile
                    data_prompt = (
                        f"Output ONLY (NO extra chracters, comments, etc.) a valid JSON input payload for the ML app '{descriptor.name}'. "
                        f"Context: {descriptor.description}. "
                        f"Input Shape/Schema: {descriptor.inference.input_shape}. "
                        "Return ONLY a JSON object with a single key 'input' containing the data list/vector. "
                        "Example: {\"input\": [0.2, ...]}. "
                        "Do NOT wrap in markdown."
                    )
                    
                    # Use existing JSON helper
                    generated_data = query_ai_json(provider, api_key, base_url, model_choice, data_prompt)
                    
                    if isinstance(generated_data, dict) and "input" in generated_data:
                        st.session_state.test_payload = generated_data
                    else:
                        st.error("AI could not generate valid data. Using default.")
                        st.session_state.test_payload = {"input": [0.1]}

            # 2. Editable Payload UI
            default_val = st.session_state.get('test_payload', {"input": [0.5]})
            payload_str = st.text_area(
                "Request Payload (JSON)", 
                value=json.dumps(default_val, indent=2), 
                height=100
            )

            # 3. Send Request
            if st.button("🚀 Send Request", type="primary"):
                try:
                    payload = json.loads(payload_str)
                    st.write(f"Sending to: `http://localhost:5000/predict`")
                    
                    res = requests.post("http://localhost:5000/predict", json=payload, timeout=5)                    
                    st.success("Response Received:")
                    st.json(res.json())

                except json.JSONDecodeError:
                    st.error("Invalid JSON in payload field.")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
                    st.info("Ensure the container is running and Flask has started (check logs).")

# --- TAB 3: CHAT ---
with tab3:
    st.markdown("### Contextual Assistant")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])
    
    if q := st.chat_input("Ask about the generated model or architecture..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.write(q)
        
        # Context building
        ctx = f"Current App Profile: {st.session_state.descriptor.model_dump() if st.session_state.descriptor else 'None'}."
        ans = query_ai_text(provider, api_key, base_url, model_choice, f"Context: {ctx}\nUser Question: {q}")
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.write(ans)
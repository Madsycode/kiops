import os
import time
import json
import graphviz
import requests
import streamlit as st
from dotenv import load_dotenv


# Local Imports
from styles import inject_custom_css
from ontology import RichMLAppProfile
from containerize import DockerExecutionEngine
from generative import query_ai_json, query_ai_text
from helpers import show_descriptor, show_code, show_arch

# Load Env
load_dotenv()

# ==========================================
# 1. SETUP & STATE
# ==========================================

st.set_page_config(layout="wide", page_icon="📡", page_title="6G MLOps Platform", initial_sidebar_state="expanded")

# Apply CSS
inject_custom_css()

# Initialize State

if 'docker' not in st.session_state: st.session_state.docker = DockerExecutionEngine()
if 'descriptor' not in st.session_state: st.session_state.descriptor = None
if 'messages' not in st.session_state: st.session_state.messages = []

tab1, tab2, tab3 = st.tabs([
    "🧩 Descriptor",
    "🏭 Factory",
    "💬 Chat",
])

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown(
        """<div style="text-align: center; margin-bottom: 10px;">
            <div style="font-size: 6rem;">📡</div>
            <h1 style="margin:0;">Agentic MLOps Platform</h1>
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
        base_url = st.text_input("URL", "http://localhost:11434/v1")
        model_choice = st.selectbox("Model", ["gemma3:27b", "gpt-oss:latest"])
    
    st.divider()
    
    # Docker Status Indicator
    dex = st.session_state.docker
    if dex.is_available():
        st.success("Docker: ONLINE")
    else:
        st.error("Docker: OFFLINE")

# ==========================================
# TAB1
# ==========================================

with tab1:
    st.markdown("### 1. Define Intent")
    intent = st.text_area("What do you want to build?", height=500)    
                            
    if st.button("Generate Descriptor (BOM)", type="primary"):
        if not intent: 
            st.error(f"Not intent provided!")
            st.stop()

        with st.spinner("AI is analyzing requirements..."):
            data = query_ai_json(provider, api_key, base_url, model_choice, intent)                
            if isinstance(data, dict) and "error" not in data:
                st.session_state.descriptor = RichMLAppProfile(**data)                          
            else:
                st.error(f"Descriptor generation failed: {data.get('error')}")

    if st.session_state.descriptor:            
        show_arch(st.session_state.descriptor)        
        st.markdown("### 📄 Descriptor (JSON)")
        st.session_state.descriptor = show_descriptor(st.session_state.descriptor, 800)

# ==========================================
# TAB2
# ==========================================

with tab2:
    if not st.session_state.descriptor:
        st.warning("⚠️ Please generate a descriptor in the 'Onboarding' tab first.")
        st.stop()        
    descriptor = st.session_state.descriptor

    # service host name
    hostname = f"{descriptor.service.host}:{descriptor.service.port}/{descriptor.service.version}{descriptor.service.endpoint}"

    st.subheader(f"🛠️ {descriptor.name}")
    #st.markdown("### 📦 Deployment Topology")
    
    # --- VISUAL TOPOLOGY ---
    t1, t2, t3 = st.columns(3)
    t1.info(f"**Train Node:**\n\n`{descriptor.training.resource.container}`")
    t2.success(f"**Target Node:**\n\n`{descriptor.inference.resource.container}`")
    t3.warning(f"**Inference:**\n\n`{hostname}`")
    st.divider()

    # -----------------------

    col_code, col_run = st.columns([1, 1])
    
    # ----------------- CODE GENERATION -----------------
    st.subheader("📝 Blueprint Generation")
    if st.button("Generate Sources", type="primary"):
        with st.spinner(f"Generating code ..."):
            prompt_train = (
                "RETURN ONLY the Python training script. Do NOT include explanations, comments, markdown, or extra characters.\n\n"

                "TASK:\n"
                "Generate Python code to train a model based on the provided context.\n\n"

                "CONTEXT:\n"
                f"1. Dataset config: {descriptor.training.dataset.model_dump()}.\n"
                f"2. Model config: {descriptor.inference.model.model_dump()}.\n"
                f"3. Training config: {descriptor.training.model_dump()}.\n\n"

                "CRITICAL REQUIREMENTS:\n"
                f"1. Datasets are provided at runtime via a volume mounted at '/dataset'. Only load files from this path.\n"
                f"2. The FULL model (not the state_dict) object should be saved in the runtime-mounted folder '/models'.\n"
                "3. The model filename must follow the format '(name)_(version).(extension)'.\n"
                "4. Use the appropriate ML library inferred from the model config (e.g., sklearn, PyTorch, etc.).\n"
                "5. Include proper train/test split or cross-validation as appropriate.\n"
                "6. Include feature extraction if required by the dataset (e.g., converting JSON to arrays).\n"
                "7. Print clear training logs like 'Epoch 1/10', 'Loss: 0.x', or 'Training complete'.\n"
                "8. Use a random seed for reproducibility if applicable.\n"
                "9. Do not hardcode paths outside the mounted volumes.\n"
                "10. Do not write temporary files outside the dataset or model directories.\n"
            )

            st.session_state.train_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_train)
            
            prompt_docker = (
                "RETURN ONLY (NO extra chracters, comments, etc.) the Dockerfile content. "
                "Do NOT include explanations, markdown, comments, or extra characters. "
                "USE base image 'python:3.11-slim'\n\n"

                "CONTEXT:\n"
                f"1. Dataset config: {descriptor.training.dataset.model_dump()}.\n"
                f"2. Model config: {descriptor.inference.model.model_dump()}.\n"
                f"3. Training config: {descriptor.training.model_dump()}.\n\n"

                "CRITICAL CONSTRAINTS:\n"
                "1. The container will be used for BOTH training and inference.\n"
                f"2. Python scripts are injected at runtime via a volume mounted at '/scripts'.\n"
                f"3. Datasets are injected at runtime via a volume mounted at '/datasets'.\n"
                f"4. Models are injected at runtime via a volume mounted at '/models'.\n"
                "5. DO NOT use COPY or ADD scripts, datasets, or models.\n"
                "6. DO NOT assume any files exist in the image except what you install.\n"
                "7. DO NOT define ENTRYPOINT or CMD that overrides 'python <script>'.\n"
                "8. Keep the image minimal and production-safe.\n\n"

                "PACKAGE REQUIREMENTS:\n"
                "1. Always run: pip install --no-cache-dir scikit-learn pandas flask torch\n"
                "2. Install additional Python packages only if required by the context.\n"
                "3. If GPU acceleration is required, install torch WITHOUT CUDA-specific base images.\n\n"

                "IMAGE SETUP:\n"
                f"1. Set WORKDIR to {descriptor.service.workdir}.\n"
                "2. Do not expose ports explicitly (runtime handles ports).\n"
                "3. Do not include comments in the Dockerfile.\n"
            )

            st.session_state.dockerfile = query_ai_text(provider, api_key, base_url, model_choice, prompt_docker)
            st.toast("Dockerfile and train.py generated!", icon="✅")

    if 'dockerfile' in st.session_state:
        st.markdown("📄 Dockerfile")
        st.session_state.dockerfile = show_code(st.session_state.dockerfile, "dockerfile", 80)

    if 'train_script' in st.session_state:
        st.markdown("📄 train.py")
        st.session_state.train_script = show_code(st.session_state.train_script, "python", 500)               

    # ----------------- CONTAINER RUN -----------------
    if 'train_script' in st.session_state:
        st.subheader("🏗️ Infrastructure Execution")
        if dex.is_available():
            if st.button("Build & Train", type="primary"):
                
                log_placeholder = st.empty()
                
                # 1. Build
                with st.status("Building Container Image...") as s:
                    res = dex.build_custom_image(st.session_state.dockerfile, "kiops:latest")
                    if "Failed" in res or "Error" in res:
                        s.update(label="Build Failed", state="error")
                        st.error(res)
                        st.stop()
                    else:
                        s.update(label="Image Built Successfully", state="complete")
                
                # 2. Train
                with st.spinner("Initializing Training Container..."):
                    container = dex.run_container(
                        "kiops:latest", st.session_state.train_script, "train.py", 
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
    
    # ----------------- GENERATE SERVICE -----------------

    if 'train_container' in st.session_state:
        st.subheader("🚀 Service Deployment")
        if st.button("Generate Service", type="primary"):           
            with st.spinner(f"Generating code ..."):
                # 1. Generate Script
                prompt_serve = (
                "RETURN ONLY the Python source code. Do NOT include explanations, markdown, comments, or extra characters.\n\n"

                "TASK:\n"
                "Generate a Python inference service using Flask.\n\n"

                "CONTEXT:\n"
                f"1. Service config: {descriptor.service.model_dump()}.\n"
                f"2. Model config: {descriptor.inference.model.model_dump()}.\n\n"

                "CRITICAL CONSTRAINTS:\n"
                "1. This script is for INFERENCE ONLY. Do NOT include training logic.\n"
                f"2. The model object is stored in '/models' and must be loaded once at startup.\n"
                "4. The model filename follows the format '(name)_(version).(extension)'.\n"
                "5. The model is a full object not just the 'state_dict' do not use mode.eval()"
                "5. Do NOT assume any other files exist.\n\n"

                "MODEL LOADING:\n"
                "1. Load the model from disk using the appropriate library inferred from the model config.\n"
                "2. Store the loaded model in a global variable.\n\n"

                "API REQUIREMENTS:\n"
                f"1. Create a POST endpoint at '{descriptor.service.endpoint}'.\n"
                "2. Extract input features from JSON key 'input'. Example: features = request.json['input']\n"
                "3. Convert input to a 2D array for sklearn-style models. Example: prediction = model.predict([features])\n"
                "4. Return a JSON response exactly in the form: {'prediction': result.tolist()}\n\n"

                "SERVER CONFIGURATION:\n"
                f"1. Run the Flask app with host='0.0.0.0' and port={descriptor.service.port}.\n"
                "2. Do NOT enable debug mode.\n"
                "3. Do NOT add extra routes.\n"
            )

            # set server code
            st.session_state.serve_script = query_ai_text(provider, api_key, base_url, model_choice, prompt_serve)  

    if 'serve_script' in st.session_state:
        st.markdown("📄 server.py")              
        st.session_state.serve_script = show_code(st.session_state.serve_script, "python", 300)

    st.divider()

    # ----------------- DEPLOY SERVICE -----------------

    if 'serve_script' in st.session_state:
        if st.button("Deploy Service", type="primary"):
            with st.spinner(f"Deploying to {descriptor.inference.resource.container}..."):                    
                # 2. Run Container
                container = dex.run_container(
                    "kiops:latest", st.session_state.serve_script, "serve.py", 
                    descriptor.inference.resource.container, mode="serving",
                    ports={f'{descriptor.service.port}/tcp': descriptor.service.port} 
                )
                
                # 3. TRANSFER MODEL ARTIFACT 
                if container:
                    with st.spinner("📦 Transferring Model Artifacts..."):
                        model_name = f"{descriptor.inference.model.name}_{descriptor.inference.model.version}{descriptor.inference.model.file_extension}"
                        src_name = st.session_state.train_container.name
                        dst_name = container.name
                        
                        # Copy from Train -> Deploy
                        res = dex.copy_model(src_name, f"/models/{model_name}", dst_name, "/models")
                        if res is True:
                            st.toast(f"Model transferred to {dst_name}", icon="📦")
                        else:
                            st.warning(f"Copy Warning: {res}")

                    st.session_state.deploy_container = container                                
                    st.code(dex.get_logs(container), language="text")
                    st.success(f"Service active at {hostname} on node **{descriptor.inference.resource.container}**")


#  # --- INTERACTIVE TESTING ---
#     if 'deploy_container' in st.session_state:
#         st.divider()
#         st.markdown("#### 🧪 Live Inference Lab")
#         c1, c2 = st.columns([1, 1])
        
#         with c1:
#             st.caption("📡 Container Logs")
#             # Auto-refresh logs button
#             if st.button("🔄 Refresh Logs", type="secondary", use_container_width=False):
#                 pass 
#             st.code(dex.get_logs(st.session_state.deploy_container), language="text")
        
#         with c2:
#             st.caption("⚡ Test Service")
            
#             # 1. AI Data Generator
#             if st.button("🎲 Generate Synthetic Input"):
#                 with st.spinner("AI is generating context-aware test data..."):
#                     # Construct prompt based on the specific App Profile
#                     data_prompt = (
#                         "RETURN ONLY! (NO extra chracters, comments, etc.) a VALID JSON object with "
#                         "a single key 'input' containing the data list/vector base on this shape. "
#                         f"Input Shape/Schema: {descriptor.model.input_shape}. "
#                         "Example: {\"input\": [0.2, ...]}. "
#                         "DO NOT wrap in markdown."
#                     )
                    
#                     # Use existing JSON helper
#                     generated_data = query_ai_json(provider, api_key, base_url, model_choice, data_prompt)
                    
#                     if isinstance(generated_data, dict) and "input" in generated_data:
#                         st.session_state.test_payload = generated_data
#                     else:
#                         st.error("AI could not generate valid data. Using default.")
#                         st.session_state.test_payload = {"input": [0.1]}

#             # 2. Editable Payload UI
#             default_val = st.session_state.get('test_payload', {"input": [0.5]})
#             payload_str = st.text_area("API Request Payload (JSON)", 
#                 value=json.dumps(default_val, indent=4), 
#                 height=150
#             )

#             # 3. Send Request
#             if st.button("🚀 Send Request", type="primary"):
#                 try:
#                     payload = json.loads(payload_str)
#                     st.write(f"Sending to: `{hostname}`")
                    
#                     res = requests.post(f"{hostname}", json=payload, timeout=5)                    
#                     st.success("Response Received:")
#                     st.json(res.json())

#                 except json.JSONDecodeError:
#                     st.error("Invalid JSON in payload field.")
#                 except Exception as e:
#                     st.error(f"Connection Failed: {e}")
#                     st.info("Ensure the container is running and Flask has started (check logs).")

# ==========================================
# TAB3
# ==========================================

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
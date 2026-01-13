import docker
import os
import json
import io
from docker.errors import BuildError, APIError, NotFound

class DockerExecutionEngine:
    def __init__(self):
        self.client = None
        self.context_path = os.path.abspath("context")
        self.data_path = os.path.abspath("data") 
        
        # Create directories
        for p in [self.context_path, self.data_path]:
            os.makedirs(p, exist_ok=True)

        self._fix_wsl_docker_config()
                
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            print(f"Docker Init Warning: Ensure Docker is running. {e}")
            self.client = None

    def _fix_wsl_docker_config(self):
        """Removes problematic credsStore on WSL 2 quietly."""
        config_path = os.path.expanduser("~/.docker/config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if config.get("credsStore") == "desktop.exe":
                del config["credsStore"]
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
        except Exception:
            pass 

    def is_available(self):
        if not self.client: return False
        try:
            return self.client.ping()
        except Exception:
            return False

    def build_custom_image(self, dockerfile_content, tag):
        """Builds an image using the Low-Level API for stability."""
        if not self.client: return "Error: Docker client not initialized."
        
        # Convert string to file-like object
        f = io.BytesIO(dockerfile_content.encode('utf-8'))
        
        logs = []
        try:
            # Use Low-Level API (client.api.build) to avoid SDK wrapper bugs
            # decode=True guarantees we get a generator of dicts (JSON objects)
            build_generator = self.client.api.build(
                fileobj=f,
                tag=tag,
                rm=True,
                decode=True 
            )

            for chunk in build_generator:
                if 'error' in chunk:
                    return f"Build Error: {chunk['error']}\n\nTrace:\n{chunk.get('errorDetail', '')}"
                
                if 'stream' in chunk:
                    line = chunk['stream'].strip()
                    if line:
                        logs.append(line)
                        # Optional: Print to stdout for debugging
                        # print(line)
            
            return "\n".join(logs)

        except APIError as e:
            return f"Docker API Error: {e}"
        except Exception as e:
            return f"Unexpected Error: {e}"

    def run_container(self, image_tag, script_content, script_name, container_name, mode="training"):
        if not self.is_available(): return None
        
        # 1. Write the script to the host context folder
        host_script_path = os.path.join(self.context_path, script_name)
        with open(host_script_path, "w", encoding='utf-8') as f:
            f.write(script_content)
            
        # 2. Map Host Paths -> Container Paths
        volumes = {
            self.context_path: {'bind': '/app', 'mode': 'rw'},
            self.data_path: {'bind': '/data', 'mode': 'rw'}
        }
        
        try:
            # Clean up existing container
            try:
                old = self.client.containers.get(container_name)
                old.remove(force=True)
            except NotFound:
                pass

            # 3. Run
            container = self.client.containers.run(
                image_tag,
                command=["python", script_name], 
                volumes=volumes,
                detach=True,
                name=container_name,
                environment={"PYTHONUNBUFFERED": "1"}, # Critical for real-time logs
                working_dir="/app"
            )
            
            if mode == "training":
                # For training, we wait for completion
                container.wait()
                
            return container
        except Exception as e:
            print(f"Run Error: {e}")
            return None
        
    def get_logs(self, container):
        try:
            # reload container status to ensure we have latest data
            container.reload()
            # logs() returns bytes, so we MUST decode here
            return container.logs().decode("utf-8", errors="replace")
        except Exception:
            return "Error reading logs or container stopped."
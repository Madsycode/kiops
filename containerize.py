import os
import io
import json
import docker
from docker.errors import NotFound

class DockerExecutionEngine:
    def __init__(self):
        self.client = None
        self.context_path = os.path.abspath("context")
        self.data_path = os.path.abspath("data") 
        
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
        config_path = os.path.expanduser("~/.docker/config.json")
        if not os.path.exists(config_path): return
        try:
            with open(config_path, "r") as f: config = json.load(f)
            if config.get("credsStore") == "desktop.exe":
                del config["credsStore"]
                with open(config_path, "w") as f: json.dump(config, f, indent=4)
        except Exception: pass 

    def is_available(self):
        if not self.client: return False
        try: return self.client.ping()
        except Exception: return False

    def build_custom_image(self, dockerfile_content, tag):
        if not self.client: return "Error: Docker client not initialized."
        f = io.BytesIO(dockerfile_content.encode('utf-8'))
        logs = []
        try:
            build_generator = self.client.api.build(
                fileobj=f, tag=tag, rm=True, decode=True 
            )
            for chunk in build_generator:
                if 'error' in chunk:
                    return f"Build Error: {chunk['error']}\n{chunk.get('errorDetail', '')}"
                if 'stream' in chunk:
                    line = chunk['stream'].strip()
                    if line: logs.append(line)
            return "\n".join(logs)
        except Exception as e: return f"Unexpected Error: {e}"

    def run_container(self, image_tag, script_content, script_name, container_name, mode="training", ports=None):
        """
        Added 'ports' argument. 
        Format: {'5000/tcp': 5000} maps container 5000 to host 5000.
        """
        if not self.is_available(): return None
        
        host_script_path = os.path.join(self.context_path, script_name)
        with open(host_script_path, "w", encoding='utf-8') as f:
            f.write(script_content)
            
        volumes = {
            self.context_path: {'bind': '/app', 'mode': 'rw'},
            self.data_path: {'bind': '/data', 'mode': 'rw'}
        }
        
        try:
            try:
                old = self.client.containers.get(container_name)
                old.remove(force=True)
            except NotFound: pass

            container = self.client.containers.run(
                image_tag,
                command=["python", script_name], 
                volumes=volumes,
                detach=True,
                name=container_name,
                ports=ports if ports else {},
                environment={"PYTHONUNBUFFERED": "1"}, 
                working_dir="/app"
            )
            
            if mode == "training":
                container.wait()
                
            return container
        except Exception as e:
            print(f"Run Error: {e}")
            return None
        
    def copy_model(self, source_container_name, source_path, destination_container_name, destination_dir):
        """Copies files between containers using tar streams."""
        try:
            src = self.client.containers.get(source_container_name)
            dst = self.client.containers.get(destination_container_name)
            
            # 1. Get file as tar stream
            print(f"Copying {source_path} from {source_container_name}...")
            bits, stat = src.get_archive(source_path)
            
            # 2. Put tar stream into destination
            # Note: put_archive takes the directory where the file should be unpacked
            dst.put_archive(path=destination_dir, data=bits)
            return True
        except Exception as e:
            return f"Copy Error: {e}"
        
    def get_logs(self, container):
        try:
            container.reload()
            return container.logs().decode("utf-8", errors="replace")
        except Exception: return "Error reading logs."
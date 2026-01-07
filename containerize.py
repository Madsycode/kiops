import docker
import os
import time
import json

class DockerExecutionEngine:
    def __init__(self):
        self._fix_wsl_docker_config()
        self.context_path = os.path.abspath("context")
        self.data_path = os.path.abspath("data") # Shared volume folder
        
        # Create directories
        for p in [self.context_path, self.data_path]:
            if not os.path.exists(p):
                os.makedirs(p)
                
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Docker Init Error: {e}")
            self.client = None

    def _fix_wsl_docker_config(self):
        """WSL 2 Fix"""
        config_path = os.path.expanduser("~/.docker/config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                if "credsStore" in config and "desktop.exe" in config["credsStore"]:
                    del config["credsStore"]
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=4)
        except: pass

    def is_available(self):
        try:
            return self.client.ping()
        except:
            return False

    def build_custom_image(self, dockerfile_content, tag_name):
        """Builds an image from AI-generated Dockerfile string."""
        if not self.is_available(): return "Docker not running."
        
        # Save Dockerfile
        df_path = os.path.join(self.context_path, "Dockerfile")
        with open(df_path, "w") as f:
            f.write(dockerfile_content)
            
        print(f"🔨 Building {tag_name}...")
        try:
            # force rebuild to ensure new dependencies are added
            self.client.images.build(path=self.context_path, tag=tag_name, nocache=True)
            return "Build Complete"
        except Exception as e:
            return f"Build Failed: {e}"

    def run_container(self, image_tag, script_content, script_name, container_name, mode="training"):
        """
        Runs a container. 
        mode='training' -> Runs script and exits.
        mode='serving' -> Runs script and stays alive (detached).
        """
        if not self.is_available(): return None
        
        # Save script to host
        host_script_path = os.path.join(self.context_path, script_name)
        with open(host_script_path, "w") as f:
            f.write(script_content)
            
        # Mounts: Code to /app, Data to /data (shared)
        volumes = {
            self.context_path: {'bind': '/app', 'mode': 'rw'},
            self.data_path: {'bind': '/data', 'mode': 'rw'}
        }
        
        try:
            # Kill existing if exists
            try:
                old = self.client.containers.get(container_name)
                old.remove(force=True)
            except: pass

            container = self.client.containers.run(
                image_tag,
                command=f"python {script_name}",
                volumes=volumes,
                detach=True,
                name=container_name,
                environment={"PYTHONUNBUFFERED": "1"}, # Force logs
                working_dir="/app"
            )
            return container
        except Exception as e:
            print(f"Run Error: {e}")
            return None

    def get_logs(self, container):
        try:
            return container.logs().decode("utf-8")
        except:
            return "Error reading logs."
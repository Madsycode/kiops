from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ObservationHook(BaseModel):
    name: str = Field(..., description="Hook name (e.g., snr_stream)")
    schema: Literal["JSON", "YAML"] = Field(..., description="schema")
    frequency: str = Field(..., description="Frequency (e.g., 10ms, 100ms)")

class ActionHook(BaseModel):
    name: str = Field(..., description="Hook name")
    target: str = Field(..., description="Target (e.g., 'set-beam-index')")
    conflict_detection: bool = Field(..., description="Conflict detection?")

class ResourceRequirement(BaseModel):    
    memory: str = Field(..., description="Memory needs (e.g., 10MB)")
    accelerator: Literal["CPU", "GPU", "NPU"] = Field(..., description="Accelerator")
    container: str = Field(..., description="Docker container name (e.g., 'kiops-train-x')")

class DatasetConfig(BaseModel):
    format: str = Field(..., description="Dataset format (e.g., csv)")    
    filenames: List[str] = Field(default_factory=list, description="Filenames (e.g., ['/datasets/data.csv', '...'])")
    input_labels: List[str] = Field(default_factory=list, description="Training input labels (e.g., ['snr', 'cqi'])")
    target_labels: List[str] = Field(default_factory=list, description="Training output labels (e.g., ['beam_index'])")

class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version (e.g., 'v1', 'v2', etc.)")
    framework: str = Field(..., description="Model framework (e.g., Pytorch, etc.)")
    input_shape: str = Field(..., description="Input shape (e.g., (2), (2, 4))")
    output_shape: str = Field(..., description="Output shape (e.g., (1), (2, 2))")
    architecture: str = Field(..., description="Architecture (e.g., Sequential, etc.)")
    file_extension: str = Field(..., description="File extension (e.g., '.pth', '.pkl', etc.)")

class InferenceConfig(BaseModel):    
    model: ModelConfig = Field(..., description="ML model configs")
    resource: ResourceRequirement = Field(..., description="Resource requirement") 

class TrainingConfig(BaseModel):    
    required: bool = Field(..., description="Is training required?")
    dataset: DatasetConfig = Field(..., description="Training dataset") 
    period: Optional[str] = Field(..., description="(e.g., '24h') or null")
    resource: ResourceRequirement = Field(..., description="Resource requirement")     

class ServiceConfig(BaseModel):  
    version: str = Field(..., description="Version (e.g., v1)")
    port: str = Field(..., description="port number (e.g., 5000)")
    host: str = Field(..., description="host (e.g., 'http://localhost')")
    endpoint: str = Field(..., description="Endpoint (e.g., '/predict'")
    workdir: str = Field(..., description="Working dir (e.g., '/app'")

class RichMLAppProfile(BaseModel):
    name: str = Field(..., description="App name")
    desc: str = Field(..., description="App description")
    version: str = Field(..., description="App version (e.g., v1.0)")
    service: ServiceConfig = Field(..., description="Service configs")
    training: TrainingConfig = Field(..., description="Training configs")
    inference: InferenceConfig = Field(..., description="Inference configs")
    actions: List[ActionHook] = Field(default_factory=list, description="List of actions")
    observables: List[ObservationHook] = Field(default_factory=list, description="List of observations")
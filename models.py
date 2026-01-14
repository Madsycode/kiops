from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ObservationHook(BaseModel):
    schema: str = Field(..., description="json, yaml, etc.")
    name: str = Field(..., description="Hook name (e.g snr_stream)")
    frequency: str = Field(..., description="Frequency (e.g. 10ms, 100ms)")

class ActionHook(BaseModel):
    name: str = Field(..., description="Hook name")
    target: str = Field(..., description="Hook target (e.g. set-configs)")
    conflict_detection: bool = Field(..., description="Conflict detection?")

class ResourceConfig(BaseModel):    
    memory: str = Field(..., description="Memory needs, e.g., 10MB")
    container: str = Field(..., description="Docker container id (e.g. xxx-node-01)")
    accelerator: Literal["CPU", "GPU", "NPU"] = Field(..., description="Accelerator")

class InferenceConfig(BaseModel):
    name: str = Field(..., description="model name")
    framework: str = Field(..., description="e.g. Pytorch")
    version: str = Field(..., description="Version e.g. v1.0")
    port: str = Field(..., description="service port nnmber (e.g 5000)")
    endpoint: str = Field(..., description="service endpoint (e.g '/predict'")
    resource: ResourceConfig = Field(..., description="Inference enviroment") 
    output_shape: str = Field(..., description="Output shape (e.g (1), (2, 2))")
    input_shape: str = Field(..., description="Input shape (e.g (2), (2, 4), etc.)")

class TrainingConfig(BaseModel):
    required: bool = Field(..., description="Is training required?")
    resource: ResourceConfig = Field(..., description="Training enviroment") 
    period: Optional[str] = Field(..., description="Period (e.g. '24h') or null")

class RichMLAppProfile(BaseModel):
    name: str = Field(..., description="App name")
    description: str = Field(..., description="App description")
    training: TrainingConfig = Field(..., description="training resource config")
    inference: InferenceConfig = Field(..., description="inference resource configs")
    actions: List[ActionHook] = Field(default_factory=list, description="List of action hooks")
    observables: List[ObservationHook] = Field(default_factory=list, description="list of observation hooks")
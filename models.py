from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ObservationHook(BaseModel):
    name: str = Field(..., description="Hook name")
    frequency: str = Field(..., description="Frequency")
    data_schema: str = Field(..., description="json, yaml or null")

class ActionHook(BaseModel):
    name: str = Field(..., description="Action name")
    target: str = Field(..., description="Target component")
    conflict_detection: bool = Field(..., description="Conflict detection?")

class ResourceConfig(BaseModel):    
    memory: str = Field(..., description="Memory needs, e.g., 10MB")
    container: str = Field(..., description="Docker container (e.g. train-node-01)")
    accelerator: Literal["CPU", "GPU", "NPU"] = Field(..., description="Accelerator or null")

class InferenceConfig(BaseModel):
    name: str = Field(..., description="model name")
    version: str = Field(..., description="Version e.g. v1.0")
    framework: str = Field(..., description="e.g. Pytorch")
    input_shape: str = Field(..., description="Input shape or null")
    output_shape: str = Field(..., description="Output shape or null")
    resource: ResourceConfig = Field(..., description="Inference enviroment") 

class TrainingConfig(BaseModel):
    required: bool = Field(..., description="Is training required?")
    resource: ResourceConfig = Field(..., description="Training enviroment") 
    period: Optional[str] = Field(..., description="Period (e.g. '24h') or null")

# --- APPLICATION PROFILE ---

class RichMLAppProfile(BaseModel):
    name: str = Field(..., description="App name")
    description: str = Field(..., description="App description")
    observables: List[ObservationHook]    
    actions: List[ActionHook]
    training: TrainingConfig
    inference: InferenceConfig

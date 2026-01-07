from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ObservationHook(BaseModel):
    name: str = Field(..., description="Hook name")
    source_type: str = Field(..., description="Source entity")
    frequency: str = Field(..., description="Frequency")
    latency_budget: Optional[str] = Field(..., description="Max latency or null")
    reliability: Optional[float] = Field(..., description="Reliability or null")
    schema_definition: Optional[str] = Field(..., description="Schema or null")

class ActionHook(BaseModel):
    name: str = Field(..., description="Action name")
    target: str = Field(..., description="Target component")
    conflict_detection: bool = Field(..., description="Conflict detection?")
    rollback_supported: bool = Field(..., description="Rollback supported?")
    safety_critical: bool = Field(..., description="Safety critical?")

class ResourceRequirements(BaseModel):
    compute: str = Field(..., description="Compute needs")
    memory: str = Field(..., description="Memory needs")
    accelerator: Optional[Literal["CPU", "GPU", "NPU", "FPGA"]] = Field(..., description="Accelerator or null")
    deadline: Optional[str] = Field(..., description="Deadline or null")
    energy_budget: Optional[str] = Field(..., description="Energy budget or null")

class DataRequirements(BaseModel):
    input_data_size: str = Field(..., description="Input size")
    output_data_size: str = Field(..., description="Output size")
    input_format: str = Field(..., description="Input format")
    output_format: str = Field(..., description="Output format")
    data_freshness: Optional[str] = Field(..., description="Freshness or null")
    privacy_level: Optional[Literal["public", "restricted", "confidential"]] = Field(..., description="Privacy")

class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Version")
    framework: str = Field(..., description="Framework")
    input_shape: Optional[str] = Field(..., description="Input shape or null")
    output_shape: Optional[str] = Field(..., description="Output shape or null")
    parameter_count: Optional[int] = Field(..., description="Params or null")
    quantization: Optional[Literal["fp32", "fp16", "int8"]] = Field(..., description="Quantization or null")

class TrainingConfig(BaseModel):
    required: bool = Field(..., description="Training required?")
    retraining_period: Optional[str] = Field(..., description="Period or null")
    algorithm: Optional[str] = Field(..., description="Algorithm or null")
    pattern: Optional[Literal["centralized", "federated", "parameter_server"]] = Field(..., description="Pattern or null")
    online_learning: bool = Field(..., description="Online learning?")
    training_target_selector: Optional[str] = Field(..., description="Criteria for selecting training node (e.g. 'has_gpu')")

class PlacementConstraints(BaseModel):
    preferred_location: Optional[Literal["far_edge", "edge_cloud", "regional_cloud", "core_cloud"]] = Field(..., description="Location or null")
    mobility_support: bool = Field(..., description="Mobility support?")
    geo_constraints: Optional[str] = Field(..., description="Geo constraints or null")

class SecurityProfile(BaseModel):
    privileged_access: bool = Field(..., description="Privileged access?")
    isolation_level: Optional[Literal["process", "container", "vm", "bare_metal"]] = Field(..., description="Isolation")
    attestation_required: bool = Field(..., description="Attestation?")

class LifecycleConfig(BaseModel):
    startup_time: Optional[str] = Field(..., description="Startup time or null")
    upgrade_strategy: Optional[Literal["rolling", "blue_green", "canary"]] = Field(..., description="Upgrade strategy")
    health_checks: bool = Field(..., description="Health checks?")

class InfrastructureTarget(BaseModel):
    target_id: str = Field(..., description="Unique ID of the node (e.g., 'edge-node-04')")
    role: Literal["training_cluster", "inference_edge", "core_server"] = Field(..., description="Role of this node")
    ip_address: Optional[str] = Field(..., description="IP address or localhost")
    hardware_tier: Literal["low_power", "standard", "high_performance"] = Field(..., description="Hardware capability")
    docker_context: Optional[str] = Field(..., description="Docker context name or null (default)")

# --- APPLICATION PROFILE ---

class RichMLAppProfile(BaseModel):
    name: str 
    description: str
    ml_model: ModelConfig 
    actions: List[ActionHook]
    training_config: TrainingConfig
    observables: List[ObservationHook]
    data_requirements: DataRequirements
    inference_resources: ResourceRequirements
    lifecycle: Optional[LifecycleConfig] = Field(..., description="Lifecycle or null")
    placement: Optional[PlacementConstraints] = Field(..., description="Placement or null")
    deployment_target: InfrastructureTarget = Field(..., description="Where the final model runs")
    training_target: InfrastructureTarget = Field(..., description="Where the training happens")
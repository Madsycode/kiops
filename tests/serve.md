
Create a 'beam-predictor' that predicts the optimal radio beam index for a User-Equipment (UE) using radio signal measurements collected from the base station 'bs01'. The model must be retrain once every 24 hours using aggregated historical logs to adapt to changing conditions. GPU acceleration is required. The trained model must then be deploy as real-time inference service to the container 'edge-box-05' on port 5000. The service must accept signal measurement features as input and return the predicted optimal beam index as output.

DATA LOCATION AND FORMAT:
- The aggregated dataset is stored as CSV files under: '/datasets/data.csv'

TRAINING TARGETS: 
Use the following fields from the CSV files:
- Input features: snr_db, rsrp_dbm, rsrq_db, cqi, speed_mps, azimuth_deg, elevation_deg, beam_candidate
- Target label: beam_index

SERVICE CONFIGS
Use these formats for in/output depending on the context
- Request payload format (json): '{ "input": [1.0, ...] }'
- Response payload format (json): '{ "prediction": [[1.0]] }'

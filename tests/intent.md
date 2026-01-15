Generate a 'Beam-Predictor' app that predicts the optimal radio beam index for user equipment (UE) using radio signal measurements collected from base station "bs01". It analyzes signal-to-noise ratio (SNR) measurements sampled every 500 ms and infer the optimal beam index for each observation. The model must use GPU acceleration for training and inference. It must retrain once every 24 hours using aggregated historical logs to adapt to changing radio and mobility conditions. After training, deploy the model to the edge node 'edge-box-05'. Expose a real-time inference service on port 5000 with the API endpoint "/predict". The inference service must accept signal measurement features as input and return the predicted optimal beam index as output.

DATA LOCATION AND FORMAT:
- The aggregated dataset is stored as CSV files under: '/datasets/data.csv'

DATASET SCHEMA:
Use the following fields from the CSV files:
- Input features: snr_db, rsrp_dbm, rsrq_db, cqi, speed_mps, azimuth_deg, elevation_deg, beam_candidate
- Target label: beam_index
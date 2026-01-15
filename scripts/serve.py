import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

model = torch.load('/models/beam_prediction_model_v1.pth', weights_only=False)
model.eval()

@app.route('/v1/predict', methods=['POST'])
def predict():
    features = request.json['input']
    input_data = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        result = model(input_data)
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
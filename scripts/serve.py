import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = torch.load('/models/beam-predictor-model_v1.pth', weights_only=False)
model.eval()

@app.route('/v1/predict', methods=['POST'])
def predict():
    features = request.json['input']
    input_tensor = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        result = model(input_tensor)
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.load('/models/beam-predictor-model_v1.pth')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['input']
    input_tensor = torch.tensor([features])
    prediction = model(input_tensor)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
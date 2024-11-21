import torch
import torch.nn as nn
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

rf_model = joblib.load("diabetes_rf_model.pkl")

torch_model = NeuralNet()

model_path = str(pathlib.Path("diabetes_rf_model_neuralnetfastai.pkl").resolve())

torch.serialization.add_safe_globals(["fastai.tabular.learner.TabularLearner"])

try:
    torch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    torch_model.eval() 
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    
app = Flask(__name__)

CORS(app)

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        data = request.get_json()

        input_data = {
            'Age': [data['Age']],
            'Gender': [data['Gender']],
            'Polyuria': [data['Polyuria']],
            'Polydipsia': [data['Polydipsia']],
            'sudden weight loss': [data['sudden weight loss']],
            'weakness': [data['weakness']],
            'Polyphagia': [data['Polyphagia']],
            'Genital thrush': [data['Genital thrush']],
            'visual blurring': [data['visual blurring']],
            'Itching': [data['Itching']],
            'Irritability': [data['Irritability']],
            'delayed healing': [data['delayed healing']],
            'partial paresis': [data['partial paresis']],
            'muscle stiffness': [data['muscle stiffness']],
            'Alopecia': [data['Alopecia']],
            'Obesity': [data['Obesity']]
        }

        input_df = pd.DataFrame(input_data)

        prediction = rf_model.predict(input_df)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    try:
        data = request.get_json()

        input_data = [
            data['Age'], data['Gender'], data['Polyuria'], data['Polydipsia'], data['sudden weight loss'],
            data['weakness'], data['Polyphagia'], data['Genital thrush'], data['visual blurring'],
            data['Itching'], data['Irritability'], data['delayed healing'], data['partial paresis'],
            data['muscle stiffness'], data['Alopecia'], data['Obesity']
        ]

        input_tensor = torch.tensor([input_data], dtype=torch.float32)

        with torch.no_grad():
            prediction = torch_model(input_tensor)

        return jsonify({'prediction': prediction.item()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

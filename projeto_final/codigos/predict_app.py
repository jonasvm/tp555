import torch
import torch.nn as nn
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib

# Classe do modelo (rede neural)
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

# Carregar o modelo do Random Forest
rf_model = joblib.load("diabetes_rf_model.pkl")

# Carregar o modelo da rede neural
torch_model = NeuralNet()

# Converter o caminho do arquivo para uma string
model_path = str(pathlib.Path("diabetes_rf_model_neuralnetfastai.pkl").resolve())

# Adicionar os globais permitidos, caso o modelo dependa de classes externas (como TabularLearner)
torch.serialization.add_safe_globals(["fastai.tabular.learner.TabularLearner"])

# Carregar o modelo com a opção de segurança para objetos personalizados
try:
    # Carregar o modelo inteiro (sem weights_only=True)
    torch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    torch_model.eval()  # Definir o modelo para modo de avaliação
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Criar a aplicação Flask
app = Flask(__name__)

# Ativar o CORS para a aplicação inteira
CORS(app)

# Rota para fazer a predição usando o modelo Random Forest
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Receber os dados JSON da requisição
        data = request.get_json()

        # Organizar os dados para o formato necessário pelo modelo
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

        # Criar DataFrame com os dados recebidos
        input_df = pd.DataFrame(input_data)

        # Fazer a previsão
        prediction = rf_model.predict(input_df)

        # Retornar o resultado da predição como resposta JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Caso ocorra algum erro, retornar uma mensagem de erro
        return jsonify({'error': str(e)})

# Rota para fazer a predição usando o modelo Neural Network
@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    try:
        # Receber os dados JSON da requisição
        data = request.get_json()

        # Organizar os dados para o formato necessário pelo modelo
        input_data = [
            data['Age'], data['Gender'], data['Polyuria'], data['Polydipsia'], data['sudden weight loss'],
            data['weakness'], data['Polyphagia'], data['Genital thrush'], data['visual blurring'],
            data['Itching'], data['Irritability'], data['delayed healing'], data['partial paresis'],
            data['muscle stiffness'], data['Alopecia'], data['Obesity']
        ]

        # Converter os dados para um tensor
        input_tensor = torch.tensor([input_data], dtype=torch.float32)

        # Fazer a previsão
        with torch.no_grad():
            prediction = torch_model(input_tensor)

        # Retornar o resultado da predição como resposta JSON
        return jsonify({'prediction': prediction.item()})

    except Exception as e:
        # Caso ocorra algum erro, retornar uma mensagem de erro
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

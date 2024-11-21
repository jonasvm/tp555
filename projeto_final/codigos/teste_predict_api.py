from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Carregar o modelo
model = joblib.load("diabetes_rf_model.pkl")

# Criar a aplicação Flask
app = Flask(__name__)

# Ativar o CORS para a aplicação inteira
CORS(app)

# Rota para fazer a predição
@app.route('/predict', methods=['POST'])
def predict():
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
        prediction = model.predict(input_df)

        # Retornar o resultado da predição como resposta JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Caso ocorra algum erro, retornar uma mensagem de erro
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

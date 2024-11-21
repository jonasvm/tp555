from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

model = joblib.load("diabetes_rf_model.pkl")

app = Flask(__name__)

CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
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

        prediction = model.predict(input_df)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

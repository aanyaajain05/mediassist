import pickle
import boto3
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# AWS Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def get_llm_explanation(age, glucose, bmi, blood_pressure, insulin, prediction):
    result = "diabetic" if prediction == 1 else "not diabetic"
    prompt = f"""A patient has the following details:
- Age: {age}
- Glucose Level: {glucose}
- BMI: {bmi}
- Blood Pressure: {blood_pressure}
- Insulin Level: {insulin}

Our model predicts this patient is {result}.
In 3-4 simple sentences, explain what this means for their health and what they should do next."""

    body = json.dumps({
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    })

    response = bedrock.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=body
    )
    result_body = json.loads(response['body'].read())
    return result_body['output']['message']['content'][0]['text']

@app.route('/')
def home():
    print("Home route hit!")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = float(data['age'])
    glucose = float(data['glucose'])
    bmi = float(data['bmi'])
    blood_pressure = float(data['blood_pressure'])
    insulin = float(data['insulin'])

    input_df = pd.DataFrame([[age, glucose, bmi, blood_pressure, insulin]], 
                        columns=['Age', 'Glucose', 'BMI', 'BloodPressure', 'Insulin'])
    prediction = model.predict(input_df)[0]
    explanation = get_llm_explanation(age, glucose, bmi, blood_pressure, insulin, prediction)

    return jsonify({
        'prediction': int(prediction),
        'explanation': explanation
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
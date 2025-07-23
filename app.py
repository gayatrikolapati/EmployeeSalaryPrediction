from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)  # ✅ This line must come BEFORE any @app.route
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    age = int(request.form['age'])
    gender = request.form['gender']
    native_country = request.form['native-country']
    occupation = request.form['occupation']
    marital_status = request.form['marital-status']
    workclass = request.form['workclass']
    education = request.form['education']
    hours_per_week = int(request.form['hours-per-week'])

    # Load model and encoders
    model = pickle.load(open('salary_model.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

    # Prepare input dictionary
    input_dict = {
        'age': age,
        'gender': gender,
        'native-country': native_country,
        'occupation': occupation,
        'marital-status': marital_status,
        'workclass': workclass,
        'education': education,
        'hours-per-week': hours_per_week
    }

    # Encode using same encoders
    for col in input_dict:
        if col in label_encoders:
            input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

    # Build DataFrame in correct order
    input_df = pd.DataFrame([input_dict], columns=[
        'age', 'gender', 'native-country', 'occupation',
        'marital-status', 'workclass', 'education', 'hours-per-week'
    ])

    # Predict
    prediction = model.predict(input_df)[0]

    # Decode income class
    income_label = label_encoders['income'].inverse_transform([prediction])[0]

    return render_template('result.html', prediction=income_label)
if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

# Load the trained models
with open('linear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('naive_bayes.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for communication with the frontend

# Mappings for categorical variables in the Naive Bayes model
gender_mapping = {'Male': 0, 'Female': 1, 'Non-binary': 2, 'Prefer not to say': 3}
stress_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
severity_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'None': 3}
reverse_severity_mapping = {v: k for k, v in severity_mapping.items()}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from JSON
        data = request.json

        # Validate input
        required_keys = [
            'Sleep_Hours', 'Work_Hours', 'Age', 'Actual_Physical_Activity_Hours',
            'Gender', 'Stress_Level'
        ]
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing one or more required fields'}), 400

        # Extract input values
        sleep_hours = data['Sleep_Hours']
        work_hours = data['Work_Hours']
        age = data['Age']
        actual_activity_hours = data['Actual_Physical_Activity_Hours']
        gender = data['Gender']
        stress_level = data['Stress_Level']

        # Ensure numeric inputs for physical activity model
        try:
            sleep_hours = float(sleep_hours)
            work_hours = float(work_hours)
            age = float(age)
            actual_activity_hours = float(actual_activity_hours)
        except ValueError:
            return jsonify({'error': 'All numerical inputs must be valid numbers'}), 400

        # Predict physical activity hours
        physical_activity_input = pd.DataFrame({
            'Sleep_Hours': [sleep_hours],
            'Work_Hours': [work_hours],
            'Age': [age]
        })
        predicted_activity_hours = linear_model.predict(physical_activity_input)
        predicted_activity_hours_rounded = round(predicted_activity_hours[0])
        recommended_hours = max(0, predicted_activity_hours_rounded - actual_activity_hours)

        # Prepare input for severity prediction
        try:
            gender_encoded = gender_mapping[gender]
            stress_level_encoded = stress_level_mapping[stress_level]
        except KeyError:
            return jsonify({'error': 'Invalid value for Gender or Stress Level'}), 400

        severity_input = pd.DataFrame({
            'Gender': [gender_encoded],
            'Stress_Level': [stress_level_encoded],
            'Work_Hours': [work_hours],
            'Sleep_Hours': [sleep_hours],
            'Physical_Activity_Hours': [actual_activity_hours],
            'Age': [age]
        })
        predicted_severity = naive_bayes_model.predict(severity_input)[0]
        predicted_severity_label = reverse_severity_mapping[predicted_severity]

        # Return results
        return jsonify({
            'predicted_physical_activity_hours': predicted_activity_hours_rounded,
            'recommended_additional_hours': recommended_hours,
            'predicted_severity': predicted_severity_label
        })

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def predict():
    if not os.path.exists('forecast_results.pkl'):
        raise FileNotFoundError("Model file 'forecast_results.pkl' not found.")

    if not os.path.exists('data/Sales.csv'):
        raise FileNotFoundError("Sales data file 'data/Sales.csv' not found.")
    # Load the saved model and results
    with open('forecast_results.pkl', 'rb') as f:
        results_dict = pickle.load(f)
    
    forecast_df = results_dict['forecast_df']
    
    # Load the latest sales data
    sales = pd.read_csv('data/Sales.csv')
    
    # 🔍 Auto-detect date column
    date_cols = [col for col in sales.columns if 'date' in col.lower()]
    if not date_cols:
        raise KeyError("No column containing 'date' found in Sales.csv")

    date_col = date_cols[0]
    sales[date_col] = pd.to_datetime(sales[date_col])

    # Get the latest date from sales data
    latest_date = sales[date_col].max()
    
    # Filter predictions for the next period
    predictions = forecast_df[forecast_df['Days'].isin([1, 3, 5, 7, 30])]
    
    # Create response dictionary
    response = {
        'last_updated': latest_date.strftime('%Y-%m-%d'),
        'predictions': predictions.to_dict('records'),
        'status': 'success'
    }
    
    # Save to JSON file
    with open('prediction_results.json', 'w') as f:
        json.dump(response, f, indent=2)
    
    return response

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        result = predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
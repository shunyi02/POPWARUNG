from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
from dashscope import Application
import dashscope
from pydantic import BaseModel
from http import HTTPStatus
from typing import List # Import List

# Imports from predict.py
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import numpy as np # Import numpy for isnan

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DashScope Configuration
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
APP_ID = "a58f8b59b1ae482e8368c7ac0b23a5c6"

if not DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY is missing from environment variables")

# Define input schema
class ChatbotRequest(BaseModel):
    prompt: str
    inventory: dict
    lstm: dict
    date: str

# MySQL DB Connection
def get_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        return conn
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Chatbot endpoint
@app.post("/api/chatbot/")
def chatbot_analysis(request: ChatbotRequest):
    try:
        print("Received chatbot request:", request)
        biz_params = {
            "inventory": request.inventory,
            "lstm": request.lstm,
            "date": request.date
        }
        print("Calling DashScope Application.call with biz_params:", biz_params)

        response = Application.call(
            api_key=DASHSCOPE_API_KEY,
            app_id=APP_ID,
            prompt=request.prompt,
            biz_params=biz_params
        )
        print("DashScope Application.call response status_code:", response.status_code)
        print("DashScope Application.call response message:", response.message)
        print("DashScope Application.call response request_id:", response.request_id)
        print("DashScope Application.call response output:", response.output)


        if response.status_code != HTTPStatus.OK:
            print(f"DashScope API error: {response.message}")
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "error": response.message,
                    "request_id": response.request_id
                }
            )
        
        if not response.output or not response.output.text:
            print("DashScope response output or text is missing.")
            raise HTTPException(status_code=500, detail="Chatbot returned an empty response.")


        return {"response": response.output.text}

    except HTTPException as http_exc:
        print(f"HTTPException in chatbot_analysis: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in chatbot_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Example endpoint to fetch inventory from DB
@app.get("/inventory")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT productID, CurrentStock  FROM inventory ORDER BY productID ASC;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/inventoryall")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM qbi_file_20250517_21_45_35_0;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
@app.get("/forecastall")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM forecasted;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/orderall")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT *  FROM inventory ORDER BY productID ASC;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/salesall")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT *  FROM sales;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/product")
def get_inventory():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query ="""
        SELECT
            p.productID,
            p.name AS product_name,
            p.description,
            p.cost,
            i.CurrentStock,
            i.ReorderPoint,
            i.SafetyStock,
            i.LeadTime,
            CASE
                WHEN i.CurrentStock < i.ReorderPoint THEN TRUE
                ELSE FALSE
            END AS restock_required
            FROM
            Inventory i
            JOIN
            Product p ON i.productID = p.productID;

        """
        cursor.execute(query)
        #cursor.execute("SELECT productID, name, release_date, description, cost  FROM product ORDER BY productID ASC;")
        results = cursor.fetchall()
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/api/tables")
def list_tables():
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        return {"tables": tables}
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to get the latest sales date from the database
def get_latest_sales_date_from_db():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # Assuming 'Order Date' is the column name in the sales table
        cursor.execute("SELECT MAX(`order_date`) FROM sales;")
        latest_date = cursor.fetchone()[0]
        return latest_date
    except Error as e:
        print(f"Error fetching latest sales date: {e}")
        return None # Or raise an exception
    finally:
        if conn:
            conn.close()

# Prediction functionality from predict.py

class PredictionRequest(BaseModel):
    date: str # Date for prediction

def predict():
    # Load the saved model and results
    with open('forecast_results.pkl', 'rb') as f:
        results_dict = pickle.load(f)

    forecast_df = results_dict['forecast_df']

    # Get the latest date from the database instead of Sales.csv
    latest_date = get_latest_sales_date_from_db()

    if not latest_date:
        raise HTTPException(status_code=500, detail="Could not retrieve latest sales date from database.")

    # Note: The original script reads from Sales.csv and uses the latest date.
    # To use the prediction_date parameter, the prediction logic within this function
    # needs to be updated to handle prediction based on a specific date.
    # This modification is beyond current capabilities.
    # For now, the prediction_date parameter is accepted by the endpoint but not used in the prediction logic below.

    # Filter predictions for the next period (based on original script's logic)
    predictions = forecast_df[forecast_df['Days'].isin([1, 3, 5, 7, 30])]

    # Process predictions: change negative Forecasted to 0, NaN Weeks to 0
    processed_predictions = []
    for index, row in predictions.iterrows():
        processed_prediction = row.to_dict()
        processed_prediction['Forecasted'] = max(0, processed_prediction['Forecasted'])
        processed_prediction['Weeks'] = 0 if np.isnan(processed_prediction['Weeks']) else processed_prediction['Weeks']
        processed_predictions.append(processed_prediction)


    # Create response dictionary
    response = {
        'last_updated_data': latest_date.strftime('%Y-%m-%d') if latest_date else None, # Indicate which data was used
        'predictions': processed_predictions, # Use processed predictions
        'status': 'success'
    }

    # Save to JSON file (optional, can be removed if not needed)
    # with open('prediction_results.json', 'w') as f:
    #     json.dump(response, f, indent=2)

    return response

@app.post("/api/predict")
def predict_endpoint(request: PredictionRequest):
    try:
        # Pass the date from the request to the predict function
        result = predict()
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

# Define Prediction model for InsertPred endpoint
class Prediction(BaseModel):
    productID: str
    days: float # Assuming days can be float based on example data (1.0, 3.0, etc.)
    forecasted: float # Assuming forecasted is integer
    current_stock: float # Assuming current stock is integer
    stock_status: str
    weeks: float # Assuming weeks can be float

@app.post("/InsertPred")
def insert_predictions(predictions: List[Prediction]):
    print("*********printTest********:",predictions)
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        for p in predictions:
            print("*********print********:",p)
            cursor.execute("""
                INSERT INTO forecasted (productID, days, forecasted, current_stock, stock_status, weeks)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (p.productID, p.days, p.forecasted, p.current_stock, p.stock_status, p.weeks))
        conn.commit()
        return {"status": "success", "message": f"{len(predictions)} predictions inserted into forcast table."}
    except Error as e:
        # Rollback changes if an error occurs
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

# To run this FastAPI application, use a command like:
# uvicorn server:app --reload --port 3334
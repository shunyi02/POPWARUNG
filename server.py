from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from dotenv import load_dotenv
import os
from dashscope import Application
import dashscope
from pydantic import BaseModel
from http import HTTPStatus

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Set your DashScope key and app ID (Use env var in production!)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
APP_ID = "a58f8b59b1ae482e8368c7ac0b23a5c6"

# Define input schema
class ChatbotRequest(BaseModel):
    prompt: str
    inventory: dict
    lstm: dict
    date: str

# Get DB connection
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=3306
    )

# API: Get total earnings by month for a given year
@app.get("/api/sales")
def get_sales_by_year(year: int):
    try:
        conn = get_connection()
        print("success connection")
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT 
              MONTH(Date) AS month,
              SUM(TotalEarn) AS total_earn
            FROM qbi_file_20250517_13_04_46_0
        """
        cursor.execute(query, (year,))
        results = cursor.fetchall()

        # Fill in missing months
        monthly_totals = [0.0] * 12
        for row in results:
            monthly_totals[row["month"] - 1] = float(row["total_earn"])

        return monthly_totals

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
@app.get("/api/tables")
def list_tables():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()  # returns list of tuples like [('sales',), ('inventory',)]
        print(tables)

        return tables

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        raise HTTPException(status_code=500, detail="Database error")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# API: Get all inventory records
@app.get("/api/inventory")
def get_inventory():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT 
              Date, ProductID, StoreID, QtyIn, QtyOut, Balance
            FROM qbi_file_20250517_13_05_05_0
            ORDER BY Date DESC;
        """
        cursor.execute(query)
        return cursor.fetchall()

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
@app.post("/api/chatbot/")
def chatbot_analysis(request: ChatbotRequest):
    try:
        biz_params = {
            "inventory": request.inventory,
            "lstm": request.lstm,
            "date": request.date
        }

        response = Application.call(
            api_key=DASHSCOPE_API_KEY,
            app_id=APP_ID,
            prompt=request.prompt,
            biz_params=biz_params
        )

        if response.status_code != HTTPStatus.OK:
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "error": response.message,
                    "request_id": response.request_id
                }
            )

        return {"response": response.output.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

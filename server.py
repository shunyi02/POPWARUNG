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

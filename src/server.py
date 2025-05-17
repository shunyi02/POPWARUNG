from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use the specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

@app.get("/api/sales")
def get_sales_by_year(year: int):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT 
              MONTH(date) AS month,
              SUM(total_earn) AS total_earn
            FROM sales
            WHERE YEAR(date) = %s
            GROUP BY MONTH(date)
            ORDER BY MONTH(date);
        """
        cursor.execute(query, (year,))
        results = cursor.fetchall()

        # Fill missing months with 0
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

A smart AI-powered inventory and sales forecasting system for retail stores.


📁 Step 1: Setup .env File
Create a .env file in the root directory and add the following content:

//will delete soon
DB_HOST=popwarung.rwlb.rds.aliyuncs.com
DB_USER=pop_admin
DB_PASSWORD=Admin@123
DB_NAME=popwarung_db
DASHSCOPE_API_KEY= ...
DB_PORT=3306

🐍 Step 2: Install Python Dependencies
Run the following command to install required Python packages:

- pip install fastapi uvicorn mysql-connector-python python-dotenv dashscope pydantic pandas numpy

🚀 Step 3: Start the FastAPI Server
Navigate to the project root and start the backend server using Uvicorn:

- uvicorn server:app --host 0.0.0.0 --port 3334 --reload

🌐 Step 4: Run the Frontend
Install frontend dependencies and start the frontend app:

npm install
npm run dev
This should start the web UI on http://localhost:3000


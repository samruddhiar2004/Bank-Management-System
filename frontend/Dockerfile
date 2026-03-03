# 1. Use Python 3.9
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install Streamlit and Requests
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy frontend code
COPY . .

# 5. Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
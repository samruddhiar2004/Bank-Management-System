# 1. Use Python 3.9 for AI stability
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
COPY requirements.txt .
# Change line 9 to this:
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 4. Copy backend code and models
COPY . .

# 5. Run the FastAPI server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 80

# Run server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]
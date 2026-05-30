# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port for Flask API (5000) and Streamlit dashboard (8501)
EXPOSE 5000
EXPOSE 8501

# Standard Python buffer flags
ENV PYTHONUNBUFFERED=1

# Default command runs the Flask API
CMD ["python", "api/flask_app.py"]

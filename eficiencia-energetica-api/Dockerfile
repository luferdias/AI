# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set working directory to the api folder to run the server
WORKDIR /app/api

# Expose port 8000
EXPOSE 8000

# Start the FastAPI dev server by default
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0", "--port", "8000"]

# Use official Python image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 (default for Uvicorn)
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

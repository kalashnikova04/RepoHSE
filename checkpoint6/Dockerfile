FROM python:3.9

WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the container
COPY main.py model.pickle ./

# Expose port 8000 for the application
EXPOSE 8000

# Start the application using Uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8898", "main:app"]

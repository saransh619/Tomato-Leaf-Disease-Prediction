# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expects apps to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]

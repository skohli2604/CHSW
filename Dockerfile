# Use a base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your project
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run your app (Flask example)
CMD ["python", "synthetictest.py"]

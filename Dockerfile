FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a script to start our application
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set the entry point
ENTRYPOINT ["/start.sh"] 
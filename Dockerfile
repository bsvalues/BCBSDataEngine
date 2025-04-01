FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0
ENV DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
ENV SESSION_SECRET=bcbs_values_session_secret_key_2025
ENV API_KEY=bcbs_values_api_key_2025
ENV BCBS_VALUES_API_KEY=bcbs_values_api_key_2025
ENV LOG_LEVEL=INFO
ENV ENABLE_CACHING=true

# Expose the application port
EXPOSE 5000

# Run the diagnostic server
CMD ["python", "quick_diagnostic_server.py"]
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies if any (e.g., for psycopg2 if not using pure Python version, or other libs)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
# Copying alembic.ini and migrations directory for Alembic
COPY alembic.ini /app/
COPY migrations /app/migrations/
# Copy the application source code
COPY ./claims_processor /app/claims_processor

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application using Uvicorn
# Adjust the path to main:app if your main.py is elsewhere
CMD ["uvicorn", "claims_processor.src.main:app", "--host", "0.0.0.0", "--port", "8000"]

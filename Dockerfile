# Use the official OpenEnv base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Install project dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy the rest of the application
COPY . .

# Expose the port used by the environment
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

# Start the environment server
CMD ["python", "server/app.py", "--port", "8000"]

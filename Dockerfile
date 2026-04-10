# Use the official OpenEnv base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Copy the entire project first
# This is necessary because 'pip install -e .' requires the directories 
# defined in pyproject.toml to exist (e.g., 'server').
COPY . .

# Install project dependencies
RUN pip install -e .

# Expose the port used by the environment
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

# Start the environment server
CMD ["python", "server/app.py", "--port", "8000"]

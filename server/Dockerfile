FROM ghcr.io/meta-pytorch/openenv-base:latest
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["uvicorn", "codeforge_pro_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

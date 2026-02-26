FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
EXPOSE 8000
CMD ["python", "server.py", "sse"]

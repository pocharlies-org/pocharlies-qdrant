FROM python:3.12-alpine
WORKDIR /app
COPY server.py .
COPY index.html static/index.html
COPY favicon.svg static/favicon.svg
EXPOSE 5001
CMD ["python", "server.py"]

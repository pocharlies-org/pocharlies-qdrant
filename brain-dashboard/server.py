"""Minimal dashboard server for brain.e-dani.com — proxies health to RAG API."""
import http.server
import json
import socketserver
import urllib.request
import os

PORT = 5001
RAG_URL = os.environ.get("RAG_URL", "http://pocharlies-rag:5000")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/app/static", **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "brain-dashboard"}).encode())
            return

        if self.path == "/api/health":
            try:
                req = urllib.request.Request(f"{RAG_URL}/health", headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "detail": str(e)}).encode())
            return

        if self.path == "/metrics":
            try:
                req = urllib.request.Request(f"{RAG_URL}/metrics")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(502)
                self.end_headers()
            return

        # Serve index.html for all non-file paths (SPA)
        if not os.path.exists(os.path.join("/app/static", self.path.lstrip("/"))):
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        pass  # Silence logs

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Brain Dashboard running on :{PORT}")
        httpd.serve_forever()

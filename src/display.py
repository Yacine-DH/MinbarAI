import threading
import http.server
import socketserver
from pathlib import Path

PORT = 8080

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2">
    <style>
        body {{
            background-color: #000000;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
        }}
        .container {{
            text-align: center;
            padding: 40px;
            max-width: 90%;
        }}
        .german {{
            color: #FFFFFF;
            font-size: 52px;
            line-height: 1.5;
            margin-bottom: 30px;
        }}
        .arabic {{
            color: #888888;
            font-size: 28px;
            direction: rtl;
        }}
        .footer {{
            color: #444444;
            font-size: 18px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="german">{german}</div>
        <div class="arabic">{arabic}</div>
        <div class="footer">MinbarAI 🕌</div>
    </div>
</html>
"""

current_arabic = "في انتظار الخطبة..."
current_german = "Warte auf die Khutbah..."

def update_display(arabic_text, german_text):
    global current_arabic, current_german
    current_arabic = arabic_text
    current_german = german_text
    
    # Also write to file for backup
    with open("display.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(
            german=current_german,
            arabic=current_arabic
        ))

class DisplayHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/display.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html_content = HTML_TEMPLATE.format(
                german=current_german,
                arabic=current_arabic
            )
            self.wfile.write(html_content.encode("utf-8"))
        else:
            super().do_GET()

def start_server():
    with socketserver.TCPServer(("", PORT), DisplayHandler) as httpd:
        print(f"✅ HTTP Server running at http://localhost:{PORT}/", flush=True)
        print(f"   Open http://localhost:{PORT}/ in your browser", flush=True)
        httpd.serve_forever()

# Start server in background thread
def start_http_server():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

# Initialize display
update_display(current_arabic, current_german)
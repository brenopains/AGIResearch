"""Debug: test server response"""
import urllib.request, json
try:
    r = urllib.request.urlopen("http://127.0.0.1:8765/data", timeout=5)
    raw = r.read()
    print("Status:", r.status)
    print("Headers:", dict(r.headers))
    print("Raw bytes:", len(raw))
    print("Raw text:", raw[:300])
    if raw:
        d = json.loads(raw)
        print("JSON OK:", list(d.keys())[:5])
except Exception as e:
    print("ERROR:", type(e).__name__, e)

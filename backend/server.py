"""
SmartFlow Backend Server
Pure Python stdlib HTTP server + WebSocket-style SSE for real-time dashboard.
No FastAPI/uvicorn needed.

Endpoints:
  GET  /api/status          - all intersection states
  GET  /api/stats           - RL agent stats
  GET  /api/predict/<id>    - congestion prediction for intersection
  POST /api/train           - trigger training run
  POST /api/phase/<id>      - manually set phase duration
  GET  /stream              - SSE stream (text/event-stream) for dashboard
"""
import json
import sys
import os
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.traffic_generator import TrafficGenerator
from ml.signal_controller import TrafficManagementSystem
from ml.congestion_predictor import CongestionPredictor

# ─── Shared state ────────────────────────────────────────────────────────────
NUM_INTERSECTIONS = 20
TICK_INTERVAL = 2.0  # seconds between sim ticks

generator = TrafficGenerator(num_intersections=NUM_INTERSECTIONS)
intersection_ids = list(generator.intersections.keys())
tms = TrafficManagementSystem(intersection_ids)
predictor = CongestionPredictor(lookahead=5)

current_states: dict = {}
current_stats: dict = {}
sse_clients: list = []
sse_lock = threading.Lock()
state_lock = threading.Lock()

metrics = {
    "total_ticks": 0,
    "total_emergencies_handled": 0,
    "avg_wait_history": [],        # AI mode wait history
    "trad_wait_history": [],       # Traditional mode wait history
    "predictor_trained": False,
    "ai_mode": True,
    "weather_condition": "clear",
    "model": {},
}

# ─── Simulation loop ──────────────────────────────────────────────────────────
def simulation_loop():
    global current_states, current_stats
    states = generator.tick()

    while True:
        if generator.ai_mode:
            tms.step(states, generator)
        else:
            for iid in states:
                generator.set_phase_duration(iid, 30.0)
        new_states = generator.tick()
        if generator.ai_mode:
            tms.learn(new_states)

        # Attach predictions
        for iid, s in new_states.items():
            s["prediction"] = predictor.predict(s)
            # Track emergency handling
            if s.get("emergency") and states.get(iid, {}).get("emergency") is False:
                metrics["total_emergencies_handled"] += 1

        metrics["total_ticks"] += 1
        avg_wait = sum(s["avg_wait"] for s in new_states.values()) / len(new_states)
        avg_trad = sum(s["traditional_wait"] for s in new_states.values()) / len(new_states)
        total_fuel = sum(s["fuel_litres"] for s in new_states.values())
        total_co2 = sum(s["co2_kg"] for s in new_states.values())
        pedestrians = sum(s["pedestrian_waiting"] for s in new_states.values())
        active_accidents = sum(1 for s in new_states.values() if s.get("accident"))
        metrics["avg_wait_history"].append(round(avg_wait, 1))
        metrics["trad_wait_history"].append(round(avg_trad, 1))
        if len(metrics["avg_wait_history"]) > 300:
            metrics["avg_wait_history"].pop(0)
        if len(metrics["trad_wait_history"]) > 300:
            metrics["trad_wait_history"].pop(0)

        with state_lock:
            current_states = new_states
            current_stats = tms.stats()

        # Broadcast to SSE clients
        event_data = json.dumps({
            "states": new_states,
            "stats": current_stats,
            "metrics": {
                "total_ticks": metrics["total_ticks"],
                "avg_wait": round(avg_wait, 1),
                "avg_trad_wait": round(avg_trad, 1),
                "emergencies_handled": metrics["total_emergencies_handled"],
                "predictor_trained": metrics["predictor_trained"],
                "ai_mode": metrics["ai_mode"],
                "weather_condition": metrics["weather_condition"],
                "weather": generator.weather,
                "fuel_litres": round(total_fuel, 2),
                "co2_kg": round(total_co2, 2),
                "pedestrians_waiting": pedestrians,
                "active_accidents": active_accidents,
                "model": metrics["model"],
            }
        })
        with sse_lock:
            dead = []
            for client in sse_clients:
                try:
                    client["queue"].append(event_data)
                except Exception:
                    dead.append(client)
            for d in dead:
                sse_clients.remove(d)

        states = new_states
        time.sleep(TICK_INTERVAL)


def training_thread():
    """Train predictor in background."""
    print("[Backend] Training congestion predictor in background...")
    dataset_path = os.environ.get("SMARTFLOW_DATASET", "").strip()
    try:
        result = predictor.train_csv(dataset_path) if dataset_path else predictor.train(n_steps=800)
    except Exception as exc:
        print(f"[Backend] Dataset training failed ({exc}); using synthetic training.")
        result = predictor.train(n_steps=800)
    metrics["predictor_trained"] = True
    metrics["model"] = result
    print(f"[Backend] Predictor trained: {result}")


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class SmartFlowHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default access log

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._serve_dashboard()
        elif path == "/api/status":
            with state_lock:
                self._send_json({"intersections": current_states, "tick": metrics["total_ticks"]})
        elif path == "/api/stats":
            with state_lock:
                self._send_json({"agents": current_stats, "metrics": metrics})
        elif path.startswith("/api/predict/"):
            iid = path.split("/")[-1]
            with state_lock:
                s = current_states.get(iid)
            if not s:
                self._send_json({"error": "intersection not found"}, 404)
            else:
                self._send_json(predictor.predict(s))
        elif path == "/api/history":
            self._send_json({
                "avg_wait_history": metrics["avg_wait_history"],
                "trad_wait_history": metrics["trad_wait_history"],
            })
        elif path == "/api/intersections":
            # Returns all intersection metadata (id, name, lat, lng)
            meta = [
                {"id": iid, **generator.intersection_meta.get(iid, {})}
                for iid in generator.intersections.keys()
            ]
            self._send_json({"intersections": meta})
        elif path == "/api/comparison":
            # AI vs Traditional comparison stats
            ai_hist = metrics["avg_wait_history"]
            tr_hist = metrics["trad_wait_history"]
            n = min(len(ai_hist), len(tr_hist), 100)
            if n > 0:
                ai_avg = sum(ai_hist[-n:]) / n
                tr_avg = sum(tr_hist[-n:]) / n
                saving_pct = round((tr_avg - ai_avg) / tr_avg * 100, 1) if tr_avg > 0 else 0
            else:
                ai_avg = tr_avg = saving_pct = 0
            self._send_json({
                "ai_avg_wait": round(ai_avg, 1),
                "traditional_avg_wait": round(tr_avg, 1),
                "saving_percent": saving_pct,
                "ticks_compared": n,
            })
        elif path == "/api/analytics":
            with state_lock:
                states = list(current_states.values())
            self._send_json({
                "avg_wait": round(sum(s["avg_wait"] for s in states) / len(states), 1) if states else 0,
                "total_queue": sum(s["total_queue"] for s in states),
                "fuel_litres": round(sum(s.get("fuel_litres", 0) for s in states), 2),
                "co2_kg": round(sum(s.get("co2_kg", 0) for s in states), 2),
                "pedestrians_waiting": sum(s.get("pedestrian_waiting", 0) for s in states),
                "active_accidents": sum(1 for s in states if s.get("accident")),
                "weather": generator.weather,
                "model": metrics["model"],
            })
        elif path == "/stream":
            self._handle_sse()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if path == "/api/train":
            dataset_path = str(body.get("dataset_path", "")).strip()
            if dataset_path:
                os.environ["SMARTFLOW_DATASET"] = dataset_path
            t = threading.Thread(target=training_thread, daemon=True)
            t.start()
            self._send_json({"status": "training started in background"})
        elif path.startswith("/api/phase/"):
            iid = path.split("/")[-1]
            duration = body.get("duration", 30)
            generator.set_phase_duration(iid, float(duration))
            self._send_json({"status": "ok", "intersection": iid, "duration": duration})
        elif path == "/api/weather":
            condition = body.get("condition", "clear")
            generator.set_weather(condition)
            metrics["weather_condition"] = condition
            self._send_json({"status": "ok", "weather": condition})
        elif path == "/api/emergency":
            iid = body.get("intersection_id")
            if iid and iid in generator.intersections:
                generator.trigger_emergency(iid)
                self._send_json({"status": "ok", "emergency_at": iid})
            else:
                self._send_json({"error": "invalid intersection_id"}, 400)
        elif path == "/api/accident":
            iid = body.get("intersection_id")
            active = bool(body.get("active", True))
            if iid and generator.set_accident(iid, active):
                self._send_json({"status": "ok", "accident_at": iid, "active": active})
            else:
                self._send_json({"error": "invalid intersection_id"}, 400)
        elif path == "/api/add_traffic":
            iid = body.get("intersection_id")
            mult = float(body.get("multiplier", 3.0))
            if iid and iid in generator.intersections:
                generator.add_traffic_spike(iid, mult)
                self._send_json({"status": "ok", "spike_at": iid, "multiplier": mult})
            else:
                self._send_json({"error": "invalid intersection_id"}, 400)
        elif path == "/api/ai_mode":
            enabled = bool(body.get("enabled", True))
            generator.set_ai_mode(enabled)
            metrics["ai_mode"] = enabled
            self._send_json({"status": "ok", "ai_mode": enabled})
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_sse(self):
        """Server-Sent Events stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        import queue as Q
        q = Q.Queue(maxsize=10)
        client = {"queue": []}

        with sse_lock:
            sse_clients.append(client)

        try:
            while True:
                if client["queue"]:
                    data = client["queue"].pop(0)
                    msg = f"data: {data}\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                else:
                    # Heartbeat every 3s
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    time.sleep(3)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with sse_lock:
                if client in sse_clients:
                    sse_clients.remove(client)

    def _serve_dashboard(self):
        """Serve the built-in HTML dashboard."""
        html = DASHBOARD_HTML
        self._send_html(html)


# ─── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartFlow — Nagpur AI Traffic</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#07111f;--card:#111d2f;--card2:#15243a;--border:#263a54;--text:#eaf2ff;--sub:#9fb1c9;--dim:#6f829d;--blue:#4da3ff;--cyan:#2dd4bf;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b}
body{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:radial-gradient(circle at 20% 0,#123159 0,transparent 34%),var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}
.topbar{background:rgba(10,24,42,.92);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;flex-shrink:0}
.logo{font-size:17px;font-weight:700;color:var(--blue);white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.conn{font-size:12px;color:var(--sub)}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tick-lbl{font-size:12px;color:var(--dim)}
.controls{background:rgba(8,19,34,.88);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.ctrl-group{display:flex;align-items:center;gap:6px}
.ctrl-label{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.btn{padding:6px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;cursor:pointer;transition:all .15s;white-space:nowrap}
.btn:hover{background:#2d3748}
.btn.active{background:#1d4ed8;border-color:#3b82f6;color:#fff}
.btn.danger{background:#7f1d1d;border-color:#ef4444;color:#fca5a5}
.btn.warn{background:#713f12;border-color:#f59e0b;color:#fde68a}
.btn.success{background:#14532d;border-color:#22c55e;color:#86efac}
.btn.purple{background:#4c1d95;border-color:#a855f7;color:#e9d5ff}
select{padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;cursor:pointer}
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:13px}
.toggle{position:relative;width:44px;height:24px}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;inset:0;background:#374151;border-radius:12px;cursor:pointer;transition:.3s}
.slider:before{content:"";position:absolute;height:18px;width:18px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.slider{background:#2563eb}
input:checked+.slider:before{transform:translateX(20px)}
.metrics{display:grid;grid-template-columns:repeat(10,minmax(105px,1fr));gap:9px;padding:12px 20px;background:rgba(8,19,34,.72);overflow-x:auto}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(5,1fr)}}
.metric{background:linear-gradient(145deg,var(--card2),var(--card));border:1px solid var(--border);border-radius:12px;padding:10px 13px;box-shadow:0 10px 25px rgba(0,0,0,.12)}
.mlabel{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.mval{font-size:20px;font-weight:700;color:var(--blue)}
.msub{font-size:10px;color:#475569;margin-top:1px}
.main{display:flex;flex:1;overflow:hidden;min-height:0}
.map-panel{flex:1.4;position:relative;min-width:0}
#map{width:100%;height:100%;min-height:400px}
.weather-overlay{position:absolute;inset:0;z-index:700;pointer-events:none;opacity:0;transition:opacity .35s;overflow:hidden}
.weather-overlay.rain,.weather-overlay.storm{opacity:1;background-image:repeating-linear-gradient(105deg,transparent 0 14px,rgba(125,211,252,.24) 15px 17px,transparent 18px 34px);background-size:140px 140px;animation:rainfall .55s linear infinite}
.weather-overlay.fog{opacity:1;background:linear-gradient(90deg,rgba(220,235,245,.12),rgba(220,235,245,.38),rgba(220,235,245,.12));animation:fogdrift 7s ease-in-out infinite}
.weather-overlay.storm{background-color:rgba(28,35,58,.28);animation:rainfall .42s linear infinite,flash 5s infinite}
@keyframes rainfall{to{background-position:-60px 140px}}
@keyframes fogdrift{50%{transform:translateX(6%);opacity:.72}}
@keyframes flash{0%,92%,100%{filter:none}94%{filter:brightness(1.8)}96%{filter:brightness(.8)}}
.map-legend{position:absolute;bottom:16px;left:16px;z-index:1000;background:rgba(15,17,23,.92);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:11px;display:flex;flex-direction:column;gap:5px}
.leg-row{display:flex;align-items:center;gap:7px}
.leg-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.gps-btn{position:absolute;top:16px;right:16px;z-index:1000;background:var(--blue);color:#fff;border:none;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;font-weight:600}
.sidebar{width:340px;flex-shrink:0;overflow-y:auto;display:flex;flex-direction:column;gap:0;border-left:1px solid var(--border)}
.sidebar-section{padding:14px 16px;border-bottom:1px solid var(--border)}
.sec-title{font-size:12px;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.chart-wrap{height:90px;position:relative}
.chart-wrap svg{width:100%;height:100%}
.chart-legend{display:flex;gap:14px;margin-top:6px}
.cleg{font-size:10px;display:flex;align-items:center;gap:4px;color:var(--sub)}
.cleg-line{width:20px;height:2px;border-radius:1px}
.cmp-bars{display:flex;flex-direction:column;gap:6px;margin-top:4px}
.cmp-row{display:flex;align-items:center;gap:8px;font-size:12px}
.cmp-lbl{width:80px;color:var(--sub);flex-shrink:0}
.cmp-bar-bg{flex:1;height:16px;background:#1e293b;border-radius:4px;overflow:hidden}
.cmp-bar-fill{height:100%;border-radius:4px;transition:width .6s}
.cmp-val{width:40px;text-align:right;font-weight:600;font-size:12px}
.saving-badge{margin-top:8px;text-align:center;font-size:13px;font-weight:700;color:#4ade80;background:#14532d;border-radius:8px;padding:6px}
.int-list{display:flex;flex-direction:column;gap:6px;max-height:320px;overflow-y:auto}
.int-item{background:#0f1117;border-radius:8px;padding:8px 10px;cursor:pointer;border:1px solid transparent;transition:border .15s}
.int-item:hover,.int-item.selected{border-color:var(--blue)}
.int-header{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.int-name{font-size:13px;font-weight:600;flex:1}
.int-badges{display:flex;gap:4px}
.badge{font-size:9px;padding:2px 7px;border-radius:99px;font-weight:700}
.badge-red{background:#7f1d1d;color:#fca5a5}
.badge-yellow{background:#713f12;color:#fde68a}
.badge-blue{background:#1e3a5f;color:#93c5fd}
.badge-purple{background:#4c1d95;color:#e9d5ff}
.int-lanes{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.int-lane{background:#1a1f2e;border-radius:4px;padding:3px 5px;text-align:center}
.int-lane-dir{font-size:9px;color:var(--dim)}
.int-lane-q{font-size:13px;font-weight:700}
.int-pred{font-size:10px;padding:2px 7px;border-radius:5px;display:inline-block;margin-top:5px}
.pred-low{background:#14532d;color:#4ade80}
.pred-medium{background:#713f12;color:#fde68a}
.pred-high{background:#7f1d1d;color:#fca5a5}
.pred-unknown{background:#1e293b;color:var(--dim)}
.nearest-card{background:#0f1117;border-radius:8px;padding:10px;border:1px solid #2563eb}
.nearest-name{font-size:15px;font-weight:700;color:var(--blue);margin-bottom:6px}
.nearest-detail{font-size:12px;color:var(--sub);line-height:1.8}
.weather-chips{display:flex;gap:6px;flex-wrap:wrap}
.wchip{padding:5px 12px;border-radius:99px;border:1px solid var(--border);font-size:12px;cursor:pointer;background:var(--card);transition:all .15s}
.wchip.active{background:#1d4ed8;border-color:#3b82f6;color:#fff}
.footer{padding:8px 20px;text-align:center;font-size:10px;color:#374151;border-top:1px solid var(--border);flex-shrink:0}
/* ── Ambulance route panel ── */
.ambu-panel{background:#1a0a2e;border:1px solid #a855f7;border-radius:8px;padding:10px;font-size:12px;display:none}
.ambu-panel.active{display:block}
.ambu-step{display:flex;align-items:center;gap:6px;padding:3px 0;color:#e9d5ff}
.ambu-step .dot-step{width:8px;height:8px;border-radius:50%;background:#a855f7;flex-shrink:0}
/* ── Signal popup override ── */
.leaflet-popup-content-wrapper{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;color:#e2e8f0;box-shadow:0 8px 32px #00000088}
.leaflet-popup-tip{background:#1a1f2e}
.leaflet-popup-content{margin:12px 16px;min-width:200px}
.sig-popup-title{font-size:14px;font-weight:700;margin-bottom:8px;color:#60a5fa}
.sig-popup-row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;border-bottom:1px solid #2d3748}
.sig-popup-row:last-of-type{border:none}
.sig-popup-actions{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.sig-popup-btn{padding:4px 10px;border-radius:6px;border:1px solid;font-size:11px;cursor:pointer;font-weight:600}
/* ── Heatmap circles ── */
.heat-circle{border-radius:50%;pointer-events:none}
/* ── Accident marker ── */
@keyframes accident-pulse{0%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:.6}100%{transform:scale(1);opacity:1}}
.accident-icon{animation:accident-pulse 1s infinite}
.countdown-pill{display:inline-flex;min-width:34px;justify-content:center;padding:2px 7px;border-radius:999px;background:#0b2744;color:#7dd3fc;font-weight:800;font-variant-numeric:tabular-nums}
.ped-badge{background:#0f3d3a;color:#5eead4}
/* SmartFlow Hybrid UI */
body{display:block;min-height:100vh;overflow-x:hidden;background:linear-gradient(145deg,#07111f,#091827 48%,#0b1422)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(77,163,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(77,163,255,.025) 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(to bottom,#000,transparent 85%)}
.app-shell{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:100vh;transition:grid-template-columns .35s cubic-bezier(.2,.8,.2,1)}
.app-shell.nav-collapsed{grid-template-columns:76px minmax(0,1fr)}
.nav-rail{position:sticky;top:0;height:100vh;padding:18px 12px;background:rgba(5,15,27,.94);border-right:1px solid var(--border);backdrop-filter:blur(22px);z-index:1500;display:flex;flex-direction:column;gap:18px;overflow:hidden}
.brand{display:flex;align-items:center;gap:10px;padding:6px 8px;min-height:44px}.brand-mark{width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--cyan));display:grid;place-items:center;color:#06111e;font-weight:900;box-shadow:0 0 28px rgba(45,212,191,.28)}
.brand-copy{white-space:nowrap}.brand-title{font-weight:800;letter-spacing:.02em}.brand-sub{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.16em;margin-top:2px}
.nav-menu{display:flex;flex-direction:column;gap:7px}.nav-item{border:0;background:transparent;color:var(--sub);display:flex;align-items:center;gap:12px;padding:11px;border-radius:11px;cursor:pointer;text-align:left;transition:.25s;white-space:nowrap}.nav-item:hover,.nav-item.active{background:linear-gradient(90deg,rgba(77,163,255,.17),rgba(45,212,191,.06));color:#fff;transform:translateX(3px)}.nav-icon{width:28px;height:28px;border-radius:8px;background:#12243a;display:grid;place-items:center;font-size:12px;font-weight:800;flex-shrink:0}.nav-item.active .nav-icon{background:var(--blue);color:#06111e}.nav-spacer{flex:1}.nav-status{padding:11px;border-radius:12px;background:#0c1b2d;border:1px solid var(--border);font-size:10px;color:var(--sub);line-height:1.6;white-space:nowrap}
.app-shell.nav-collapsed .brand-copy,.app-shell.nav-collapsed .nav-text,.app-shell.nav-collapsed .nav-status{opacity:0;pointer-events:none}.app-shell.nav-collapsed .nav-item{justify-content:center}.app-main{min-width:0;padding:0 22px 28px}
.topbar{position:sticky;top:0;z-index:1400;margin:0 -22px;padding:14px 22px;background:rgba(7,17,31,.86);border-bottom:1px solid rgba(38,58,84,.75)}
.headline{font-size:18px;font-weight:800}.headline span{color:var(--cyan)}.header-meta{font-size:11px;color:var(--dim);margin-top:2px}.icon-btn{width:38px;height:38px;border-radius:11px;border:1px solid var(--border);background:var(--card);color:var(--text);cursor:pointer;transition:.22s}.icon-btn:hover{transform:translateY(-2px);border-color:var(--blue);box-shadow:0 8px 24px rgba(0,0,0,.22)}.alert-count{position:absolute;top:-4px;right:-4px;min-width:18px;height:18px;padding:0 4px;border-radius:9px;background:var(--red);display:grid;place-items:center;font-size:9px;font-weight:800}
.controls{margin:18px 0 12px;padding:12px 14px;border:1px solid var(--border);border-radius:14px;background:rgba(14,29,48,.78);box-shadow:0 14px 35px rgba(0,0,0,.14)}
.btn{padding:8px 12px;border-radius:9px;transition:transform .2s,box-shadow .2s,border-color .2s}.btn:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.2)}
.metrics{padding:0 0 12px;background:transparent;grid-template-columns:repeat(5,minmax(125px,1fr));gap:10px}.metric{position:relative;overflow:hidden;min-height:78px;transition:.25s}.metric:after{content:"";position:absolute;width:70px;height:70px;right:-30px;top:-34px;border-radius:50%;background:var(--blue);opacity:.07}.metric:hover{transform:translateY(-3px);border-color:#3d5d82}.mval{font-size:21px;font-variant-numeric:tabular-nums;transition:color .25s}
.dashboard-grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:12px;align-items:start}.panel-card{background:linear-gradient(145deg,rgba(20,37,59,.96),rgba(11,25,42,.96));border:1px solid var(--border);border-radius:16px;box-shadow:0 16px 44px rgba(0,0,0,.18);overflow:hidden;animation:panel-in .55s both;transition:transform .25s,border-color .25s,box-shadow .25s}.panel-card:hover{border-color:#355477;box-shadow:0 20px 50px rgba(0,0,0,.25)}@keyframes panel-in{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.panel-head{min-height:52px;padding:12px 15px;border-bottom:1px solid rgba(38,58,84,.8);display:flex;align-items:center;gap:10px}.panel-title{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.panel-kicker{font-size:10px;color:var(--dim);margin-top:2px}.panel-actions{margin-left:auto;display:flex;gap:6px}.panel-body{padding:14px}.map-card{grid-column:span 7}.map-panel{height:430px;position:relative}.map-panel #map{height:100%;min-height:0}.ops-card{grid-column:span 5;min-height:482px}.analytics-card{grid-column:span 4}.forecast-card{grid-column:span 4}.events-card{grid-column:span 4}.intersections-card{grid-column:span 7}.routes-card{grid-column:span 5}.map-legend{bottom:12px;left:12px;padding:8px 10px;display:grid;grid-template-columns:repeat(2,auto);gap:5px 10px}.map-legend>div:first-child{grid-column:1/-1}.gps-btn{top:12px;right:12px}.map-expand{position:absolute;top:12px;left:56px;z-index:1000}.map-card.fullscreen-map{position:fixed;inset:18px;z-index:2500}.map-card.fullscreen-map .map-panel{height:calc(100vh - 72px)}
.score-wrap{display:grid;grid-template-columns:120px 1fr;gap:16px;align-items:center}.score-ring{--score:78;width:112px;height:112px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--cyan) calc(var(--score)*1%),#1a2c43 0);position:relative;transition:background .7s}.score-ring:before{content:"";position:absolute;inset:10px;border-radius:50%;background:#0e1d30}.score-value{position:relative;font-size:26px;font-weight:900}.score-label{position:relative;font-size:8px;color:var(--dim);text-transform:uppercase}.decision{padding:12px;border-radius:12px;background:#0c1a2b;border:1px solid #23405f}.decision-title{font-size:13px;font-weight:800;color:var(--cyan)}.decision-copy{font-size:11px;line-height:1.6;color:var(--sub);margin-top:5px}.signal-stage{margin-top:14px;display:grid;grid-template-columns:150px 1fr;gap:14px;align-items:center}.digital-junction{height:150px;position:relative;border-radius:14px;overflow:hidden;background:#14202b;box-shadow:inset 0 0 35px #03080f}.road-v{position:absolute;width:54px;height:100%;left:48px;background:#273440}.road-h{position:absolute;height:54px;width:100%;top:48px;background:#273440}.road-v:after,.road-h:after{content:"";position:absolute;border-color:#d7b94e;border-style:dashed;opacity:.55}.road-v:after{left:26px;height:100%;border-width:0 0 0 2px}.road-h:after{top:26px;width:100%;border-width:2px 0 0}.mini-car{position:absolute;width:13px;height:7px;border-radius:3px;background:var(--blue);box-shadow:0 0 8px var(--blue);animation:drive-v 3s linear infinite}.mini-car.c2{background:var(--yellow);box-shadow:0 0 8px var(--yellow);animation:drive-h 3.6s linear infinite}.mini-car.c3{animation-delay:-1.4s;background:var(--cyan)}@keyframes drive-v{from{left:53px;top:-12px}to{left:53px;top:160px}}@keyframes drive-h{from{top:70px;left:-14px}to{top:70px;left:160px}}.signal-orb{width:58px;height:58px;border-radius:50%;display:grid;place-items:center;background:#0b1523;border:5px solid var(--green);box-shadow:0 0 24px rgba(34,197,94,.3);font-size:20px;font-weight:900;transition:.35s}.signal-meta{font-size:11px;color:var(--sub);line-height:1.8;margin-top:7px}
.forecast-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.forecast-item{padding:12px 8px;border-radius:12px;background:#0d1c2e;text-align:center}.forecast-time{font-size:9px;color:var(--dim);text-transform:uppercase}.forecast-value{font-size:22px;font-weight:900;margin:4px 0}.forecast-risk{font-size:9px;padding:3px 7px;border-radius:20px;background:#183148;color:#7dd3fc}.spark-bars{height:88px;display:flex;align-items:end;gap:5px;margin-top:12px}.spark-bars span{flex:1;min-height:8px;border-radius:4px 4px 1px 1px;background:linear-gradient(var(--cyan),var(--blue));transition:height .65s cubic-bezier(.2,.8,.2,1)}
.event-log{max-height:255px;overflow:auto;display:flex;flex-direction:column;gap:9px}.event-row{display:grid;grid-template-columns:8px 1fr auto;gap:9px;align-items:start;padding:9px;border-radius:10px;background:#0c1a2b;animation:slide-in .3s both}.event-dot{width:8px;height:8px;border-radius:50%;margin-top:4px;background:var(--blue)}.event-row.alert .event-dot{background:var(--red)}.event-row.success .event-dot{background:var(--green)}.event-text{font-size:10px;color:var(--sub);line-height:1.5}.event-time{font-size:8px;color:var(--dim)}@keyframes slide-in{from{opacity:0;transform:translateX(12px)}}
.scenario-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.scenario-box{padding:10px;border-radius:10px;background:#0c1a2b}.scenario-box label{font-size:9px;color:var(--dim);display:block;margin-bottom:5px}.scenario-result{margin-top:10px;padding:10px;border-radius:10px;background:linear-gradient(90deg,rgba(77,163,255,.12),rgba(45,212,191,.12));font-size:10px;color:var(--sub);line-height:1.6}
.timeline{display:flex;align-items:center;gap:8px;margin-top:12px}.timeline input{flex:1;accent-color:var(--blue)}.timeline-time{font-size:9px;color:var(--dim);min-width:55px;text-align:right}.route-modes{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.route-mode{padding:11px;border-radius:11px;background:#0c1a2b;border:1px solid var(--border);cursor:pointer;transition:.2s}.route-mode:hover,.route-mode.active{border-color:var(--cyan);transform:translateY(-2px)}.route-name{font-size:11px;font-weight:800}.route-stat{font-size:9px;color:var(--dim);margin-top:4px}.corridor-list{display:flex;gap:5px;flex-wrap:wrap;margin-top:10px}.corridor-chip{padding:4px 8px;border-radius:20px;background:#12304a;color:#8ad9ff;font-size:9px}
.int-list{max-height:360px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.int-item{background:#0c1a2b;transition:.22s}.int-item:hover{transform:translateY(-2px)}
.alert-drawer{position:fixed;right:0;top:0;width:min(390px,92vw);height:100vh;z-index:3000;background:rgba(8,20,35,.98);border-left:1px solid var(--border);box-shadow:-25px 0 65px rgba(0,0,0,.35);transform:translateX(105%);transition:.35s cubic-bezier(.2,.8,.2,1);padding:18px}.alert-drawer.open{transform:none}.drawer-head{display:flex;align-items:center;padding-bottom:14px;border-bottom:1px solid var(--border)}.drawer-list{display:flex;flex-direction:column;gap:9px;margin-top:14px}.drawer-alert{padding:12px;border-radius:12px;background:#111f31;border-left:3px solid var(--yellow);font-size:11px;line-height:1.5}.drawer-alert.critical{border-color:var(--red)}
.toast-stack{position:fixed;right:20px;bottom:20px;z-index:3100;display:flex;flex-direction:column;gap:8px}.toast{padding:11px 14px;border-radius:11px;background:#13243a;border:1px solid #355477;box-shadow:0 12px 32px rgba(0,0,0,.35);font-size:11px;animation:toast-in .3s both}@keyframes toast-in{from{opacity:0;transform:translateY(12px)}}
.footer{margin-top:14px;border:0;color:var(--dim)}
@media(max-width:1180px){.app-shell{grid-template-columns:76px minmax(0,1fr)}.brand-copy,.nav-text,.nav-status{display:none}.nav-item{justify-content:center}.map-card,.ops-card{grid-column:span 12}.analytics-card,.forecast-card,.events-card{grid-column:span 6}.intersections-card,.routes-card{grid-column:span 12}}
@media(max-width:760px){.app-main{padding:0 12px 18px}.topbar{margin:0 -12px;padding:12px}.app-shell{display:block}.nav-rail{position:fixed;left:0;right:0;bottom:0;top:auto;width:auto;height:66px;flex-direction:row;padding:8px;z-index:2400}.brand,.nav-spacer,.nav-status{display:none}.nav-menu{flex-direction:row;width:100%;justify-content:space-around}.nav-item{padding:7px}.metrics{grid-template-columns:repeat(2,minmax(120px,1fr))}.analytics-card,.forecast-card,.events-card,.intersections-card,.routes-card{grid-column:span 12}.map-panel{height:340px}.int-list{grid-template-columns:1fr}.controls{overflow-x:auto;flex-wrap:nowrap}.signal-stage{grid-template-columns:1fr}.digital-junction{width:150px;margin:auto}.app-main{padding-bottom:82px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">⚡ SmartFlow</div>
  <div class="dot" id="dot"></div>
  <div class="conn" id="conn-status">Connecting...</div>
  <div class="topbar-right">
    <span class="tick-lbl" id="tick-display">tick 0</span>
    <span class="tick-lbl" id="weather-display">☀ clear</span>
  </div>
</div>

<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">AI Mode</span>
    <div class="toggle-wrap">
      <span style="font-size:12px;color:var(--dim)">Traditional</span>
      <label class="toggle">
        <input type="checkbox" id="ai-toggle" checked onchange="toggleAI(this.checked)">
        <span class="slider"></span>
      </label>
      <span style="font-size:12px;color:var(--blue)">AI</span>
    </div>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Intersection</span>
    <select id="int-select" onchange="updateActionTarget()">
      <option value="">— select —</option>
    </select>
  </div>
  <div class="ctrl-group">
    <button class="btn danger" onclick="triggerEmergency()">🚨 Emergency</button>
    <button class="btn warn" onclick="addTraffic()">🚗 Add Traffic</button>
    <button class="btn danger" onclick="triggerAccident()">💥 Accident</button>
    <button class="btn purple" onclick="routeAmbulance()">🚑 Ambulance Route</button>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Heatmap</span>
    <label class="toggle">
      <input type="checkbox" id="heatmap-toggle" onchange="toggleHeatmap(this.checked)">
      <span class="slider"></span>
    </label>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Vehicles</span>
    <label class="toggle">
      <input type="checkbox" id="vehicles-toggle" checked onchange="toggleVehicles(this.checked)">
      <span class="slider"></span>
    </label>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Weather</span>
    <div class="weather-chips">
      <div class="wchip active" id="wchip-clear" onclick="setWeather('clear')">☀ Clear</div>
      <div class="wchip" id="wchip-rain" onclick="setWeather('rain')">🌧 Rain</div>
      <div class="wchip" id="wchip-fog" onclick="setWeather('fog')">🌫 Fog</div>
      <div class="wchip" id="wchip-storm" onclick="setWeather('storm')">⛈ Storm</div>
    </div>
  </div>
  <div class="ctrl-group" style="margin-left:auto">
    <button class="btn" onclick="locateUser()">📍 My Location</button>
  </div>
</div>

<div class="metrics">
  <div class="metric"><div class="mlabel">AI Avg Wait</div><div class="mval" id="m-wait">—</div><div class="msub">seconds</div></div>
  <div class="metric"><div class="mlabel">Traditional</div><div class="mval" id="m-trad" style="color:#f59e0b">—</div><div class="msub">seconds (fixed)</div></div>
  <div class="metric"><div class="mlabel">Time Saved</div><div class="mval" id="m-save" style="color:#22c55e">—</div><div class="msub">vs traditional</div></div>
  <div class="metric"><div class="mlabel">Emergencies</div><div class="mval" id="m-emerg">0</div><div class="msub">handled</div></div>
  <div class="metric"><div class="mlabel">Total Ticks</div><div class="mval" id="m-ticks">0</div><div class="msub">steps</div></div>
  <div class="metric"><div class="mlabel">AI Predictor</div><div class="mval" id="m-ai" style="font-size:13px;padding-top:4px">Training...</div><div class="msub">status</div></div>
  <div class="metric"><div class="mlabel">Fuel Idling</div><div class="mval" id="m-fuel">0L</div><div class="msub">estimated</div></div>
  <div class="metric"><div class="mlabel">CO2</div><div class="mval" id="m-co2">0kg</div><div class="msub">estimated</div></div>
  <div class="metric"><div class="mlabel">Humidity</div><div class="mval" id="m-humidity">42%</div><div class="msub">weather</div></div>
  <div class="metric"><div class="mlabel">Temperature</div><div class="mval" id="m-temp">31C</div><div class="msub">Nagpur sim</div></div>
</div>

<div class="main">
  <div class="map-panel">
    <div id="map"></div>
    <div id="weather-overlay" class="weather-overlay"></div>
    <button class="gps-btn" onclick="locateUser()">📍 Locate Me</button>
    <div class="map-legend">
      <div style="font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:4px">Queue Level</div>
      <div class="leg-row"><div class="leg-dot" style="background:#22c55e"></div>Low (&lt;10)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#f59e0b"></div>Medium (10–25)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#ef4444"></div>High (&gt;25)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#a855f7"></div>Emergency</div>
      <div class="leg-row"><div class="leg-dot" style="background:#f97316"></div>Accident</div>
      <div class="leg-row"><div class="leg-dot" style="background:#3b82f6;border:2px solid #fff"></div>Nearest</div>
    </div>
  </div>

  <div class="sidebar">
    <div class="sidebar-section">
      <div class="sec-title">Wait Time: AI vs Traditional</div>
      <div class="chart-wrap">
        <svg id="cmp-chart" viewBox="0 0 300 80" preserveAspectRatio="none"></svg>
      </div>
      <div class="chart-legend">
        <div class="cleg"><div class="cleg-line" style="background:#3b82f6"></div>AI</div>
        <div class="cleg"><div class="cleg-line" style="background:#f59e0b"></div>Traditional</div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sec-title">AI vs Traditional (live)</div>
      <div class="cmp-bars">
        <div class="cmp-row">
          <div class="cmp-lbl">AI Signal</div>
          <div class="cmp-bar-bg"><div class="cmp-bar-fill" id="cmp-ai-bar" style="background:#3b82f6;width:0%"></div></div>
          <div class="cmp-val" id="cmp-ai-val" style="color:#60a5fa">—</div>
        </div>
        <div class="cmp-row">
          <div class="cmp-lbl">Traditional</div>
          <div class="cmp-bar-bg"><div class="cmp-bar-fill" id="cmp-tr-bar" style="background:#f59e0b;width:0%"></div></div>
          <div class="cmp-val" id="cmp-tr-val" style="color:#fbbf24">—</div>
        </div>
      </div>
      <div class="saving-badge" id="saving-badge">Calculating savings...</div>
    </div>

    <!-- Ambulance Route Panel -->
    <div class="sidebar-section">
      <div class="sec-title">🚑 Ambulance Route</div>
      <div class="ambu-panel" id="ambu-panel">
        <div id="ambu-steps"></div>
        <div style="margin-top:8px;font-size:11px;color:#a855f7" id="ambu-eta"></div>
      </div>
      <div id="ambu-idle" style="font-size:12px;color:var(--dim)">Select intersection → click 🚑 Ambulance Route</div>
    </div>

    <div class="sidebar-section" id="nearest-section" style="display:none">
      <div class="sec-title">📍 Nearest Signal</div>
      <div class="nearest-card">
        <div class="nearest-name" id="nearest-name">—</div>
        <div class="nearest-detail" id="nearest-detail">—</div>
      </div>
    </div>

    <div class="sidebar-section" style="flex:1">
      <div class="sec-title">All Intersections</div>
      <div class="int-list" id="int-list"></div>
    </div>
  </div>
</div>

<div class="footer">SmartFlow v3.0 — Nagpur AI Traffic — Animated Vehicles · Heatmap · Accident Sim · Ambulance Routing</div>

<script>
// ══════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════
let allStates = {};
let mapMarkers = {};
let userMarker = null;
let userLatLng = null;
let selectedInt = null;
const aiHistory = [], tradHistory = [];
const MAX_H = 80;

// New feature state
let heatmapLayers = [];       // L.circle objects for heatmap
let heatmapVisible = false;
let vehiclesVisible = true;
let vehicleMarkers = [];      // animated vehicle divIcons
let accidentMarkers = {};     // id -> L.marker
let ambulanceMarker = null;
let ambulanceRouteLine = null;
let accidents = {};           // intId -> {ts, marker}
let stateTimeline = [];
let timelinePaused = false;
let activeRouteMode = 'fastest';
let operationEvents = [];
let previousAlerts = new Set();

function panelCard(id, title, kicker, body) {
  const card = document.createElement('section');
  card.className = 'panel-card';
  card.id = id;
  card.innerHTML = `<div class="panel-head"><div><div class="panel-title">${title}</div><div class="panel-kicker">${kicker}</div></div></div><div class="panel-body">${body}</div>`;
  return card;
}

function buildHybridUI() {
  const topbar = document.querySelector('.topbar');
  const controls = document.querySelector('.controls');
  const metricsEl = document.querySelector('.metrics');
  const oldMain = document.querySelector('.main');
  const mapPanel = document.querySelector('.map-panel');
  const sidebar = document.querySelector('.sidebar');
  const footer = document.querySelector('.footer');
  if (!topbar || !oldMain || !mapPanel || !sidebar) return;

  const shell = document.createElement('div');
  shell.className = 'app-shell'; shell.id = 'app-shell';
  const nav = document.createElement('aside');
  nav.className = 'nav-rail';
  nav.innerHTML = `<div class="brand"><div class="brand-mark">SF</div><div class="brand-copy"><div class="brand-title">SmartFlow</div><div class="brand-sub">Mobility OS</div></div></div>
    <nav class="nav-menu">
      <button class="nav-item active" onclick="scrollPanel('city-map')"><span class="nav-icon">01</span><span class="nav-text">City Overview</span></button>
      <button class="nav-item" onclick="scrollPanel('signal-ops')"><span class="nav-icon">02</span><span class="nav-text">Signal AI</span></button>
      <button class="nav-item" onclick="scrollPanel('forecast-panel')"><span class="nav-icon">03</span><span class="nav-text">Forecast</span></button>
      <button class="nav-item" onclick="scrollPanel('routes-panel')"><span class="nav-icon">04</span><span class="nav-text">Routes</span></button>
      <button class="nav-item" onclick="scrollPanel('event-panel')"><span class="nav-icon">05</span><span class="nav-text">Events</span></button>
      <button class="nav-item" onclick="scrollPanel('intersection-panel')"><span class="nav-icon">06</span><span class="nav-text">Signals</span></button>
    </nav><div class="nav-spacer"></div><div class="nav-status"><span class="dot" style="display:inline-block;margin-right:6px"></span>Network operational<br>20 signals connected</div>`;
  const appMain = document.createElement('main'); appMain.className = 'app-main';
  document.body.insertBefore(shell, topbar); shell.append(nav, appMain);
  [topbar, controls, metricsEl].forEach(el => appMain.appendChild(el));
  topbar.insertAdjacentHTML('afterbegin', `<button class="icon-btn" onclick="toggleNav()">&#9776;</button><div><div class="headline">Nagpur <span>Mobility Command</span></div><div class="header-meta">Adaptive city traffic orchestration and digital twin</div></div>`);
  topbar.insertAdjacentHTML('beforeend', `<button class="icon-btn" style="position:relative" onclick="toggleAlerts()">AL<span class="alert-count" id="alert-count">0</span></button>`);

  const grid = document.createElement('div'); grid.className = 'dashboard-grid';
  appMain.appendChild(grid);
  const mapCard = document.createElement('section'); mapCard.className = 'panel-card map-card'; mapCard.id = 'city-map';
  mapCard.innerHTML = `<div class="panel-head"><div><div class="panel-title">Live City Map</div><div class="panel-kicker">OpenStreetMap digital traffic layer</div></div><div class="panel-actions"><button class="btn" onclick="toggleHeatmapFromButton()">Heat</button><button class="btn" onclick="toggleMapSize()">Expand</button></div></div>`;
  mapCard.appendChild(mapPanel); grid.appendChild(mapCard);

  const ops = panelCard('signal-ops','AI Signal Cockpit','Decision intelligence and intersection twin',`<div class="score-wrap"><div class="score-ring" id="score-ring"><div style="text-align:center"><div class="score-value" id="city-score">--</div><div class="score-label">City health</div></div></div><div class="decision"><div class="decision-title" id="decision-title">Waiting for live traffic</div><div class="decision-copy" id="decision-copy">SmartFlow explains timing decisions using queue balance, prediction, weather and priority demand.</div></div></div>
    <div class="signal-stage"><div class="digital-junction"><div class="road-v"></div><div class="road-h"></div><div class="mini-car"></div><div class="mini-car c2"></div><div class="mini-car c3"></div></div><div><div class="signal-orb" id="signal-orb">--</div><div class="signal-meta" id="signal-meta">Select an intersection to inspect its digital twin.</div></div></div>
    <div class="scenario-row"><div class="scenario-box"><label>Traffic multiplier</label><input id="whatif-traffic" type="range" min="1" max="5" step=".5" value="2" oninput="runWhatIf()" style="width:100%"></div><div class="scenario-box"><label>Scenario weather</label><select id="whatif-weather" onchange="runWhatIf()" style="width:100%"><option>clear</option><option>rain</option><option>fog</option><option>storm</option></select></div></div><div class="scenario-result" id="whatif-result">Adjust demand and weather to preview operational impact.</div>`);
  ops.classList.add('ops-card'); grid.appendChild(ops);

  const analytics = document.createElement('section'); analytics.className='panel-card analytics-card'; analytics.innerHTML='<div class="panel-head"><div><div class="panel-title">AI vs Traditional</div><div class="panel-kicker">Live control performance and replay</div></div></div><div class="panel-body" id="analytics-slot"></div>';
  const sideChildren = Array.from(sidebar.children);
  sideChildren.slice(0,2).forEach(el=>analytics.querySelector('#analytics-slot').appendChild(el));
  analytics.querySelector('#analytics-slot').insertAdjacentHTML('beforeend',`<div class="timeline"><button class="btn" id="timeline-toggle" onclick="toggleTimeline()">Pause</button><input id="timeline-range" type="range" min="0" max="0" value="0" oninput="scrubTimeline(this.value)"><span class="timeline-time" id="timeline-time">LIVE</span></div>`);
  grid.appendChild(analytics);

  const forecast = panelCard('forecast-panel','AI Forecast','5, 15 and 30 minute congestion outlook',`<div class="forecast-grid"><div class="forecast-item"><div class="forecast-time">5 min</div><div class="forecast-value" id="fc-5">--</div><span class="forecast-risk" id="risk-5">learning</span></div><div class="forecast-item"><div class="forecast-time">15 min</div><div class="forecast-value" id="fc-15">--</div><span class="forecast-risk" id="risk-15">learning</span></div><div class="forecast-item"><div class="forecast-time">30 min</div><div class="forecast-value" id="fc-30">--</div><span class="forecast-risk" id="risk-30">learning</span></div></div><div class="spark-bars" id="forecast-bars"></div>`); forecast.classList.add('forecast-card'); grid.appendChild(forecast);

  const events = panelCard('event-panel','Operations Log','AI actions, incidents and manual overrides',`<div class="event-log" id="event-log"><div class="event-row"><span class="event-dot"></span><span class="event-text">Operations timeline initialized.</span><span class="event-time">now</span></div></div>`); events.classList.add('events-card'); events.querySelector('.panel-head').insertAdjacentHTML('beforeend','<button class="btn" style="margin-left:auto" onclick="clearEventLog()">Clear</button>'); grid.appendChild(events);

  const intersectionCard = document.createElement('section'); intersectionCard.className='panel-card intersections-card'; intersectionCard.id='intersection-panel'; intersectionCard.innerHTML='<div class="panel-head"><div><div class="panel-title">Signal Network</div><div class="panel-kicker">20 connected Nagpur intersections</div></div><button class="btn" style="margin-left:auto" onclick="sortIntersections()">Sort congestion</button></div><div class="panel-body" id="intersection-slot"></div>';
  const intSection = sideChildren[sideChildren.length-1]; intersectionCard.querySelector('#intersection-slot').appendChild(intSection); grid.appendChild(intersectionCard);

  const routes = panelCard('routes-panel','Priority Routing','Emergency, low-carbon and fastest corridors',`<div class="route-modes"><div class="route-mode active" onclick="selectRouteMode('fastest',this)"><div class="route-name">Fastest</div><div class="route-stat" id="route-fast">Select origin</div></div><div class="route-mode" onclick="selectRouteMode('green',this)"><div class="route-name">Low emission</div><div class="route-stat" id="route-green">Select origin</div></div><div class="route-mode" onclick="selectRouteMode('resilient',this)"><div class="route-name">Low congestion</div><div class="route-stat" id="route-safe">Select origin</div></div></div><div class="corridor-list" id="corridor-list"><span class="corridor-chip">Select a signal to build corridor</span></div><div style="display:flex;gap:8px;margin:12px 0"><button class="btn purple" onclick="routeAmbulance()">Route Ambulance</button><button class="btn success" onclick="buildGreenWave()">Build Green Wave</button></div><div id="route-slots"></div>`); routes.classList.add('routes-card');
  sideChildren.slice(2,-1).forEach(el=>routes.querySelector('#route-slots').appendChild(el)); grid.appendChild(routes);

  appMain.appendChild(footer); oldMain.remove(); sidebar.remove();
  document.body.insertAdjacentHTML('beforeend',`<aside class="alert-drawer" id="alert-drawer"><div class="drawer-head"><div><div class="panel-title">City Alerts</div><div class="panel-kicker">Prioritized operational exceptions</div></div><button class="icon-btn" style="margin-left:auto" onclick="toggleAlerts()">X</button></div><div class="drawer-list" id="drawer-list"><div class="drawer-alert">No active exceptions.</div></div></aside><div class="toast-stack" id="toast-stack"></div>`);
}

buildHybridUI();

// ══════════════════════════════════════════════════════════════════
// MAP INIT
// ══════════════════════════════════════════════════════════════════
const map = L.map('map', {zoomControl: true}).setView([21.1458, 79.0882], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution:'&copy; OpenStreetMap', maxZoom:19
}).addTo(map);

// ══════════════════════════════════════════════════════════════════
// INTERSECTION MARKERS + SIGNAL POPUP
// ══════════════════════════════════════════════════════════════════
function markerColor(s) {
  if (s.emergency) return '#a855f7';
  if (s.accident || accidents[s.id]) return '#f97316';
  if (s.total_queue > 25) return '#ef4444';
  if (s.total_queue > 10) return '#f59e0b';
  return '#22c55e';
}

function makeIcon(color, isNearest, isAccident) {
  const size = isNearest ? 22 : 16;
  const border = isNearest ? '#ffffff' : 'transparent';
  const inner = isAccident
    ? `<div style="font-size:14px;line-height:${size}px;text-align:center" class="accident-icon">💥</div>`
    : `<div style="width:${size}px;height:${size}px;border-radius:50%;background:${color};border:3px solid ${border};box-shadow:0 0 8px ${color}88"></div>`;
  return L.divIcon({
    className:'',
    html: inner,
    iconSize:[size, size],
    iconAnchor:[size/2, size/2]
  });
}

function signalPopupHTML(s) {
  const isGreen = s.phase === 'NS_GREEN' ? ['NS','EW'] : ['EW','NS'];
  const phaseBar = `
    <div style="display:flex;gap:4px;margin:6px 0">
      <div style="flex:1;height:6px;border-radius:3px;background:${isGreen[0]==='NS'?'#22c55e':'#ef4444'}"></div>
      <div style="flex:1;height:6px;border-radius:3px;background:${isGreen[0]==='EW'?'#22c55e':'#ef4444'}"></div>
    </div>`;
  const pred = s.prediction || {};
  return `
    <div class="sig-popup-title">📍 ${s.name}</div>
    ${phaseBar}
    <div class="sig-popup-row"><span>Phase</span><span style="color:#60a5fa">${s.phase}</span></div>
    <div class="sig-popup-row"><span>Countdown</span><span class="countdown-pill">${Math.ceil(s.countdown||0)}s</span></div>
    <div class="sig-popup-row"><span>Queue</span><span style="color:${s.total_queue>25?'#ef4444':s.total_queue>10?'#f59e0b':'#22c55e'}">${s.total_queue} vehicles</span></div>
    <div class="sig-popup-row"><span>Avg Wait</span><span>${s.avg_wait}s</span></div>
    <div class="sig-popup-row"><span>Prediction</span><span style="color:${pred.congestion_level==='high'?'#ef4444':pred.congestion_level==='medium'?'#f59e0b':'#22c55e'}">${pred.congestion_level||'unknown'}</span></div>
    <div class="sig-popup-row"><span>Pedestrians</span><span>${s.pedestrian_waiting||0}${s.pedestrian_crossing?' crossing':''}</span></div>
    <div class="sig-popup-row"><span>Fuel / CO2</span><span>${s.fuel_litres||0}L / ${s.co2_kg||0}kg</span></div>
    ${s.emergency?'<div style="color:#a855f7;font-weight:700;margin-top:6px">🚨 EMERGENCY ACTIVE</div>':''}
    ${accidents[s.id]?'<div style="color:#f97316;font-weight:700;margin-top:6px">💥 ACCIDENT AT THIS INTERSECTION</div>':''}
    <div class="sig-popup-actions">
      <button class="sig-popup-btn" style="background:#7f1d1d;border-color:#ef4444;color:#fca5a5"
        onclick="selectIntersection('${s.id}');triggerEmergency();map.closePopup()">🚨 Emergency</button>
      <button class="sig-popup-btn" style="background:#713f12;border-color:#f59e0b;color:#fde68a"
        onclick="selectIntersection('${s.id}');addTraffic();map.closePopup()">🚗 Traffic</button>
      <button class="sig-popup-btn" style="background:#431407;border-color:#f97316;color:#fed7aa"
        onclick="selectIntersection('${s.id}');triggerAccident();map.closePopup()">💥 Accident</button>
      <button class="sig-popup-btn" style="background:#4c1d95;border-color:#a855f7;color:#e9d5ff"
        onclick="selectIntersection('${s.id}');routeAmbulance();map.closePopup()">🚑 Ambu</button>
    </div>`;
}

function initMarkers(states) {
  for (const [id, s] of Object.entries(states)) {
    if (mapMarkers[id]) continue;
    const m = L.marker([s.lat, s.lng], {icon: makeIcon(markerColor(s), false, false)})
      .addTo(map)
      .bindPopup('', {maxWidth: 260});
    m.on('click', () => {
      m.setPopupContent(signalPopupHTML(allStates[id] || s));
    });
    mapMarkers[id] = m;
  }
}

function nearestId(lat, lng) {
  let best = null, bestDist = Infinity;
  for (const [id, s] of Object.entries(allStates)) {
    const d = Math.hypot(s.lat - lat, s.lng - lng);
    if (d < bestDist) { bestDist = d; best = id; }
  }
  return best;
}

function updateMarkers(states) {
  const nid = userLatLng ? nearestId(userLatLng[0], userLatLng[1]) : null;
  for (const [id, s] of Object.entries(states)) {
    const m = mapMarkers[id];
    if (!m) continue;
    m.setIcon(makeIcon(markerColor(s), id === nid, !!accidents[id]));
  }
}

// ══════════════════════════════════════════════════════════════════
// ANIMATED VEHICLES
// Each intersection spawns small emoji cars that travel toward it
// They stop when phase is red, move when green.
// ══════════════════════════════════════════════════════════════════
const VEHICLE_EMOJIS = ['🚗','🚕','🚙','🚌','🏍️','🚎','🚐'];
const DIRECTIONS_OFFSET = {
  N: [-0.0018, 0],
  S: [0.0018, 0],
  E: [0, 0.0025],
  W: [0, -0.0025]
};

// Each vehicle: {marker, intId, dir, progress 0..1, speed, active}
let vehiclePool = [];
let vehicleAnimationStarted = false;

function initVehicles(states) {
  if (vehiclePool.length > 0) return; // already initialized
  for (const [id, s] of Object.entries(states)) {
    // 2 vehicles per direction per intersection
    for (const dir of ['N','S','E','W']) {
      for (let k = 0; k < 2; k++) {
        const offset = DIRECTIONS_OFFSET[dir];
        const startLat = s.lat + offset[0];
        const startLng = s.lng + offset[1];
        const emoji = VEHICLE_EMOJIS[Math.floor(Math.random()*VEHICLE_EMOJIS.length)];
        const icon = L.divIcon({
          className:'',
          html:`<div style="font-size:14px;line-height:1">${emoji}</div>`,
          iconSize:[18,18], iconAnchor:[9,9]
        });
        const progress = Math.random(); // stagger start positions
        const marker = L.marker([startLat, startLng], {icon, zIndexOffset: -100}).addTo(map);
        vehiclePool.push({
          marker, intId: id, dir,
          progress: progress,
          speed: 0.003 + Math.random() * 0.004,
          active: true,
          startLat, startLng
        });
      }
    }
  }
}

function animateVehicles() {
  vehicleAnimationStarted = true;
  for (const v of vehiclePool) {
    const s = allStates[v.intId];
    if (!s) continue;

    // Check if this direction has green light
    const nsGreen = s.phase === 'NS_GREEN';
    const ewGreen = s.phase === 'EW_GREEN';
    const dirIsGreen = (v.dir === 'N' || v.dir === 'S') ? nsGreen : ewGreen;

    // Vehicles slow/stop on red, move on green
    // Also stop if accident at intersection
    const hasAccident = !!accidents[v.intId] || !!s.accident;
    const queueLoad = Math.min((s.lanes[v.dir]||{queue:0}).queue / 20, 1);

    let speedMult = dirIsGreen ? (1 - queueLoad * 0.7) : 0.0;
    if (hasAccident) speedMult = 0;

    v.progress += v.speed * speedMult;

    // When vehicle reaches intersection, reset to start
    if (v.progress >= 1) {
      v.progress = 0;
      // Random re-emit from a slightly varied start
      const offset = DIRECTIONS_OFFSET[v.dir];
      const jitter = (Math.random()-0.5)*0.0005;
      v.startLat = s.lat + offset[0] + jitter;
      v.startLng = s.lng + offset[1] + jitter;
    }

    // Interpolate position: from start toward intersection center
    const lat = v.startLat + (s.lat - v.startLat) * v.progress;
    const lng = v.startLng + (s.lng - v.startLng) * v.progress;
    v.marker.setLatLng([lat, lng]);
    v.marker.setOpacity(vehiclesVisible ? 1 : 0);
  }
  requestAnimationFrame(animateVehicles);
}

function toggleVehicles(on) {
  vehiclesVisible = on;
  for (const v of vehiclePool) {
    v.marker.setOpacity(on ? 1 : 0);
  }
  if (on && !vehicleAnimationStarted) requestAnimationFrame(animateVehicles);
}

// ══════════════════════════════════════════════════════════════════
// CONGESTION HEATMAP
// Circles centered on each intersection, radius+opacity = queue
// ══════════════════════════════════════════════════════════════════
function renderHeatmap(states) {
  // Remove old circles
  heatmapLayers.forEach(l => map.removeLayer(l));
  heatmapLayers = [];
  if (!heatmapVisible) return;

  for (const [id, s] of Object.entries(states)) {
    const predicted = Number((s.prediction||{}).predicted_queue ?? s.total_queue);
    const intensity = Math.min(Math.max(s.total_queue, predicted) / 40, 1);
    const radius = 120 + intensity * 350; // metres
    const r = Math.round(intensity * 255);
    const g = Math.round((1 - intensity) * 180);
    const color = `rgb(${r},${g},0)`;
    const circle = L.circle([s.lat, s.lng], {
      radius,
      color: 'transparent',
      fillColor: color,
      fillOpacity: 0.18 + intensity * 0.32,
      interactive: false
    }).addTo(map);
    heatmapLayers.push(circle);
  }
}

function toggleHeatmap(on) {
  heatmapVisible = on;
  if (on) renderHeatmap(allStates);
  if (!on) {
    heatmapLayers.forEach(l => map.removeLayer(l));
    heatmapLayers = [];
  }
}

// ══════════════════════════════════════════════════════════════════
// ACCIDENT SIMULATION
// ══════════════════════════════════════════════════════════════════
function triggerAccident() {
  const iid = selectedInt;
  if (!iid) { alert('Select an intersection first'); return; }
  if (accidents[iid]) {
    clearAccident(iid);
    return;
  }
  // Add accident marker
  const s = allStates[iid];
  if (!s) return;
  const icon = L.divIcon({
    className:'',
    html:'<div class="accident-icon" style="font-size:22px">💥</div>',
    iconSize:[28,28], iconAnchor:[14,14]
  });
  const m = L.marker([s.lat + 0.0003, s.lng + 0.0003], {icon, zIndexOffset: 500})
    .addTo(map)
    .bindPopup(`<b>💥 Accident at ${s.name}</b><br>Traffic blocked. Vehicles stopped.<br>
      <button onclick="selectIntersection('${iid}');clearAccident('${iid}')" 
        style="margin-top:6px;padding:3px 10px;border-radius:5px;border:1px solid #f97316;background:#431407;color:#fed7aa;cursor:pointer">
        ✓ Clear Accident</button>`);
  accidents[iid] = { marker: m, ts: Date.now() };
  m.openPopup();

  // Spike traffic at this intersection
  fetch('/api/accident', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid, active: true})});
  logEvent(`Accident simulation started at ${s.name}.`, 'alert');
  toast(`Accident response active at ${s.name}.`);

  // Also trigger emergency
  fetch('/api/emergency', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid})});
}

function clearAccident(iid) {
  if (accidents[iid]) {
    if (accidents[iid].marker) map.removeLayer(accidents[iid].marker);
    delete accidents[iid];
    map.closePopup();
  }
  fetch('/api/accident', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid, active: false})});
  logEvent(`Accident cleared at ${allStates[iid]?.name||iid}.`, 'success');
}

function syncAccidents(states) {
  for (const [id, s] of Object.entries(states)) {
    if (s.accident && !accidents[id]) {
      const icon = L.divIcon({
        className:'', html:'<div class="accident-icon" style="font-size:20px">!</div>',
        iconSize:[26,26], iconAnchor:[13,13]
      });
      const marker = L.marker([s.lat + 0.0003, s.lng + 0.0003], {icon, zIndexOffset:500})
        .addTo(map).bindPopup(`<b>Accident at ${s.name}</b><br>Traffic capacity is blocked.`);
      accidents[id] = {marker, ts:Date.now()};
    } else if (!s.accident && accidents[id]) {
      if (accidents[id].marker) map.removeLayer(accidents[id].marker);
      delete accidents[id];
    }
  }
}

// ══════════════════════════════════════════════════════════════════
// AMBULANCE ROUTE OPTIMIZATION
// Finds path from selected intersection to the emergency intersection
// Prefers low-queue intersections as waypoints (Dijkstra-lite)
// ══════════════════════════════════════════════════════════════════
function routeAmbulance() {
  const startId = selectedInt;
  if (!startId) { alert('Select a START intersection first (origin)'); return; }

  // Find intersection with highest emergency / queue as destination
  let destId = null, maxScore = -1;
  for (const [id, s] of Object.entries(allStates)) {
    if (id === startId) continue;
    const score = (s.emergency ? 1000 : 0) + s.total_queue;
    if (score > maxScore) { maxScore = score; destId = id; }
  }
  if (!destId) return;

  // Build adjacency: each intersection connects to 3 nearest neighbors
  const ids = Object.keys(allStates);
  const adj = {};
  for (const id of ids) {
    const s = allStates[id];
    const others = ids.filter(x => x !== id).map(x => ({
      id: x,
      dist: Math.hypot(allStates[x].lat - s.lat, allStates[x].lng - s.lng)
    })).sort((a,b)=>a.dist-b.dist).slice(0,4);
    adj[id] = others;
  }

  // Dijkstra — cost = geographic distance * (1 + queue/10)
  const dist = {}, prev = {};
  ids.forEach(id => { dist[id] = Infinity; });
  dist[startId] = 0;
  const unvisited = new Set(ids);

  while (unvisited.size > 0) {
    let u = null;
    for (const id of unvisited) {
      if (u === null || dist[id] < dist[u]) u = id;
    }
    if (u === destId || dist[u] === Infinity) break;
    unvisited.delete(u);
    for (const nb of (adj[u]||[])) {
      const queuePenalty = 1 + (allStates[nb.id]||{total_queue:0}).total_queue / 10;
      const alt = dist[u] + nb.dist * queuePenalty;
      if (alt < dist[nb.id]) { dist[nb.id] = alt; prev[nb.id] = u; }
    }
  }

  // Reconstruct path
  const path = [];
  let cur = destId;
  while (cur) { path.unshift(cur); cur = prev[cur]; }
  if (path[0] !== startId) path.unshift(startId);

  // Draw route on map
  if (ambulanceRouteLine) map.removeLayer(ambulanceRouteLine);
  if (ambulanceMarker) map.removeLayer(ambulanceMarker);

  const latlngs = path.map(id => [allStates[id].lat, allStates[id].lng]);
  ambulanceRouteLine = L.polyline(latlngs, {
    color:'#a855f7', weight:5, opacity:0.85, dashArray:'10,6'
  }).addTo(map);

  // Ambulance emoji marker animating along route
  const ambuIcon = L.divIcon({
    className:'',
    html:'<div style="font-size:22px;filter:drop-shadow(0 0 6px #a855f7)">🚑</div>',
    iconSize:[28,28], iconAnchor:[14,14]
  });
  ambulanceMarker = L.marker(latlngs[0], {icon: ambuIcon, zIndexOffset:1000}).addTo(map);
  animateAmbulance(latlngs, 0);
  map.fitBounds(ambulanceRouteLine.getBounds(), {padding:[40,40]});

  // Show route in sidebar
  const panel = document.getElementById('ambu-panel');
  const idle = document.getElementById('ambu-idle');
  panel.classList.add('active');
  idle.style.display = 'none';
  document.getElementById('ambu-steps').innerHTML = path.map((id, i) => {
    const name = (allStates[id]||{}).name || id;
    const q = (allStates[id]||{total_queue:0}).total_queue;
    const icon = i===0?'🔵':i===path.length-1?'🔴':'⚪';
    return `<div class="ambu-step"><div class="dot-step"></div>${icon} ${name} <span style="color:#64748b;font-size:10px">(q:${q})</span></div>`;
  }).join('');
  const etaSecs = path.length * 12;
  document.getElementById('ambu-eta').textContent = `ETA: ~${etaSecs}s via ${path.length} intersections — low-congestion route`;
  logEvent(`Ambulance route activated through ${path.length} intersections.`, 'success');
  toast('Emergency green-wave route dispatched.');

  // Green-wave: pre-clear all signals on route
  for (const id of path) {
    fetch('/api/phase/'+id, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({duration: 10})});
  }
}

let ambuAnimFrame = null;
function animateAmbulance(latlngs, segIdx) {
  if (segIdx >= latlngs.length - 1) {
    // Reached destination — pulse
    return;
  }
  const start = latlngs[segIdx];
  const end = latlngs[segIdx+1];
  let t = 0;
  const STEPS = 60;
  function step() {
    t++;
    if (!ambulanceMarker) return;
    const lat = start[0] + (end[0]-start[0]) * (t/STEPS);
    const lng = start[1] + (end[1]-start[1]) * (t/STEPS);
    ambulanceMarker.setLatLng([lat, lng]);
    if (t < STEPS) {
      ambuAnimFrame = requestAnimationFrame(step);
    } else {
      animateAmbulance(latlngs, segIdx+1);
    }
  }
  ambuAnimFrame = requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════════
// GPS + NEAREST
// ══════════════════════════════════════════════════════════════════
function locateUser() {
  if (!navigator.geolocation) { alert('GPS not available in browser'); return; }
  navigator.geolocation.getCurrentPosition(pos => {
    const {latitude: lat, longitude: lng} = pos.coords;
    userLatLng = [lat, lng];
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.marker([lat, lng], {
      icon: L.divIcon({
        className:'',
        html:'<div style="width:18px;height:18px;border-radius:50%;background:#3b82f6;border:3px solid #fff;box-shadow:0 0 12px #3b82f688"></div>',
        iconSize:[18,18], iconAnchor:[9,9]
      })
    }).addTo(map).bindPopup('📍 You are here').openPopup();
    map.setView([lat, lng], 15);
    showNearest(lat, lng);
  }, () => {
    userLatLng = [21.1458, 79.0882];
    showNearest(21.1458, 79.0882);
  });
}

function showNearest(lat, lng) {
  const nid = nearestId(lat, lng);
  if (!nid || !allStates[nid]) return;
  const s = allStates[nid];
  document.getElementById('nearest-section').style.display = '';
  document.getElementById('nearest-name').textContent = `${s.name} (${nid})`;
  document.getElementById('nearest-detail').innerHTML =
    `Queue: <b>${s.total_queue}</b> vehicles &nbsp;|&nbsp; Wait: <b>${s.avg_wait}s</b><br>
     Phase: <b>${s.phase}</b> &nbsp;|&nbsp; Duration: <b>${s.phase_duration}s</b><br>
     Congestion: <b>${(s.prediction||{}).congestion_level||'unknown'}</b>`;
  updateMarkers(allStates);
}

// ══════════════════════════════════════════════════════════════════
// EXISTING CONTROLS (unchanged)
// ══════════════════════════════════════════════════════════════════
function toggleAI(on) {
  fetch('/api/ai_mode', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled: on})});
  logEvent(`Control mode changed to ${on?'AI adaptive':'traditional fixed-time'}.`, 'info');
  toast(`${on?'AI adaptive':'Traditional'} control enabled.`);
}
function updateActionTarget() {
  selectedInt = document.getElementById('int-select').value || null;
}
function triggerEmergency() {
  const iid = selectedInt;
  if (!iid) { alert('Select an intersection first'); return; }
  fetch('/api/emergency', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid})});
  logEvent(`Emergency priority requested at ${allStates[iid]?.name||iid}.`, 'alert');
}
function addTraffic() {
  const iid = selectedInt;
  if (!iid) { alert('Select an intersection first'); return; }
  fetch('/api/add_traffic', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid, multiplier: 3.0})});
  logEvent(`Traffic surge injected at ${allStates[iid]?.name||iid}.`, 'info');
}
function setWeather(cond) {
  fetch('/api/weather', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({condition: cond})});
  updateWeatherUI(cond);
  ['clear','rain','fog','storm'].forEach(c =>
    document.getElementById('wchip-'+c).classList.toggle('active', c === cond));
  const icons = {clear:'☀',rain:'🌧',fog:'🌫',storm:'⛈'};
  document.getElementById('weather-display').textContent = (icons[cond]||'')+' '+cond;
  logEvent(`City weather scenario changed to ${cond}.`, cond==='storm'?'alert':'info');
  toast(`${cond} mobility profile applied.`);
}
function updateWeatherUI(cond) {
  const overlay = document.getElementById('weather-overlay');
  overlay.className = 'weather-overlay' + (cond === 'clear' ? '' : ' ' + cond);
  ['clear','rain','fog','storm'].forEach(c => {
    const chip = document.getElementById('wchip-'+c);
    if (chip) chip.classList.toggle('active', c === cond);
  });
}
function selectIntersection(id) {
  selectedInt = id;
  document.getElementById('int-select').value = id;
  document.querySelectorAll('.int-item').forEach(el =>
    el.classList.toggle('selected', el.dataset.id === id));
  const s = allStates[id];
  if (s) {
    map.flyTo([s.lat, s.lng], 15, {duration:.8});
    updateOperationalUI(allStates, window.latestMetrics || {});
    updateRouteComparison();
    logEvent(`Operator selected ${s.name} for detailed monitoring.`, 'info');
  }
}

function toggleNav() {
  document.getElementById('app-shell').classList.toggle('nav-collapsed');
  setTimeout(()=>map.invalidateSize(), 380);
}
function scrollPanel(id) {
  document.getElementById(id)?.scrollIntoView({behavior:'smooth', block:'start'});
  document.querySelectorAll('.nav-item').forEach(el=>el.classList.remove('active'));
  event?.currentTarget?.classList.add('active');
}
function toggleAlerts() { document.getElementById('alert-drawer').classList.toggle('open'); }
function toggleMapSize() {
  document.getElementById('city-map').classList.toggle('fullscreen-map');
  setTimeout(()=>map.invalidateSize(), 280);
}
function toggleHeatmapFromButton() {
  const checkbox = document.getElementById('heatmap-toggle');
  checkbox.checked = !checkbox.checked; toggleHeatmap(checkbox.checked);
}
function toast(message) {
  const stack = document.getElementById('toast-stack');
  if (!stack) return;
  const el = document.createElement('div'); el.className='toast'; el.textContent=message;
  stack.appendChild(el); setTimeout(()=>el.remove(), 3200);
}
function logEvent(message, type='info') {
  const item = {message,type,time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})};
  operationEvents.unshift(item); operationEvents = operationEvents.slice(0,40);
  const log = document.getElementById('event-log'); if (!log) return;
  log.innerHTML = operationEvents.map(e=>`<div class="event-row ${e.type}"><span class="event-dot"></span><span class="event-text">${e.message}</span><span class="event-time">${e.time}</span></div>`).join('');
}
function clearEventLog() { operationEvents=[]; document.getElementById('event-log').innerHTML=''; }

function focusedState(states) {
  if (selectedInt && states[selectedInt]) return states[selectedInt];
  return Object.values(states).sort((a,b)=>b.total_queue-a.total_queue)[0];
}
function congestionRisk(value) { return value < 20 ? 'low' : value < 45 ? 'medium' : 'high'; }
function updateOperationalUI(states, m) {
  const values = Object.values(states); if (!values.length) return;
  const focus = focusedState(states);
  const avgQueue = values.reduce((n,s)=>n+s.total_queue,0)/values.length;
  const incidents = values.filter(s=>s.accident||s.incident).length;
  const health = Math.max(12, Math.min(98, Math.round(100 - avgQueue*.65 - (m.avg_wait||0)*.15 - incidents*5)));
  document.getElementById('city-score').textContent=health;
  document.getElementById('score-ring').style.setProperty('--score',health);
  const ns=(focus.lanes.N.queue+focus.lanes.S.queue), ew=(focus.lanes.E.queue+focus.lanes.W.queue);
  let reason = focus.emergency ? 'Emergency priority override is cycling this signal for a green-wave response.' : focus.accident ? 'Accident capacity loss detected; the controller is containing queue spillback.' : focus.pedestrian_crossing ? 'Pedestrian demand threshold reached; protected crossing phase is active.' : `${ns>ew?'North-south':'East-west'} demand is ${Math.abs(ns-ew)} vehicles higher, so AI selected a ${focus.phase_duration}s adaptive phase.`;
  document.getElementById('decision-title').textContent=`${focus.name}: ${focus.phase.replace('_',' ')}`;
  document.getElementById('decision-copy').textContent=reason;
  const orb=document.getElementById('signal-orb'); orb.textContent=Math.ceil(focus.countdown||0); orb.style.borderColor=focus.pedestrian_crossing?'#2dd4bf':focus.accident?'#ef4444':'#22c55e';
  document.getElementById('signal-meta').innerHTML=`<b>${focus.name}</b><br>${focus.phase} | ${focus.total_queue} vehicles<br>${focus.pedestrian_waiting||0} pedestrians waiting`;

  const pred=Number((focus.prediction||{}).predicted_queue||focus.total_queue);
  const horizons=[pred, pred*1.16 + avgQueue*.1, pred*1.34 + avgQueue*.18];
  ['5','15','30'].forEach((key,i)=>{const val=Math.round(horizons[i]);document.getElementById('fc-'+key).textContent=val;document.getElementById('risk-'+key).textContent=congestionRisk(val);});
  const bars=Array.from({length:12},(_,i)=>Math.max(8,Math.min(100,(pred*(.68+i*.055)+Math.sin(i)*5))));
  document.getElementById('forecast-bars').innerHTML=bars.map(v=>`<span style="height:${v}%"></span>`).join('');
  updateAlerts(states); runWhatIf(); updateRouteComparison();
}

function updateAlerts(states) {
  const alerts=[];
  Object.values(states).forEach(s=>{
    if(s.accident) alerts.push({id:'acc-'+s.id,critical:true,text:`Accident blocks capacity at ${s.name}.`});
    else if(s.emergency) alerts.push({id:'em-'+s.id,critical:true,text:`Emergency priority active at ${s.name}.`});
    else if(s.total_queue>65) alerts.push({id:'q-'+s.id,critical:false,text:`Severe congestion at ${s.name}: ${s.total_queue} vehicles.`});
  });
  document.getElementById('alert-count').textContent=alerts.length;
  document.getElementById('drawer-list').innerHTML=alerts.length?alerts.map(a=>`<div class="drawer-alert ${a.critical?'critical':''}">${a.text}</div>`).join(''):'<div class="drawer-alert">No active exceptions.</div>';
  const current=new Set(alerts.map(a=>a.id));
  alerts.filter(a=>!previousAlerts.has(a.id)).forEach(a=>{toast(a.text);logEvent(a.text,a.critical?'alert':'info');});
  previousAlerts=current;
}

function runWhatIf() {
  const focus=focusedState(allStates); if(!focus)return;
  const mult=Number(document.getElementById('whatif-traffic')?.value||1);
  const weather=document.getElementById('whatif-weather')?.value||'clear';
  const penalty={clear:1,rain:1.22,fog:1.34,storm:1.62}[weather];
  const projected=Math.round(focus.total_queue*mult*penalty);
  const wait=Math.round(focus.avg_wait*mult*penalty);
  const recommendation=projected>70?'activate corridor control and emergency diversion':projected>35?'extend the dominant green phase':'retain adaptive timing';
  document.getElementById('whatif-result').innerHTML=`Projected <b>${projected} vehicles</b> and <b>${wait}s wait</b> at ${focus.name}. Recommendation: ${recommendation}.`;
}

function toggleTimeline() {
  timelinePaused=!timelinePaused;
  document.getElementById('timeline-toggle').textContent=timelinePaused?'Resume':'Pause';
  document.getElementById('timeline-time').textContent=timelinePaused?'REPLAY':'LIVE';
  if(!timelinePaused && stateTimeline.length){allStates=stateTimeline[stateTimeline.length-1].states;renderReplayState();}
}
function scrubTimeline(value) {
  if(!stateTimeline.length)return; timelinePaused=true;
  document.getElementById('timeline-toggle').textContent='Resume';
  const snapshot=stateTimeline[Number(value)]; if(!snapshot)return;
  allStates=snapshot.states; document.getElementById('timeline-time').textContent=snapshot.time; renderReplayState();
}
function renderReplayState(){updateMarkers(allStates);renderHeatmap(allStates);renderIntList(allStates);updateOperationalUI(allStates,window.latestMetrics||{});}

function selectRouteMode(mode,el){activeRouteMode=mode;document.querySelectorAll('.route-mode').forEach(x=>x.classList.remove('active'));el.classList.add('active');updateRouteComparison();}
function updateRouteComparison(){
  const focus=focusedState(allStates);if(!focus)return;
  const distance=Math.max(1.4,focus.total_queue/18);
  document.getElementById('route-fast').textContent=`${Math.round(distance*3+6)} min | direct`;
  document.getElementById('route-green').textContent=`${Math.round(distance*3.7+8)} min | -${Math.min(38,Math.round(focus.total_queue*.45))}% CO2`;
  document.getElementById('route-safe').textContent=`${Math.round(distance*3.3+7)} min | low queue`;
}
function buildGreenWave(){
  const focus=focusedState(allStates);if(!focus){toast('Select an intersection first.');return;}
  const corridor=Object.values(allStates).sort((a,b)=>Math.hypot(a.lat-focus.lat,a.lng-focus.lng)-Math.hypot(b.lat-focus.lat,b.lng-focus.lng)).slice(0,4);
  corridor.forEach((s,i)=>fetch('/api/phase/'+s.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({duration:20+i*5})}));
  if(ambulanceRouteLine)map.removeLayer(ambulanceRouteLine);ambulanceRouteLine=L.polyline(corridor.map(s=>[s.lat,s.lng]),{color:'#2dd4bf',weight:5,opacity:.8,dashArray:'8,7'}).addTo(map);
  document.getElementById('corridor-list').innerHTML=corridor.map(s=>`<span class="corridor-chip">${s.name}</span>`).join('');
  logEvent(`Green wave synchronized across ${corridor.map(s=>s.name).join(', ')}.`, 'success');toast('Green wave corridor activated.');
}
function sortIntersections(){
  const sorted=Object.fromEntries(Object.entries(allStates).sort((a,b)=>b[1].total_queue-a[1].total_queue));renderIntList(sorted);toast('Signals sorted by congestion.');
}

// ══════════════════════════════════════════════════════════════════
// CHARTS
// ══════════════════════════════════════════════════════════════════
function drawCmpChart() {
  const svg = document.getElementById('cmp-chart');
  if (aiHistory.length < 2) return;
  const w=300, h=80;
  const all=[...aiHistory,...tradHistory];
  const maxV=Math.max(...all,1);
  function line(arr, color) {
    const pts=arr.map((v,i)=>{
      const x=(i/(arr.length-1))*w;
      const y=h-(v/maxV)*(h-8)-4;
      return `${x},${y}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" opacity=".9"/>`;
  }
  svg.innerHTML=line(tradHistory,'#f59e0b')+line(aiHistory,'#3b82f6');
}

// ══════════════════════════════════════════════════════════════════
// SIDEBAR LIST
// ══════════════════════════════════════════════════════════════════
function renderIntList(states) {
  const list = document.getElementById('int-list');
  list.innerHTML = Object.entries(states).map(([id, s]) => {
    const pred = s.prediction || {};
    const plevel = pred.congestion_level || 'unknown';
    const hasAcc = !!accidents[id] || !!s.accident;
    const lanesHtml = ['N','S','E','W'].map(d => {
      const q = (s.lanes[d]||{}).queue||0;
      const col = q>25?'#ef4444':q>10?'#f59e0b':'#22c55e';
      return `<div class="int-lane"><div class="int-lane-dir">${d}</div><div class="int-lane-q" style="color:${col}">${q}</div></div>`;
    }).join('');
    return `<div class="int-item${selectedInt===id?' selected':''}" data-id="${id}" onclick="selectIntersection('${id}')">
      <div class="int-header">
        <div class="int-name">${s.name}</div>
        <div class="int-badges">
          ${s.emergency?'<span class="badge badge-red">🚨</span>':''}
          ${s.incident?'<span class="badge badge-yellow">⚠</span>':''}
          ${hasAcc?'<span class="badge badge-red">💥</span>':''}
          ${s.pedestrian_crossing?'<span class="badge ped-badge">PED</span>':''}
          <span class="badge badge-blue">${s.phase.replace('_GREEN','')}</span>
          <span class="countdown-pill">${Math.ceil(s.countdown||0)}s</span>
        </div>
      </div>
      <div class="int-lanes">${lanesHtml}</div>
      <div class="int-pred pred-${plevel}">⟳ ${pred.predicted_queue??'?'} in 5 ticks — ${plevel}</div>
    </div>`;
  }).join('');
}

function populateSelect(states) {
  const sel = document.getElementById('int-select');
  if (sel.options.length > 1) return;
  Object.entries(states).forEach(([id, s]) => {
    const o = document.createElement('option');
    o.value = id; o.textContent = `${id} — ${s.name}`;
    sel.appendChild(o);
  });
}

// ══════════════════════════════════════════════════════════════════
// SSE — main data loop
// ══════════════════════════════════════════════════════════════════
const es = new EventSource('/stream');
es.onopen = () => {
  document.getElementById('conn-status').textContent = 'Live';
  document.getElementById('dot').style.background = '#22c55e';
};
es.onerror = () => {
  document.getElementById('conn-status').textContent = 'Reconnecting...';
  document.getElementById('dot').style.background = '#ef4444';
};
es.onmessage = (e) => {
  const data = JSON.parse(e.data);
  const states = data.states || {};
  const m = data.metrics || {};
  window.latestMetrics = m;
  stateTimeline.push({states:JSON.parse(JSON.stringify(states)),time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})});
  if(stateTimeline.length>60)stateTimeline.shift();
  const range=document.getElementById('timeline-range');
  if(range){range.max=Math.max(0,stateTimeline.length-1);if(!timelinePaused)range.value=range.max;}
  if (!timelinePaused) allStates = states;
  const displayStates = timelinePaused ? allStates : states;
  syncAccidents(displayStates);

  // Metrics bar
  document.getElementById('m-wait').textContent = m.avg_wait + 's';
  document.getElementById('m-trad').textContent = m.avg_trad_wait + 's';
  const saved = m.avg_trad_wait > 0 ? Math.round((m.avg_trad_wait - m.avg_wait) / m.avg_trad_wait * 100) : 0;
  document.getElementById('m-save').textContent = saved + '%';
  document.getElementById('m-emerg').textContent = m.emergencies_handled;
  document.getElementById('m-ticks').textContent = m.total_ticks;
  document.getElementById('m-fuel').textContent = (m.fuel_litres||0) + 'L';
  document.getElementById('m-co2').textContent = (m.co2_kg||0) + 'kg';
  document.getElementById('m-humidity').textContent = ((m.weather||{}).humidity_pct ?? 0) + '%';
  document.getElementById('m-temp').textContent = ((m.weather||{}).temperature_c ?? 0) + 'C';
  document.getElementById('m-ai').textContent = m.predictor_trained ? '✓ Active' : 'Training...';
  document.getElementById('tick-display').textContent = 'tick ' + m.total_ticks;
  document.getElementById('weather-display').textContent = `${m.weather_condition||'clear'} | ${((m.weather||{}).humidity_pct ?? 0)}% humidity`;
  updateWeatherUI(m.weather_condition || 'clear');

  // Comparison bars
  const maxWait = Math.max(m.avg_wait, m.avg_trad_wait, 1);
  document.getElementById('cmp-ai-bar').style.width = (m.avg_wait/maxWait*100)+'%';
  document.getElementById('cmp-tr-bar').style.width = (m.avg_trad_wait/maxWait*100)+'%';
  document.getElementById('cmp-ai-val').textContent = m.avg_wait+'s';
  document.getElementById('cmp-tr-val').textContent = m.avg_trad_wait+'s';
  document.getElementById('saving-badge').textContent =
    saved > 0 ? `✓ AI saves ${saved}% wait time` : 'Collecting comparison data...';

  // History charts
  aiHistory.push(m.avg_wait); tradHistory.push(m.avg_trad_wait);
  if (aiHistory.length > MAX_H) aiHistory.shift();
  if (tradHistory.length > MAX_H) tradHistory.shift();
  drawCmpChart();

  // Map markers
  if (Object.keys(mapMarkers).length === 0) {
    initMarkers(displayStates);
    initVehicles(states);      // ← start vehicles on first data
    requestAnimationFrame(animateVehicles); // ← kick off animation loop
  }
  updateMarkers(displayStates);

  // Heatmap
  renderHeatmap(displayStates);

  // Sidebar
  populateSelect(displayStates);
  renderIntList(displayStates);
  updateOperationalUI(displayStates,m);
  if (userLatLng) showNearest(userLatLng[0], userLatLng[1]);
  document.getElementById('ai-toggle').checked = m.ai_mode !== false;
};
</script>
</body>
</html>"""


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    PORT = int(os.environ.get("PORT", os.environ.get("SMARTFLOW_PORT", "8000")))

    # Start background training
    t_train = threading.Thread(target=training_thread, daemon=True)
    t_train.start()

    # Start simulation loop
    t_sim = threading.Thread(target=simulation_loop, daemon=True)
    t_sim.start()

    _legacy_banner = f"""
╔══════════════════════════════════════════╗
║   SmartFlow — AI Traffic Management      ║
╠══════════════════════════════════════════╣
║   Dashboard: http://localhost:{PORT}        ║
║   API:       http://localhost:{PORT}/api    ║
╚══════════════════════════════════════════╝

Endpoints:
  GET  /api/status          - all intersection states
  GET  /api/stats           - RL agent stats
  GET  /api/predict/<id>    - congestion prediction
  POST /api/train           - retrain predictor
  POST /api/phase/<id>      - set phase duration (body: {{"duration": 45}})
  GET  /stream              - SSE live stream

Press Ctrl+C to stop.
"""
    print(
        f"SmartFlow AI Traffic Management\n"
        f"Dashboard: http://localhost:{PORT}\n"
        f"API:       http://localhost:{PORT}/api\n"
        "Press Ctrl+C to stop."
    )

    server = ThreadingHTTPServer(("0.0.0.0", PORT), SmartFlowHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SmartFlow] Shutting down. Saving model checkpoints...")
        tms.save_all("checkpoints/")
        print("[SmartFlow] Done.")


if __name__ == "__main__":
    main()

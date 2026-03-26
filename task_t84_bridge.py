#!/usr/bin/env python3
"""
T84: Agent Communication Bridge
Cross-machine task coordination for Vulkan inference fleet.
Supports sys1 (Vulkan), sys2-sys7 (MLX), sys11 (CUDA).
"""
import json
import socket
import threading
import time
import os
from datetime import datetime

FLEET_NODES = {
    "sys1": {"type": "vulkan", "role": "orchestrator", "ip": "192.168.1.128"},
    "sys2": {"type": "mlx", "role": "worker", "ip": "192.168.1.129"},
    "sys3": {"type": "mlx", "role": "worker", "ip": "192.168.1.130"},
    "sys4": {"type": "mlx", "role": "worker", "ip": "192.168.1.131"},
    "sys5": {"type": "mlx", "role": "worker", "ip": "192.168.1.132"},
    "sys6": {"type": "mlx", "role": "worker", "ip": "192.168.1.133"},
    "sys7": {"type": "mlx", "role": "worker", "ip": "192.168.1.134"},
    "sys11": {"type": "cuda", "role": "brain", "ip": "192.168.1.11"}
}

PORT = 9999
QUEUE_FILE = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")

class AgentBridge:
    def __init__(self):
        self.tasks = []
        self.lock = threading.Lock()
        self.running = True

    def load_tasks(self):
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                content = f.read()
            # Parse ### TXX: [STATUS] description
            for line in content.split('\n'):
                if line.startswith('### T') and '[READY]' in line:
                    parts = line.split(':', 1)
                    tid = parts[0].replace('### ', '').strip()
                    desc = parts[1].split(']', 1)[-1].strip() if ']' in parts[1] else ''
                    self.tasks.append({'id': tid, 'desc': desc, 'status': 'READY'})

    def broadcast_status(self, sock):
        status = {
            'timestamp': datetime.now().isoformat(),
            'node': socket.gethostname(),
            'tasks_loaded': len(self.tasks),
            'fleet': list(FLEET_NODES.keys())
        }
        sock.sendto(json.dumps(status).encode(), ('192.168.1.255', PORT))

    def handle_request(self, data, addr, sock):
        req = json.loads(data.decode())
        resp = {'ack': True, 'from': socket.gethostname()}
        if req.get('action') == 'get_task':
            with self.lock:
                if self.tasks:
                    task = self.tasks.pop(0)
                    resp['task'] = task
                else:
                    resp['task'] = None
        elif req.get('action') == 'update_progress':
            print(f"Progress update: {req.get('task_id')} -> {req.get('percent')}%")
        sock.sendto(json.dumps(resp).encode(), addr)

    def run_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(('0.0.0.0', PORT))
        sock.settimeout(5.0)
        print(f"[BRIDGE] Listening on port {PORT}...")
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                self.handle_request(data, addr, sock)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")

    def start(self):
        self.load_tasks()
        server_thread = threading.Thread(target=self.run_server, daemon=True)
        server_thread.start()
        print(f"[BRIDGE] Started. Loaded {len(self.tasks)} tasks.")
        return server_thread

if __name__ == '__main__':
    bridge = AgentBridge()
    bridge.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.running = False
        print("[BRIDGE] Stopped.")

#!/usr/bin/env python3
"""
T48: Fleet registration — announce this node's model/TPS to the network.
Writes a JSON status file and can optionally POST to a central registry.

Usage:
  python fleet_register.py  # writes ~/AGENT/fleet_status.json
  python fleet_register.py --registry http://10.255.255.1:9000/register
"""
import json
import os
import socket
import time
import subprocess
import requests

def get_system_info():
    hostname = socket.gethostname()
    ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.1", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "unknown"

    gpu = "unknown"
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if "deviceName" in line:
                gpu = line.split("=")[-1].strip()
                break
    except:
        pass

    ram_gb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_gb = int(line.split()[1]) // (1024 * 1024)
                    break
    except:
        pass

    return {
        "hostname": hostname,
        "ip": ip,
        "gpu": gpu,
        "ram_gb": ram_gb,
        "os": "Fedora Asahi Remix",
    }


def get_model_info():
    """Check if vLLM server is running and get model info."""
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=2)
        models = r.json().get("data", [])
        return [{"id": m["id"], "port": 8000} for m in models]
    except:
        return []


def get_performance():
    """Quick TPS test if server is running."""
    try:
        models = get_model_info()
        if not models:
            return {}
        model_id = models[0]["id"]
        t0 = time.time()
        r = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10, "temperature": 0,
        }, timeout=30)
        elapsed = time.time() - t0
        data = r.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {"tps": round(tokens / elapsed, 1) if elapsed > 0 else 0, "latency_ms": round(elapsed * 1000)}
    except:
        return {}


def register(registry_url=None):
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": get_system_info(),
        "models": get_model_info(),
        "performance": get_performance(),
        "engine": "ggml-vulkan",
        "backend": "Vulkan 1.4 (Mesa Honeykrisp)",
    }

    # Save locally
    path = os.path.expanduser("~/AGENT/fleet_status.json")
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Status saved to {path}")
    print(json.dumps(status, indent=2))

    # POST to registry if specified
    if registry_url:
        try:
            r = requests.post(registry_url, json=status, timeout=5)
            print(f"Registered with {registry_url}: {r.status_code}")
        except Exception as e:
            print(f"Registry POST failed: {e}")

    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", help="Central registry URL")
    args = parser.parse_args()
    register(args.registry)

# Agent-to-Model Routing Protocol Design

## Decision: HTTP/REST with JSON (Phase 1) -> gRPC (Phase 2)

### Rationale:
- HTTP/REST: Faster to implement, easier debugging, works with existing tools
- gRPC: Better for production (binary serialization, streaming, multiplexing)
- Start with HTTP, migrate to gRPC once core logic validated

## Architecture Overview

Client -> Router -> Model Servers (0.5B, 8B, 120B)

## Request Format (HTTP/JSON)

POST /generate
Content-Type: application/json

Body:
{
  "model": "auto",
  "prompt": "Capital of France?",
  "params": {
    "temperature": 0,
    "max_tokens": 10,
    "top_p": 0.9
  },
  "priority": "normal",
  "timeout_ms": 5000
}

## Response Format

{
  "request_id": "uuid-1234",
  "model_used": "8b",
  "text": "Paris",
  "tokens_generated": 1,
  "tps": 22.3,
  "latency_ms": 45,
  "status": "success"
}

## Routing Logic

Simple facts (<5 tokens expected) -> 0.5B (fast)
General Q/A, chat -> 8B (balanced)
Complex reasoning, math, code -> 120B (smart)
Priority override -> User-specified

## Implementation Plan

1. Create Python router service (router_service.py)
2. Add model health checks (ping endpoints)
3. Implement load balancing (round-robin + health-aware)
4. Add request queuing for overloaded models
5. Monitor TPS, latency, error rates

## Files to Create

- ~/AGENT/router_service.py - Main routing server
- ~/AGENT/model_client.py - Client library for agents
- ~/AGENT/test_routing.py - Load testing suite

Total estimated time: 4 hours

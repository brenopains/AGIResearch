"""Shim: redireciona p/ llm_client (Venice GLM -> Nemotron fallback).
Os agentes continuam fazendo 'from shared.claude_client import ask' sem mudar."""
from .llm_client import ask, ask_smart, DEFAULT_MODEL, DEFAULT_HAIKU, SMART_MODEL, set_telemetry

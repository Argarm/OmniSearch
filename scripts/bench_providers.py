"""Benchmark cost and latency across LLM providers.

Runs a handful of prompts against each configured provider, measuring wall-clock
latency and token usage, then estimates cost from a small pricing table. Intended
for local, ad-hoc measurement — it is NOT run in CI and requires API keys for the
providers you want to test.

Usage:
    # Test whichever providers have keys in the environment:
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python scripts/bench_providers.py

    # Limit to specific providers:
    python scripts/bench_providers.py --providers openai

Pricing below is list price per 1M tokens (USD) and will drift — update before
quoting numbers. Source: each provider's public pricing page.
"""
from __future__ import annotations

import argparse
import os
import time

from langchain_core.messages import HumanMessage

from backend.llm_factory import _PROVIDERS

# USD per 1M tokens (input, output). Update from provider pricing pages.
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {"gpt-4o-mini": (0.15, 0.60)},
    "anthropic": {"claude-3-5-haiku-20241022": (0.80, 4.00)},
}

PROMPTS = [
    "Summarize the role of a vector database in a RAG system in two sentences.",
    "List three trade-offs between self-hosted and managed embedding models.",
    "Explain what query rewriting does in a conversational RAG pipeline.",
]


def _estimate_cost(provider_key: str, model_id: str, in_tok: int, out_tok: int) -> float | None:
    rates = PRICING.get(provider_key, {}).get(model_id)
    if not rates:
        return None
    in_rate, out_rate = rates
    return (in_tok / 1_000_000) * in_rate + (out_tok / 1_000_000) * out_rate


def _token_counts(response) -> tuple[int, int]:
    """Best-effort extraction of (input, output) tokens from a LangChain response."""
    usage = getattr(response, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def bench_provider(provider_key: str) -> None:
    provider = _PROVIDERS[provider_key]()
    model_id = provider.model_id
    print(f"\n=== {provider_key} ({model_id}) ===")
    try:
        llm = provider.build()
    except Exception as exc:  # surface config/key errors plainly
        print(f"  skipped: {exc}")
        return

    latencies: list[float] = []
    in_total = out_total = 0
    for prompt in PROMPTS:
        start = time.perf_counter()
        resp = llm.invoke([HumanMessage(content=prompt)])
        latencies.append(time.perf_counter() - start)
        in_tok, out_tok = _token_counts(resp)
        in_total += in_tok
        out_total += out_tok

    n = len(PROMPTS)
    avg_latency = sum(latencies) / n
    cost = _estimate_cost(provider_key, model_id, in_total // n, out_total // n)
    print(f"  avg latency:   {avg_latency:.2f}s over {n} prompts")
    print(f"  avg tokens:    {in_total // n} in / {out_total // n} out")
    if cost is not None:
        print(f"  est cost/query: ${cost:.6f}")
    else:
        print("  est cost/query: (no pricing entry for this model)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=sorted(_PROVIDERS),
        default=sorted(_PROVIDERS),
        help="Providers to benchmark (default: all with available keys).",
    )
    args = parser.parse_args()

    key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    for provider_key in args.providers:
        if not os.getenv(key_env.get(provider_key, "")):
            print(f"\n=== {provider_key} === skipped: {key_env[provider_key]} not set")
            continue
        bench_provider(provider_key)


if __name__ == "__main__":
    main()

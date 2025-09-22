#!/usr/bin/env python3
"""
OpenRouter coherency checker example.

Runs coherency tests for a given OpenRouter model, optionally enforcing
reasoning ("thinking") behavior, and prints good/bad sub‑providers.

Usage:
  python examples/coherency_check.py --model qwen/qwen3-next-80b-a3b-thinking \
      --workers 6 --reasoning --reasoning-tokens 2048

  # Force a specific OpenRouter sub‑provider (may repeat the flag)
  python examples/coherency_check.py --model deepseek/deepseek-chat-v3.1 \
      --force-subprovider deepseek --reasoning --workers 4 --verbose

Notes:
  - --reasoning and --no-reasoning are mutually exclusive.
  - Specify at most one of --reasoning-tokens or --reasoning-effort.
  - --force-subprovider is only valid with OpenRouter targets.
"""
import argparse
import sys
import os

# Ensure local package is importable when running from a clone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client.testing import CoherencyTester


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run coherency tests for an OpenRouter model and report good/bad sub‑providers.")
    p.add_argument("--model", required=True, help="OpenRouter model id (e.g., qwen/qwen3-next-80b-a3b-thinking)")
    p.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    p.add_argument("--verbose", action="store_true", help="Verbose mode: dump full raw responses on failures")
    p.add_argument(
        "--force-subprovider",
        dest="force_subproviders",
        action="append",
        help="Force one OpenRouter sub‑provider (may be repeated). Only valid with OpenRouter.",
    )

    # Reasoning controls (mutually exclusive enable/disable)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--reasoning", action="store_true", help="Enable reasoning and enforce thinking output")
    group.add_argument("--no-reasoning", action="store_true", help="Disable reasoning and fail providers that return thinking output")

    p.add_argument("--reasoning-tokens", type=int, help="Reasoning max_tokens budget (when --reasoning)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="Reasoning effort level (when --reasoning)")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Build request_overrides
    request_overrides = {}
    reasoning_cfg = {}

    # Only add enforcement if user opted in either direction
    if args.reasoning:
        reasoning_cfg["enabled"] = True
    elif args.no_reasoning:
        reasoning_cfg["enabled"] = False

    if reasoning_cfg.get("enabled") is True:
        if args.reasoning_tokens is not None and args.reasoning_effort is not None:
            print("ERROR: Specify only one of --reasoning-tokens or --reasoning-effort.", file=sys.stderr)
            return 2
        if args.reasoning_tokens is not None:
            reasoning_cfg["max_tokens"] = args.reasoning_tokens
        if args.reasoning_effort is not None:
            reasoning_cfg["effort"] = args.reasoning_effort

    if reasoning_cfg:
        request_overrides["reasoning"] = reasoning_cfg

    # Safety: --force-subprovider only valid when using OpenRouter
    if args.force_subproviders and len(args.force_subproviders) > 0:
        target_provider = "openrouter"
        if target_provider.lower() != "openrouter":
            print("ERROR: --force-subprovider is only supported with OpenRouter.", file=sys.stderr)
            return 2

    # Set up tester and run
    tester = CoherencyTester(
        target_provider_name="openrouter",
        target_model_id=args.model,
        num_workers=args.workers,
        test_prompts=None,
        allowed_subproviders=args.force_subproviders if args.force_subproviders else None,
        request_overrides=request_overrides if request_overrides else None,
        verbose=bool(args.verbose),
    )

    results = tester.run_tests()

    passed = results.get("passed_providers", [])
    failed = results.get("failed_providers", [])

    # Report
    print("\n=== Coherency Results ===")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    if args.force_subproviders:
        print(f"Forced sub‑providers: {', '.join(args.force_subproviders)}")
    if reasoning_cfg:
        print(f"Reasoning config: {reasoning_cfg}")
    else:
        print("Reasoning config: <none>")

    print(f"\nGood providers ({len(passed)}): {', '.join(passed) if passed else '<none>'}")
    print(f"Bad providers  ({len(failed)}): {', '.join(failed) if failed else '<none>'}")

    if failed:
        print("\nTip: exclude bad providers using ignore_list when calling retry_request.")

    # Return non-zero if no provider passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

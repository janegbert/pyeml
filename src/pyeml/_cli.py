"""CLI entry point for pyeml."""

from __future__ import annotations

import argparse
import cmath
import math
import sys

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyeml",
        description="EML symbolic regression — all elementary functions from exp(x) - ln(y)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── discover ─────────────────────────────────────────────────────
    p_disc = sub.add_parser("discover", help="Find EML form of a function from data")
    p_disc.add_argument("--expr", required=True,
                        help="Target: exp, ln, sin, cos, sqrt",
                        choices=["exp", "ln", "sin", "cos", "sqrt"])
    p_disc.add_argument("--depth", type=int, default=4, help="Max tree depth")
    p_disc.add_argument("--restarts", type=int, default=10, help="Restarts per depth")
    p_disc.add_argument("--x-min", type=float, default=0.5)
    p_disc.add_argument("--x-max", type=float, default=3.0)
    p_disc.add_argument("--points", type=int, default=200)

    # ── compile ──────────────────────────────────────────────────────
    p_comp = sub.add_parser("compile", help="Compile expression to EML form")
    p_comp.add_argument("--expr", required=True, help="Expression to compile",
                        choices=["exp(x)", "ln(x)", "e", "exp(exp(x))"])

    # ── verify ───────────────────────────────────────────────────────
    sub.add_parser("verify", help="Verify EML identities from the paper")

    # ── demo ─────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Run demonstration (exp, e, ln)")

    args = parser.parse_args(argv)

    if args.command == "discover":
        _cmd_discover(args)
    elif args.command == "compile":
        _cmd_compile(args)
    elif args.command == "verify":
        _cmd_verify()
    elif args.command == "demo":
        _cmd_demo()
    else:
        parser.print_help()


# Map of safe target functions (no eval needed)
_TARGET_FNS = {
    "exp": cmath.exp,
    "ln": cmath.log,
    "sin": cmath.sin,
    "cos": cmath.cos,
    "sqrt": cmath.sqrt,
}


def _cmd_discover(args: argparse.Namespace) -> None:
    from pyeml import discover

    fn = _TARGET_FNS[args.expr]
    x = np.linspace(args.x_min, args.x_max, args.points)
    y = np.array([fn(xi) for xi in x], dtype=np.complex128)

    print(f"Target: {args.expr}(x)")
    print(f"Searching depths 1..{args.depth}, {args.restarts} restarts each")

    result = discover(x, y.real, max_depth=args.depth, n_restarts=args.restarts)
    if result and result.is_exact:
        print(f"\nFound EXACT match: {result.expression}")
        print(f"  EML form: {result.eml_expression}")
        print(f"  Depth: {result.depth}")
    elif result:
        print(f"\nBest approximation: {result.expression}")
        print(f"  Loss: {result.snapped_loss:.2e}")
    else:
        print("\nNo solution found. Try increasing --depth or --restarts.")


def _cmd_compile(args: argparse.Namespace) -> None:
    from pyeml._compiler import compile_expr

    try:
        node = compile_expr(args.expr)
        print(f"{args.expr} = {node}")
        print(f"  Depth: {node.depth}")
        print(f"  Leaves: {node.leaf_count}")
    except NotImplementedError as e:
        print(f"Cannot compile: {e}")


def _cmd_verify() -> None:
    from pyeml._operator import eml_scalar

    E = eml_scalar
    checks = [
        ("eml(1,1) = e",           E(1, 1),                     cmath.e),
        ("eml(x,1) = exp(2)",      E(2, 1),                     cmath.exp(2)),
        ("eml(e,1) = e^e",         E(cmath.e, 1),               cmath.exp(cmath.e)),
        ("ln(2)",                   E(1, E(E(1, 2), 1)),         cmath.log(2)),
        ("ln(0.5)",                 E(1, E(E(1, 0.5), 1)),       cmath.log(0.5)),
    ]

    print("Verifying EML identities from the paper:\n")
    all_ok = True
    for name, got, expected in checks:
        err = abs(got - expected)
        ok = err < 1e-10
        status = "OK" if ok else f"FAIL (err={err:.2e})"
        print(f"  {name:30s}  {status}")
        all_ok = all_ok and ok

    print(f"\n{'All passed!' if all_ok else 'Some checks failed.'}")


def _cmd_demo() -> None:
    from pyeml import discover

    print("=" * 60)
    print("  pyeml demo — EML Symbolic Regression")
    print("  Based on Odrzywołek (2026), arXiv:2603.21852")
    print("=" * 60)

    targets = [
        ("exp(x)", cmath.exp, 1, 5, (0.1, 2.0)),
        ("e",      lambda x: cmath.e, 1, 5, (0.5, 3.0)),
        ("ln(x)",  cmath.log, 3, 20, (0.5, 4.0)),
    ]

    for name, fn, depth, restarts, xr in targets:
        print(f"\n--- Target: {name} ---")
        x = np.linspace(xr[0], xr[1], 200)
        y = np.array([fn(xi) for xi in x], dtype=np.complex128)

        result = discover(
            x, y.real,
            max_depth=depth,
            n_restarts=restarts,
            x_range=xr,
        )

        if result and result.is_exact:
            print(f"  EXACT: {result.expression}")
        elif result:
            print(f"  Best: {result.expression} (loss={result.snapped_loss:.2e})")
        else:
            print("  Failed to find expression.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

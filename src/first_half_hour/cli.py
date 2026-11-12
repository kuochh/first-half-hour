"""The `fhh` command."""

import argparse
from pathlib import Path

from first_half_hour import fingerprint, reproduce


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fhh")
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("fingerprint", help="per-symbol fingerprint of the raw bars")
    fp.add_argument("--data", type=Path, required=True, help="folder holding <SYMBOL>/ subfolders")
    fp.add_argument("--hashes", type=Path, help="also write a sha256 per file to this path")
    rp = sub.add_parser(
        "reproduce",
        help="chapters 1 and 2, logged to MLflow",
        description="Run chapter 1 (items 1-9) and chapter 2 (walk-forward ML vs the volatility "
        "formula) end to end. Thread use follows the environment, nothing is capped here: "
        "set e.g. OMP_NUM_THREADS=4 (boosted trees, BLAS) to leave cores free.",
    )
    rp.add_argument(
        "--chapter", choices=("1", "2", "all"), default="all", help="which chapter (default all)"
    )
    rp.add_argument("--synthetic", action="store_true", help="run on synthetic bars in a temp dir")
    rp.add_argument("--data", type=Path, default=Path("data/raw"), help="folder of <SYMBOL>/ bars")
    rp.add_argument("--daily", type=Path, default=Path("data/daily"), help="daily files folder")
    rp.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="run outputs and the MLflow store (mlruns.db)",
    )
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        print(reproduce.reproduce(args.data, args.daily, args.out, args.synthetic, args.chapter))
        return

    table, dropped = fingerprint.fingerprint(args.data)
    print(fingerprint.to_markdown(table))
    print(f"\nDropped sessions (missing a key bar): {dropped.height}")
    if dropped.height:
        print(fingerprint.to_markdown(dropped.select("symbol", "session", "lacking")))
    if args.hashes:
        args.hashes.write_text("\n".join(fingerprint.file_hashes(args.data)) + "\n")

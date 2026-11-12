"""The `fhh` command."""

import argparse
from pathlib import Path

from first_half_hour import fingerprint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fhh")
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("fingerprint", help="per-symbol fingerprint of the raw bars")
    fp.add_argument("--data", type=Path, required=True, help="folder holding <SYMBOL>/ subfolders")
    fp.add_argument("--hashes", type=Path, help="also write a sha256 per file to this path")
    args = parser.parse_args(argv)

    table, dropped = fingerprint.fingerprint(args.data)
    print(fingerprint.to_markdown(table))
    print(f"\nDropped sessions (missing a key bar): {dropped.height}")
    if dropped.height:
        print(fingerprint.to_markdown(dropped.select("symbol", "session", "lacking")))
    if args.hashes:
        args.hashes.write_text("\n".join(fingerprint.file_hashes(args.data)) + "\n")

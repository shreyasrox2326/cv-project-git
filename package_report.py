#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Zip generated interpretability report assets.")
    parser.add_argument("--report-root", default="./interpretability_report")
    parser.add_argument("--zip-path", default="./interpretability_report.zip")
    args = parser.parse_args()

    report_root = Path(args.report_root)
    zip_path = Path(args.zip_path)
    if not report_root.exists():
        raise SystemExit(f"Report root not found: {report_root}")
    if zip_path.exists():
        zip_path.unlink()
    archive_base = zip_path.with_suffix("")
    shutil.make_archive(archive_base.as_posix(), "zip", report_root)
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pearls AQI Predictor - Master Project Orchestrator & CLI Runner.

Provides a unified command-line entrypoint to run pipelines, launch the dashboard,
execute verification suites, run audits, and generate project reports.
"""

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> int:
    """Execute a command list directly and stream output."""
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130
    except Exception as e:
        print(f"[ERROR] Failed to run command {' '.join(cmd)}: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pearls AQI Predictor - Master Project CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audit                   # Run pre-submission audit suite
  python main.py verify                  # Run end-to-end system verification
  python main.py feature --city Karachi  # Run feature engineering pipeline
  python main.py train --city Karachi    # Run multi-model training & benchmarking
  python main.py app                     # Launch Streamlit dashboard
  python main.py report                  # Compile final submission report
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Subcommand: feature
    sp_feature = subparsers.add_parser("feature", help="Run 1_feature_pipeline.py")
    sp_feature.add_argument("--city", default="Karachi", help="Target city (Karachi, Lahore, Islamabad)")
    sp_feature.add_argument("--past-days", type=int, default=90, help="Days of historical hourly backfill")
    sp_feature.add_argument("--forecast-days", type=int, default=3, help="Days of future hourly forecast")
    sp_feature.add_argument("--data-dir", default="data", help="Directory to save parquet tables")

    # Subcommand: train
    sp_train = subparsers.add_parser("train", help="Run 2_training_pipeline.py")
    sp_train.add_argument("--city", default="Karachi", help="Target city for training")
    sp_train.add_argument("--data-dir", default="data", help="Directory containing parquet tables")
    sp_train.add_argument("--models-dir", default="models", help="Directory to save model artifacts")

    # Subcommand: app
    subparsers.add_parser("app", help="Launch Streamlit interactive dashboard (3_app.py)")

    # Subcommand: audit
    subparsers.add_parser("audit", help="Run standalone pre-submission audit suite (audit_submission.py)")

    # Subcommand: verify
    subparsers.add_parser("verify", help="Run end-to-end system verification suite (verify_system.py)")

    # Subcommand: report
    subparsers.add_parser("report", help="Generate final markdown and HTML reports (generate_final_report.py)")

    args, extra_args = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    py_exe = sys.executable

    if args.command == "feature":
        cmd = [
            py_exe,
            "1_feature_pipeline.py",
            "--city", args.city,
            "--past-days", str(args.past_days),
            "--forecast-days", str(args.forecast_days),
            "--data-dir", args.data_dir,
        ] + extra_args
        sys.exit(run_cmd(cmd))

    elif args.command == "train":
        cmd = [
            py_exe,
            "2_training_pipeline.py",
            "--city", args.city,
            "--data-dir", args.data_dir,
            "--models-dir", args.models_dir,
        ] + extra_args
        sys.exit(run_cmd(cmd))

    elif args.command == "app":
        cmd = ["streamlit", "run", "3_app.py"] + extra_args
        sys.exit(run_cmd(cmd))

    elif args.command == "audit":
        cmd = [py_exe, "audit_submission.py"] + extra_args
        sys.exit(run_cmd(cmd))

    elif args.command == "verify":
        cmd = [py_exe, "verify_system.py"] + extra_args
        sys.exit(run_cmd(cmd))

    elif args.command == "report":
        cmd = [py_exe, "generate_final_report.py"] + extra_args
        sys.exit(run_cmd(cmd))


if __name__ == "__main__":
    main()

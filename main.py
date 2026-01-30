"""
Master Experiment.

DESCRIPTION:
    Unified entry point for running Classical, Quantum, or Comparative experiments.
    It wraps 'classical.py' and 'quantum.py' to allow side-by-side execution.

USAGE:
    # 1. Run Quantum Only (Same as before)
    uv run main.py --mode quantum -p 48 -j 8

    # 2. Run Classical Only
    uv run main.py --mode classical -p 48

    # 3. Run BOTH and Compare
    uv run main.py --mode compare -p 48 411 -j 8
"""

import argparse
import sys
from loguru import logger

from quantum import run_quantum
from classical import run_classical


def parse_arguments():
    parser = argparse.ArgumentParser(description="Runner for QSVM vs SVM Experiments.")

    parser.add_argument(
        '-m', '--mode',
        type=str,
        default='compare',
        choices=['quantum', 'classical', 'compare'],
        help="Experiment Mode: 'quantum', 'classical', or 'compare' (runs both)."
    )

    parser.add_argument(
        '-p', '--patients',
        nargs='+',
        default=["48", "411"],
        help="List of Patient IDs to use."
    )

    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=None,
        help="Total number of samples to use."
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=8,
        help="Number of CPU cores."
    )

    # Quantum Specific Arguments
    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=None,
        help="[Quantum] Broadcast Batch Size."
    )

    parser.add_argument(
        '-be', '--backend',
        type=str,
        default="lightning.gpu",
        choices=["lightning.gpu", "lightning.qubit", "default.qubit"],
        help="[Quantum] PennyLane Backend."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info(f"Experiment starting | Mode: {args.mode.upper()}")

    results = {}

    if args.mode in ['classical', 'compare']:
        logger.info("Starting Classical SVM Pipeline")
        try:
            metrics_c = run_classical(args)
            results['Classical'] = metrics_c
            logger.success("Classical Pipeline Finished Successfully.")
        except Exception as e:
            logger.error(f"Classical Pipeline Failed: {e}")

    if args.mode in ['quantum', 'compare']:
        logger.info("Starting Quantum QSVM Pipeline")
        try:
            metrics_q = run_quantum(args)
            results['Quantum'] = metrics_q
            logger.success("Quantum Pipeline Finished Successfully.")
        except Exception as e:
            logger.error(f"Quantum Pipeline Failed: {e}")

    logger.info("All tasks completed.")

    if 'Classical' in results:
        m = results['Classical']
        logger.info(f"CLASSICAL RESULT | MSE: {m['mse']:.4f}  | RMSE: {m['rmse']:.4f} | "
                    f"R2: {m['r2']:.4f} | R: {m['pearson']:.4f} | 95% CI: {m['ci']:.4f}")

    if 'Quantum' in results:
        m = results['Quantum']
        logger.info(f"QUANTUM   RESULT | MSE: {m['mse']:.4f}  | RMSE: {m['rmse']:.4f} | "
                    f"R2: {m['r2']:.4f} | R: {m['pearson']:.4f} | 95% CI: {m['ci']:.4f}")


if __name__ == "__main__":
    main()

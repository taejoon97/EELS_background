from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eels_common import crop_window, fit_background, load_msa


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python version of EELS_fitting.m")
    p.add_argument("msa_file", type=Path, help="Input .msa file")
    p.add_argument("--delimiter", default=",", help="Delimiter in .msa file")
    p.add_argument("--skiprows", type=int, default=20, help="Header lines to skip")
    p.add_argument("--start-edge", type=float, default=176.0)
    p.add_argument("--end-edge", type=float, default=381.0)
    p.add_argument("--model", choices=["exp1", "exp2", "power1", "power2"], default="power2")
    p.add_argument("--exclude-start", type=float, default=200.0)
    p.add_argument("--exclude-stop", type=float, default=280.0)
    p.add_argument("--exclude-step", type=float, default=10.0)
    p.add_argument("--save-plot", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    x, y = load_msa(args.msa_file, delimiter=args.delimiter, skiprows=args.skiprows)
    x2, y2 = crop_window(x, y, start_edge=args.start_edge, end_edge=args.end_edge)

    fig, ax = plt.subplots(figsize=(10, 6))

    for excl in np.arange(args.exclude_start, args.exclude_stop + 1e-9, args.exclude_step):
        _, _, residuals = fit_background(x2, y2, model_name=args.model, exclude_above=float(excl))
        ax.plot(x2, residuals, linewidth=2, label=f"{excl:.0f}")

    ax.plot(x2, y2, color="black", linewidth=2, label="Original EELS data")
    ax.set_title("EEL spectra after subtracting fitted curves")
    ax.set_xlabel("eV")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.4)
    ax.legend(title="Fits excluding data above (i) eV", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()

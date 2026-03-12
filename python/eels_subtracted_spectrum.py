from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from eels_common import load_msa


def load_vector_from_mat(path: Path, var_name: str | None) -> np.ndarray:
    mat = loadmat(path)
    if var_name:
        if var_name not in mat:
            raise KeyError(f"Variable '{var_name}' not found in {path}")
        vec = np.asarray(mat[var_name]).squeeze()
        return vec

    candidates = [k for k in mat.keys() if not k.startswith("__")]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer variable from {path}. Found variables: {candidates}. Pass --x-var/--y-var."
        )
    return np.asarray(mat[candidates[0]]).squeeze()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python version of EELS_subtracted_spectrum.m")
    p.add_argument("msa_file", type=Path)
    p.add_argument("xdata_mat", type=Path, help=".mat file containing xdata2")
    p.add_argument("residuals_mat", type=Path, help=".mat file containing residuals")
    p.add_argument("--x-var", default=None, help="Variable name for xdata2 inside xdata_mat")
    p.add_argument("--y-var", default=None, help="Variable name for residuals inside residuals_mat")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--skiprows", type=int, default=20)
    p.add_argument("--output-txt", type=Path, default=Path("subtracted-spectrum.txt"))
    p.add_argument("--save-plot", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    x1, y1 = load_msa(args.msa_file, delimiter=args.delimiter, skiprows=args.skiprows)
    x2 = load_vector_from_mat(args.xdata_mat, args.x_var)
    y2 = load_vector_from_mat(args.residuals_mat, args.y_var)

    out = np.column_stack([x2, y2])
    np.savetxt(args.output_txt, out, delimiter="\t", header="x2\ty2", comments="")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x1, y1, color="black", linewidth=2, label="Original EEL spectrum")
    ax.plot(x2, y2, color="red", linewidth=2, label="EEL spectrum after fitting")
    ax.set_title("Original and subtracted EEL spectra")
    ax.set_xlabel("eV")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()

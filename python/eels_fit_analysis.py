from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eels_common import crop_window, fit_background, load_msa


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python version of EELS_fit_analysis.m")
    p.add_argument("msa_file", type=Path, help="Input .msa file")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--skiprows", type=int, default=20)
    p.add_argument("--start-edge", type=float, default=176.0)
    p.add_argument("--end-edge", type=float, default=381.0)
    p.add_argument("--exclude-above", type=float, default=280.0)
    p.add_argument("--model", choices=["exp1", "exp2", "power1", "power2"], default="power2")
    p.add_argument("--int-start", type=float, default=284.0)
    p.add_argument("--int-end", type=float, default=300.0)
    p.add_argument("--save-prefix", type=Path, default=Path("eels_fit"))
    p.add_argument("--save-plot", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    x, y = load_msa(args.msa_file, delimiter=args.delimiter, skiprows=args.skiprows)
    x2, y2 = crop_window(x, y, start_edge=args.start_edge, end_edge=args.end_edge)

    fit_result, fitted_y, residuals = fit_background(
        x2, y2, model_name=args.model, exclude_above=args.exclude_above
    )

    int_mask = (x2 > args.int_start) & (x2 < args.int_end)
    x_int = x2[int_mask]
    residuals_int = residuals[int_mask]
    fitted_int = fitted_y[int_mask]

    ik = float(np.trapz(residuals_int, x_int))
    ib = float(np.trapz(fitted_int, x_int))
    varib = float(np.var(fitted_int))
    h = float((ib + varib) / ib) if ib != 0 else float("inf")
    snr = float(ik / np.sqrt(ik + (h * ib))) if np.isfinite(h) else float("nan")

    params_path = args.save_prefix.parent / f"{args.save_prefix.name}_params.json"
    data_path = args.save_prefix.parent / f"{args.save_prefix.name}_residuals.csv"

    data_out = np.column_stack([x2, y2, fitted_y, residuals])
    np.savetxt(data_path, data_out, delimiter=",", header="x_eV,counts,fitted,residuals", comments="")

    summary = {
        "model": args.model,
        "exclude_above_eV": args.exclude_above,
        "params": fit_result.params.tolist(),
        "covariance": fit_result.covariance.tolist(),
        "ik": ik,
        "ib": ib,
        "varib": varib,
        "h": h,
        "snr": snr,
    }
    params_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(x2, y2, color="black", linewidth=2, label="Original data")
    ax1.plot(x2, fitted_y, color="blue", linewidth=2, label="Fitted curve")
    ax1.plot(x2, residuals, color="red", linewidth=2, label="Subtracted spectrum")
    ax1.set_xlabel("eV")
    ax1.set_ylabel("Counts")
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc="best", frameon=False)
    ax1.set_title(f"Background fit of data below {args.exclude_above:.0f} eV")

    residual_fit_mask = x2 < args.exclude_above
    ax2.scatter(x2[residual_fit_mask], residuals[residual_fit_mask], s=10, label="Residuals")
    ax2.axhline(0, linestyle="--", linewidth=1, color="black", label="Zero line")
    ax2.set_xlabel("eV")
    ax2.set_ylabel("Counts")
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc="best", frameon=False)
    ax2.set_title("Residuals of fit")

    plt.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()

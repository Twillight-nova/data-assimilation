#!/usr/bin/env python3
import os
import sys
import subprocess
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse


def run_once(python_exe: str, script_path: str) -> Tuple[int, float, float, float, float, float]:
    """adjoint_noise.py を1回実行し、FINAL行をパースして返す。

    Returns: (iter, cost, gnorm, xf, yf, ystd)
    """
    result = subprocess.run(
        [python_exe, script_path],
        stdout=subprocess.PIPE,  # FINAL行のみ取得
        stderr=None,             # 子プロセスのstderrを継承してtqdmを表示
        text=True,
        check=True,
    )

    final_line = None
    for line in result.stdout.splitlines():
        if line.startswith("FINAL"):
            final_line = line
    if final_line is None:
        raise RuntimeError("FINAL 行が見つかりませんでした")

    if final_line.startswith("FINAL,"):
        parts = final_line.split(",")
        if len(parts) != 7:
            raise RuntimeError(f"FINAL 行の形式が不正です: {final_line}")
        _, iter_str, cost_str, gnorm_str, xf_str, yf_str, ystd_str = parts
    elif final_line.startswith("FINAL:"):
        payload = final_line.split(":", 1)[1]
        parts = payload.split(",")
        if len(parts) != 6:
            raise RuntimeError(f"FINAL 行の形式が不正です: {final_line}")
        iter_str, cost_str, gnorm_str, xf_str, yf_str, ystd_str = parts
    else:
        raise RuntimeError(f"FINAL 行の形式が不正です: {final_line}")
    return int(iter_str), float(cost_str), float(gnorm_str), float(xf_str), float(yf_str), float(ystd_str)


# def write_csv_row(path: str, header: str, values: List[float]) -> None:
#     with open(path, "w") as f:
#         f.write(header)
#         if values:
#             f.write("," + ",".join(str(v) for v in values))
#         f.write("\n")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py = os.path.join(script_dir, "adjoint_noise.py")
    outdir = os.path.join(script_dir, "images", "batch")
    os.makedirs(outdir, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Run adjoint_noise.py multiple times and aggregate results."
    )
    parser.add_argument(
        "-n",
        dest="n_runs",
        type=int,
        default=10,
        help="number of runs (default: 10)",
    )
    args = parser.parse_args()
    n_runs = args.n_runs
    print(f"Run {n_runs} times...", file=sys.stderr)
    
    iters: List[int] = []
    costs: List[float] = []
    gnorms: List[float] = []
    xfin: List[float] = []
    yfin: List[float] = []
    ystds: List[float] = []

    for i in range(1, n_runs + 1):
        print(f"[{i}/{n_runs}] running adjoint_noise.py", file=sys.stderr)
        itr, cst, gnm, xf, yf, ystd = run_once(sys.executable, py)
        iters.append(itr)
        costs.append(cst)
        gnorms.append(gnm)
        xfin.append(xf)
        yfin.append(yf)
        ystds.append(ystd)

    # バッチ結果を1つのCSVに統合
    csv_path = os.path.join(outdir, "batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "iter", "cost", "gnorm", "xf", "yf", "ystd"])
        for idx in range(n_runs):
            writer.writerow([idx + 1, iters[idx], costs[idx], gnorms[idx], xfin[idx], yfin[idx], ystds[idx]])
    print(f"Saved consolidated CSV to {csv_path}", file=sys.stderr)

    # 図の作成
    dpi = 144

    # 1) 最終的な x, y （復元された初期値）の散布図（範囲を自動検知して設定）
    # y_stdの値に応じてマーカー色を設定
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(xfin, yfin, c=ystds, cmap="viridis")
    plt.xlabel("x_final")
    plt.ylabel("y_final")
    plt.grid(True)
    plt.title(f"Final (x,y) over {n_runs} runs")
    # x, y の範囲を自動で少し余裕を持って設定
    if len(xfin) > 0 and len(yfin) > 0:
        x_min, x_max = min(xfin), max(xfin)
        y_min, y_max = min(yfin), max(yfin)
        x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    cbar = plt.colorbar(scatter)
    cbar.set_label("y_std")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_xy.png"), dpi=dpi)
    plt.close()

    # 2) ノイズ標準偏差 × Final cost
    plt.figure(figsize=(6, 4))
    plt.plot(ystds, costs, marker="o", linestyle="none")
    plt.xlabel("y_std")
    plt.ylabel("Final cost")
    plt.grid(True)
    plt.title("y_std vs Final cost")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ystd_cost.png"), dpi=dpi)
    plt.close()

    # 3) ノイズ標準偏差 × Gradient norm
    plt.figure(figsize=(6, 4))
    plt.plot(ystds, gnorms, marker="o", linestyle="none", color="black")
    plt.xlabel("y_std")
    plt.ylabel("Gradient norm")
    plt.grid(True)
    plt.title("y_std vs Gradient norm")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ystd_gnorm.png"), dpi=dpi)
    plt.close()

    # 4) ノイズ標準偏差 × 最終反復回数
    plt.figure(figsize=(6, 6))
    plt.plot(ystds, iters, marker="o", linestyle="none", color="black")
    plt.xlabel("y_std")
    plt.ylabel("Final iterations")
    plt.grid(True)
    plt.title("y_std vs Final iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ystd_final.png"), dpi=dpi)
    plt.close()

if __name__ == "__main__":
    main()

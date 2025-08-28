import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# forward model
def forward(dt, a, x1, y1, nmax):
    # dt: time step, a: parameters, x1: initial x, y1: initial y, nmax: number of time steps
    t = np.zeros(nmax)
    x = np.zeros(nmax)
    y = np.zeros(nmax)
    t[0] = 0.0
    x[0] = x1
    y[0] = y1
    for n in range(nmax - 1):
        t[n + 1] = t[n] + dt
        x[n + 1] = x[n] + dt * x[n] * (a[0] + a[1] * x[n] + a[2] * y[n])
        y[n + 1] = y[n] + dt * y[n] * (a[3] + a[4] * y[n] + a[5] * x[n])
    return t, x, y


if __name__ == "__main__":
    # default values
    dt = 0.01
    a = [4, -2, -4, -6, 2, 4]
    x1 = 2.0
    y1 = 1.0
    nmax = 1500
    
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "images"))

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run the forward model and save results plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--dt", type=float, default=dt, help="Time step size.")
    parser.add_argument(
        "-a", "--a", type=float, nargs=6,
        default=a,
        metavar=("a0", "a1", "a2", "a3", "a4", "a5"),
        help="Six parameters for the model."
    )
    parser.add_argument("-x", "--x1", type=float, default=x1, help="Initial value for x.")
    parser.add_argument("-y", "--y1", type=float, default=y1, help="Initial value for y.")
    parser.add_argument("-n", "--nmax", type=int, default=nmax, help="Number of time steps.")
    parser.add_argument("-o", "--outdir", type=str, default=outdir, help="Output directory for the figure.")
    args = parser.parse_args()

    # create model
    t, x, y = forward(args.dt, args.a, args.x1, args.y1, args.nmax)
    print(args)

    # save model
    os.makedirs(args.outdir, exist_ok=True)

    # 横軸をステップ、縦軸を値として x, y を同時に描画（背景透過）
    #plt.figure(facecolor='none')
    plt.figure()
    plt.plot(t, x, label='Prey1', color='blue')
    plt.plot(t, y, label='Predator1', color='red')

    # default parameters
    x1 = 1.0
    y1 = 1.0
    t, x, y = forward(args.dt, args.a, x1, y1, args.nmax)
    plt.plot(t, x, label='Prey2', color='black', linestyle='--')
    plt.plot(t, y, label='Predator2', color='gray', linestyle='--')

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    
    save_path = os.path.abspath(os.path.join(args.outdir, f"model_{args.nmax}.png"))
    #plt.savefig(save_path, transparent=True, dpi=1024)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Model saved to {save_path}")

    #---
    # phase plane
    #---
    plt.figure()
    # a1 + a2*x + a3*y = 0, a4 + a5*y + a6*x = 0 の直線（ヌルクライン）を描画
    a1 = args.a[0]
    a2 = args.a[1]
    a3 = args.a[2]
    a4 = args.a[3]
    a5 = args.a[4]
    a6 = args.a[5]

    # 軌道のx範囲に少しマージンを加えて描画範囲を決定
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_pad = 0.05 * max(1e-8, x_max - x_min)
    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    x_line = np.linspace(x_lo, x_hi, 200)

    # ヌルクライン計算（存在する場合のみ）
    y_line1 = None
    y_line2 = None
    if a3 != 0:
        y_line1 = -(a1 + a2 * x_line) / a3
        plt.plot(x_line, y_line1, 'r--', label='a1+a2*x+a3*y=0')
    else:
        print("a3が0のため、直線を描画できません。")

    if a5 != 0:
        y_line2 = -(a4 + a6 * x_line) / a5
        plt.plot(x_line, y_line2, 'g--', label='a4+a5*y+a6*x=0')
    else:
        print("a5が0のため、直線を描画できません。")

    # 軌道
    plt.plot(x, y, label='trajectory')

    # y範囲を軌道とヌルクラインの両方から決定（有限値のみを使用）
    y_arrays = [y]
    if y_line1 is not None:
        y_arrays.append(y_line1)
    if y_line2 is not None:
        y_arrays.append(y_line2)
    y_concat = np.concatenate([arr[np.isfinite(arr)] for arr in y_arrays])
    if y_concat.size == 0:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
    else:
        y_min = float(np.min(y_concat))
        y_max = float(np.max(y_concat))
    y_pad = 0.05 * max(1e-8, y_max - y_min)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    # 目盛りと凡例、軸範囲
    plt.xlabel("Prey")
    plt.ylabel("Predator")
    plt.xlim([x_lo, x_hi])
    plt.ylim([y_lo, y_hi])
    plt.legend()

    phase_save_path = os.path.abspath(os.path.join(args.outdir, f"phaseplane_{args.nmax}.png"))
    plt.savefig(phase_save_path)
    plt.close()

    print(f"Phase plane saved to {phase_save_path}")

    exit(0)
    
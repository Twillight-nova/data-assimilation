import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

def forward(dt, a, x1, y1, nmax):
    x = np.zeros(nmax)
    y = np.zeros(nmax)
    x[0] = x1
    y[0] = y1
    for n in range(nmax - 1):
        gx = a[0] + a[1] * x[n] + a[2] * y[n]
        gy = a[3] + a[4] * y[n] + a[5] * x[n]
        x[n + 1] = x[n] + dt * x[n] * gx
        y[n + 1] = y[n] + dt * y[n] * gy

    # オーバーフロー抑制のための安全上限
    # clip_val = 1.0e6
    # with np.errstate(over='ignore', invalid='ignore'):
    #     for n in range(nmax - 1):
    #         gx = a[0] + a[1] * x[n] + a[2] * y[n]
    #         gy = a[3] + a[4] * y[n] + a[5] * x[n]
    #         xn1 = x[n] + dt * x[n] * gx
    #         yn1 = y[n] + dt * y[n] * gy
    #         # 非有効値のガード
    #         if not np.isfinite(xn1):
    #             xn1 = np.sign(x[n]) * clip_val if x[n] != 0 else 0.0
    #         if not np.isfinite(yn1):
    #             yn1 = np.sign(y[n]) * clip_val if y[n] != 0 else 0.0
    #         # 上限クリップ
    #         if xn1 > clip_val:
    #             xn1 = clip_val
    #         elif xn1 < -clip_val:
    #             xn1 = -clip_val
    #         if yn1 > clip_val:
    #             yn1 = clip_val
    #         elif yn1 < -clip_val:
    #             yn1 = -clip_val
    #         x[n + 1] = xn1
    #         y[n + 1] = yn1
    return x, y

def adjoint(dt, a, x, y, xo, yo, tobs) :
    nmax = x.size
    aa = np.zeros(6)
    ax = np.zeros(nmax)
    ay = np.zeros(nmax)
    for n in reversed(range(nmax - 1)):
        if n in tobs:
            ax[n] = ax[n] + (x[n] - xo[n])
            ay[n] = ay[n] + (y[n] - yo[n])
        ax[n] = ax[n] + dt * a[5] * y[n] * ay[n+1]
        ay[n] = ay[n] + dt * a[4] * y[n] * ay[n+1]
        ay[n] = ay[n] + (1 + dt * (a[3] + a[4] * y[n] + a[5] * x[n])) * ay[n+1]
        ay[n] = ay[n] + dt * a[2] * x[n] * ax[n+1]
        ax[n] = ax[n] + dt * a[1] * x[n] * ax[n+1]
        ax[n] = ax[n] + (1 + dt * (a[0] + a[1] * x[n] + a[2] * y[n])) * ax[n+1]
    return np.array([ax[0], ay[0]])

# 観測関数
def obs_fn(a, dt, nmax, x1, y1, tobs):
    x, y = forward(dt, a, x1, y1, nmax)
    # ランダムノイズ
    # x = x + np.random.normal(0, 0.1, nmax)
    # y = y + np.random.normal(0, 0.1, nmax)
    
    # 正弦波ノイズを加える
    #x = x + 0.05 * np.sin(np.linspace(0, 10 * np.pi, nmax))
    # ランダムノイズを加える
    y = y + np.random.normal(0, 0.05, nmax)
    # yの標準偏差を計算
    y_std = np.std(y)
    return x, y, y_std

# 勾配
def gr(par, dt, nmax, xo, yo, tobs):
    x, y = forward(dt, par[0:6], par[6], par[7], nmax)
    hist["par"].append(par.copy())
    cost = calc_cost(x[tobs], y[tobs], xo[tobs], yo[tobs])
    hist["cost"].append(cost)
    grad = adjoint(dt, par[0:6], x, y, xo, yo, tobs)
    hist["gnorm"].append(np.sqrt(np.dot(grad, grad)))
    return grad

# コスト函数
def calc_cost(xf, yf, xo, yo):
    # 2ノルム（二乗誤差）の場合
    # return np.sum((xf - xo) ** 2) + np.sum((yf - yo) ** 2)

    # 1ノルム（絶対値誤差）の場合
    # return np.sum(np.abs(xf - xo)) + np.sum(np.abs(yf - yo))

    # 無次元化した平均二乗誤差（MSE）の場合
    # return 0.5 * (np.mean((xf - xo) ** 2) + np.mean((yf - yo) ** 2))

    # 最大誤差（∞ノルム）の場合
    # return max(np.max(np.abs(xf - xo)), np.max(np.abs(yf - yo)))

    # ロバストなHuber損失の例
    # delta = 1.0
    # def huber(r, delta):
    #     return np.where(np.abs(r) < delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    # return np.sum(huber(xf - xo, delta)) + np.sum(huber(yf - yo, delta))

    # デフォルト（2ノルム）
    return np.sum((xf - xo) ** 2) + np.sum((yf - yo) ** 2)

def create_phase_plot(xt, yt, dpi, lw=1.0):
    fig, ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('tab10')
    ax.plot(xt,yt,c='black',lw=lw,label='nature')
    ax.plot(xt[0],yt[0],c='black',marker='*',ms=10)
    return fig, ax, cmap

def plot_iter_trajectory(ax, cmap, par, dt, nmax, niter, tobs, nplot, maxplot, tplot_state):
    if maxplot == 0:
        maxplot = niter
    if niter % nplot == 0 and len(tplot_state) < maxplot:
        x, y = forward(dt, par[0:6], par[6], par[7], nmax)
        ax.plot(x[tobs], y[tobs], ls='--', c=cmap(len(tplot_state)))
        ax.plot(x[0], y[0], c=cmap(len(tplot_state)), marker='*')
        tplot_state.append(niter)

def finalize_phase_plot(fig, ax, par, dt, nmax, dpi, linewidth=2.0, filename="phase.png"):
    x, y = forward(dt, par[0:6], par[6], par[7], nmax)
    ax.plot(x,y,c='r',ls='--',lw=linewidth,label='final')
    ax.plot(x[0],y[0],c='r',marker='*',ms=10)
    ax.plot([0.0,2.0],[1.0,0.0],ls=':',c='k',zorder=0)
    ax.plot([0.5,1.5],[2.0,0.0],ls=':',c='k',zorder=0)
    ax.set_xlim(0.0,2.0)
    ax.set_ylim(0.0,2.0)
    ax.grid()
    ax.set_xlabel('Prey')
    ax.set_ylabel('Predator')
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def plot_cost_history(hist, alg, dpi, fig_size=(9, 4.5), linewidth=1.0, filename="cost.png") :
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(np.log10(hist["cost"]), lw=linewidth)
    ax.set_title(f"cost {alg}", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"log$_{10}$ J", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def plot_gnorm_history(hist, alg, dpi, fig_size=(9, 4.5), linewidth=1.0, filename="gnorm.png"):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(np.log10(hist["gnorm"]), lw=linewidth, color = "black")
    ax.set_title(f"gnorm {alg}", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"log$_{10}$|g|", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def plot_init_history(hist, alg, dpi, fig_size=(9, 4.5), linewidth=1.0, filename="init.png"):
    par = np.array(hist["par"])
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(par[:, 6], label=r"x$_0$", lw=linewidth, color="black")
    ax.plot(par[:, 7], label=r"y$_0$", lw=linewidth, color="red")
    ax.set_title(f"init {alg}", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel("Initial conditions", fontsize=15)
    ax.set_ylim(0, 2)
    ax.legend(loc="upper right", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    nmax = 3001
    dt = 0.001
    at = [4, -2, -4, -6, 2, 4]
    x1, y1 = 1, 1
    lw = 1.0
    test = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, "images", "orig1")
    os.makedirs(outdir, exist_ok=True)

    # 設定ファイルの読み込み（-f/--config）
    parser = argparse.ArgumentParser(
        description="adjoint method demo with optional JSON config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-f", "--config", type=str, default=None, help="JSON設定ファイルのパス")
    args = parser.parse_args()

    cfg = {}
    if args.config is not None:
        config_path = args.config
        # カレントで見つからなければスクリプトディレクトリを試す
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            alt_path = os.path.join(script_dir, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # デフォルト値を上書き
        dt = float(cfg.get("dt", dt))
        nmax = int(cfg.get("nmax", nmax))
        x1 = float(cfg.get("x1", x1))
        y1 = float(cfg.get("y1", y1))
        at = cfg.get("a", at)

    # 観測時刻 tobs の設定（設定ファイルがあれば優先）
    if "tobs" in cfg:
        raw_tobs = cfg.get("tobs")
        if isinstance(raw_tobs, list):
            tobs = np.array(raw_tobs, dtype=int)
        else:
            tobs = np.array([int(raw_tobs)], dtype=int)
        # 範囲外の値は除去
        tobs = tobs[(tobs >= 0) & (tobs < nmax)]
        if tobs.size == 0:
            tobs = np.arange(1, nmax, 10)
    else:
        # ランダムな個数、ランダムな値を昇順で配列生成（必要なら復活）
        # num_tobs = np.random.randint(1, nmax//2)
        # tobs = np.sort(np.random.choice(np.arange(1, nmax), size=num_tobs, replace=False))
        tobs = np.arange(1, nmax, 10)
    
    tobs = np.arange(1, nmax, 10)

    # tobsの値とその配列数を可視化して保存
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(tobs)), tobs, marker='o', linestyle='none', color='blue')
    plt.title("tobs indices and values")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tobs.png"), dpi=144)
    plt.close()
    
    try:
        xt, yt, y_std = obs_fn(at, dt, nmax, x1, y1, tobs) # 観測値
        xo, yo = xt, yt
    except Exception as e:
        print("obs_fnの実行中にエラーが発生しました:", e)
        xt, yt = None, None
        xo, yo = None, None
     
    #a = [1, 0, 0, -1, 0, 0] # 不正確なパラメータ
    a = at.copy()
    x1, y1 = 1.5, 1.5 # 初期条件
    par = np.concatenate([a, [x1, y1]]) # 固定パラメータ

    alg = "steepest descent" # 最適化アルゴリズム
    maxiter = 2000 # 最大反復回数
    alpha = 1.0e-3 # 学習率
    gtol = 1.0e-10 # 勾配の収束条件
    hist = {"cost":[], "gnorm":[], "par":[]} # 履歴

    dpi = 144 # 画像の解像度
    fig, ax, cmap = create_phase_plot(xt, yt, dpi, lw) # 初期軌道を描画
    
    niter = 0
    tplot_state = []
    nplot = 10
    maxplot = 100
    while (niter < maxiter):
        # 進捗バーを表示するためにtqdmを使う（importはファイル先頭で行うこと）
        if niter == 0:
            try:
                pbar = tqdm(total=maxiter, desc="最適化進捗", ncols=80, leave=False)
            except ImportError:
                pbar = None
        if 'pbar' in locals() and pbar is not None:
            pbar.update(1)
        # 目的関数を評価し、随伴法で勾配 g = [∂J/∂x0, ∂J/∂y0] を計算
        try:
            grad = gr(par, dt, nmax, xo, yo, tobs) # 勾配を計算
        except Exception as e:
            print("勾配の計算中にエラーが発生しました:", e)
            break
        #print(grad)
        #print("{} iterations, cost = {:.4e}, gradient norm = {:.4e}"\
        #    .format(niter,hist["cost"][-1],hist["gnorm"][-1]))
        #print("x, y = {}, {}"\
        #    .format(hist["par"][-1][6],hist["par"][-1][7]))
        plot_iter_trajectory(ax, cmap, par, dt, nmax, niter, tobs, nplot, maxplot, tplot_state) # 軌道を描画
        # 収束条件を満たしたら終了
        if hist["gnorm"][-1] < gtol:
            print("Convergence: {} iterations".format(niter))
            print("Final cost = {:.4e}, gradient norm = {:.4e}"\
                .format(hist["cost"][-1],hist["gnorm"][-1]))
            print("x, y = {}, {}"\
                .format(hist["par"][-1][6],hist["par"][-1][7]))
            break
        # 補正
        par[6:8] = par[6:8] - alpha*grad # インデックスが6と7のパラメータを更新
        niter += 1
    if 'pbar' in locals() and pbar is not None:
        pbar.close()

    finalize_phase_plot(fig, ax, par, dt, nmax, dpi, linewidth=lw, filename=os.path.join(outdir, "phase.png"))
    
    fig_size = (9, 4.5)

    plot_cost_history(hist, alg, dpi, fig_size=fig_size, linewidth=lw, filename=os.path.join(outdir, "cost.png"))

    plot_gnorm_history(hist, alg, dpi, fig_size=fig_size, linewidth=lw, filename=os.path.join(outdir, "gnorm.png"))

    plot_init_history(hist, alg, dpi, fig_size=fig_size, linewidth=lw, filename=os.path.join(outdir, "init.png"))

    # バッチ用: 最終結果を機械可読なCSV形式で標準出力
    # 最適化：より簡潔かつ例外処理不要な形に変更
    if len(hist["par"]) > 0:
        xf = hist["par"][-1][6]
        yf = hist["par"][-1][7]
    else:
        xf = float('nan')
        yf = float('nan')
    final_cost = hist["cost"][-1] if len(hist["cost"]) > 0 else float('nan')
    final_gnorm = hist["gnorm"][-1] if len(hist["gnorm"]) > 0 else float('nan')
    print(f"FINAL:{niter},{final_cost},{final_gnorm},{xf},{yf},{y_std}")


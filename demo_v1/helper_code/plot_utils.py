from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_straddle_payoff(K, call_premium, put_premium,
                         S_T = None, xlim = (50,150), path = None, short = False):

    # --- Underlying price range ---
    S_grid = np.linspace(xlim[0], xlim[1], 1000)

    # --- Expiry P/L (long positions) ---
    if short:
        cValu = call_premium - np.maximum(S_grid - K, 0)
        pValu  = put_premium  - np.maximum(K - S_grid, 0)
        straddleValu = cValu + pValu
    else:
        cValu = np.maximum(S_grid - K, 0) - call_premium
        pValu  = np.maximum(K - S_grid, 0) - put_premium
        straddleValu = cValu + pValu

    # --- Plot ---
    plt.plot(S_grid, cValu, label="Long Call (expiry P/L)", linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(S_grid, pValu,  label="Long Put (expiry P/L)", linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(S_grid, straddleValu, label="Long Straddle = Call + Put", linewidth=2, color='C0')

    if S_T is not None:
        plt.axvline(S_T, linewidth=0.7, color='red', label=f"$S_T$ = {S_T:.2f}")

    # Reference lines
    plt.axhline(0, linewidth=1, color='black')

    # Lock axis scale to 1:1
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title("Straddle Expiry Profit and Loss")
    plt.xlabel("Underlying Price at Expiry")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # save figure to path (BEFORE show)
    if path is not None:
        plt.savefig(path, dpi=600)
    plt.show()



def plot_call_butterfly_payoff(K1, K2, K3, c1, c2, c3,
                               S_T = None, xlim = (50,150), path = None, short = False):

    # --- Underlying price range ---
    S_grid = np.linspace(xlim[0], xlim[1], 1000)

    # --- Expiry P/L components ---
    if short:
        call_low = c1 - np.maximum(S_grid - K1, 0)
        call_mid = c2 - np.maximum(S_grid - K2, 0)
        call_high = c3 - np.maximum(S_grid - K3, 0)
        butterfly_pl = -call_low + 2 * call_mid - call_high
    else:
        call_low = np.maximum(S_grid - K1, 0) - c1
        call_mid = np.maximum(S_grid - K2, 0) - c2
        call_high = np.maximum(S_grid - K3, 0) - c3
        butterfly_pl = call_low - 2 * call_mid + call_high

    # --- Plot ---
    plt.plot(S_grid, call_low, label=f"{'Short' if short else 'Long'} Call", linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(S_grid, -2 * call_mid if not short else 2 * call_mid, label=f"{'Long' if short else 'Short'} 2 Calls", linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(S_grid, call_high, label=f"{'Short' if short else 'Long'} Call", linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(S_grid, butterfly_pl, label=f"{'Short' if short else 'Long'} Call Butterfly", linewidth=2, color='C0')

    if S_T is not None:
        plt.axvline(S_T, linewidth=0.7, color='red', label=f"$S_T$ = {S_T:.2f}")

    # Reference lines
    plt.axhline(0, linewidth=1, color='black')

    # Lock axis scale to 1:1
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title("Call Butterfly Expiry Profit and Loss")
    plt.xlabel("Underlying Price at Expiry")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # save figure to path (BEFORE show)
    if path is not None:
        plt.savefig(path, dpi=600)
    plt.show()
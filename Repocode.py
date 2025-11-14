#!/usr/bin/env python3
"""
Reproducibility code for:

- Figure 1:  λ_min(A) vs. graph size for paths / cycles / grids with varying m
- Figure 2:  Gradient descent convergence on a 50-node grid
- Figure 3:  Mode-wise dynamic response for a cycle graph

All computations are done for scalar node values (d = 1) with
uniform node penalty M_i = m I, so A = L + m I.

Usage
-----
    python reproduce_figures.py

This will write three PNGs in the current directory:

    fig1_lambda_vs_n.png
    fig2_gd_grid50.png
    fig3_modes_cycle.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Graph Laplacian constructors
# ---------------------------------------------------------------------


def laplacian_path(n: int) -> np.ndarray:
    """Laplacian of a path graph P_n (n nodes, line)."""
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:
            L[i, i - 1] = -1.0
            L[i, i] += 1.0
        if i < n - 1:
            L[i, i + 1] = -1.0
            L[i, i] += 1.0
    return L


def laplacian_cycle(n: int) -> np.ndarray:
    """Laplacian of a cycle graph C_n (n nodes, ring)."""
    L = laplacian_path(n)
    # Add wrap-around edge between 0 and n-1
    L[0, n - 1] = L[n - 1, 0] = -1.0
    L[0, 0] += 1.0
    L[n - 1, n - 1] += 1.0
    return L


def laplacian_grid(n_rows: int, n_cols: int) -> np.ndarray:
    """
    Laplacian of a 2D Cartesian grid graph with n_rows * n_cols nodes.
    Grid is 4-nearest-neighbour, no periodic boundaries.

    Node indexing: row-major (i, j) -> i * n_cols + j.
    """
    N = n_rows * n_cols
    L = np.zeros((N, N), dtype=float)

    def idx(i, j):
        return i * n_cols + j

    for i in range(n_rows):
        for j in range(n_cols):
            k = idx(i, j)
            # Up
            if i > 0:
                k2 = idx(i - 1, j)
                L[k, k2] = -1.0
                L[k, k] += 1.0
            # Down
            if i < n_rows - 1:
                k2 = idx(i + 1, j)
                L[k, k2] = -1.0
                L[k, k] += 1.0
            # Left
            if j > 0:
                k2 = idx(i, j - 1)
                L[k, k2] = -1.0
                L[k, k] += 1.0
            # Right
            if j < n_cols - 1:
                k2 = idx(i, j + 1)
                L[k, k2] = -1.0
                L[k, k] += 1.0
    return L


# ---------------------------------------------------------------------
# Figure 1: λ_min(A) vs. graph size
# ---------------------------------------------------------------------


def figure1_lambda_vs_n(outfile: str = "fig1_lambda_vs_n.png") -> None:
    """
    Figure 1:
    Plot λ_min(A) vs. graph size for paths, cycles, and grids,
    for several values of m.

    For grids, we use square grids with side length s, so the number
    of nodes is N = s^2. We plot λ_min(A) against N.
    """
    m_values = [1.0, 2.0, 3.0, 4.0]

    # Path / cycle sizes (number of nodes)
    n_values = np.arange(10, 51, 10)  # 10, 20, 30, 40, 50

    # Grid sizes: 4x4, 6x6, 8x8, 10x10, 12x12
    grid_sides = np.array([4, 6, 8, 10, 12])
    grid_nodes = grid_sides**2

    fig, ax = plt.subplots(figsize=(6, 4))

    for m in m_values:
        # Paths
        lam_min_path = []
        for n in n_values:
            L = laplacian_path(n)
            A = L + m * np.eye(n)
            vals = np.linalg.eigvalsh(A)
            lam_min_path.append(vals[0])
        lam_min_path = np.array(lam_min_path)
        ax.plot(
            n_values,
            lam_min_path,
            "-",
            label=f"path, m={m:g}",
        )

        # Cycles
        lam_min_cycle = []
        for n in n_values:
            L = laplacian_cycle(n)
            A = L + m * np.eye(n)
            vals = np.linalg.eigvalsh(A)
            lam_min_cycle.append(vals[0])
        lam_min_cycle = np.array(lam_min_cycle)
        ax.plot(
            n_values,
            lam_min_cycle,
            "--",
            label=f"cycle, m={m:g}",
        )

        # Grids (square)
        lam_min_grid = []
        for s in grid_sides:
            L = laplacian_grid(s, s)
            N = s * s
            A = L + m * np.eye(N)
            vals = np.linalg.eigvalsh(A)
            lam_min_grid.append(vals[0])
        lam_min_grid = np.array(lam_min_grid)
        ax.plot(
            grid_nodes,
            lam_min_grid,
            ":",
            label=f"grid (s×s), m={m:g}",
        )

    ax.set_xlabel("Graph size (number of nodes)")
    ax.set_ylabel(r"$\lambda_{\min}(A)$")
    ax.set_title(r"$\lambda_{\min}(A)$ vs. graph size for paths / cycles / grids")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: Gradient descent convergence on a 50-node grid
# ---------------------------------------------------------------------


def figure2_gradient_descent(outfile: str = "fig2_gd_grid50.png") -> None:
    """
    Figure 2:
    Gradient descent convergence on a 50-node grid.

    We build a 5×10 grid (N=50), define A = L + m I, set a random b,
    and run gradient descent on f(x) = 0.5 x^T A x - b^T x.

    We compare the observed error norm with the theoretical rate from §7.1:
        ρ = (κ - 1) / (κ + 1),
    using the optimal step size α* = 2 / (λ_max + λ_min).
    """

    np.random.seed(0)

    # 5x10 grid: N = 50 nodes
    n_rows, n_cols = 5, 10
    L = laplacian_grid(n_rows, n_cols)
    N = n_rows * n_cols
    m = 1.0
    A = L + m * np.eye(N)

    # Eigenvalues for condition number and optimal step size
    evals = np.linalg.eigvalsh(A)
    lam_min = evals[0]
    lam_max = evals[-1]
    kappa = lam_max / lam_min

    alpha_star = 2.0 / (lam_min + lam_max)
    rho = (kappa - 1.0) / (kappa + 1.0)

    # Random quadratic: f(x) = 0.5 x^T A x - b^T x
    b = np.random.randn(N)

    # True minimizer
    x_star = np.linalg.solve(A, b)

    # Gradient descent iterations
    max_iter = 200
    x = np.zeros_like(b)
    errors = []

    for k in range(max_iter + 1):
        err = np.linalg.norm(x - x_star)
        errors.append(err)
        grad = A.dot(x) - b
        x = x - alpha_star * grad

    errors = np.array(errors)
    # Predicted rate: ||e_k|| <= ρ^k ||e_0||
    predicted = errors[0] * rho ** np.arange(max_iter + 1)

    # Plot (log scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    iters = np.arange(max_iter + 1)
    ax.semilogy(iters, errors, label="Observed rate")
    ax.semilogy(iters, predicted, "--", label="Predicted rate")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Error norm (log scale)")
    ax.set_title("Gradient descent on a 50-node grid")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: Mode-wise dynamic response for a cycle graph
# ---------------------------------------------------------------------


def figure3_modewise_cycle(outfile: str = "fig3_modes_cycle.png") -> None:
    """
    Figure 3:
    Mode-wise dynamic response for a cycle graph.

    Here we use *gradient flow* (first-order system)
        dx/dt = -A x,
    with A = L + m I on a cycle graph, rather than the second-order
    ODE. In the eigenbasis of A, each mode decays as e^{-λ t}.

    We:
      - choose initial conditions aligned with eigenmodes 1, 2, 3,
      - integrate dx/dt = -A x numerically with forward Euler,
      - compare |mode amplitude| with the theoretical envelope e^{-λ t}.
    """

    n = 20  # number of nodes in the cycle graph
    L = laplacian_cycle(n)
    m = 1.0
    A = L + m * np.eye(n)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(A)  # evals sorted ascending
    # Select the first three modes beyond the constant one
    # (mode 0 is constant; we take 1, 2, 3)
    mode_indices = [1, 2, 3]

    # Time discretization
    T = 20.0
    dt = 0.01
    num_steps = int(T / dt) + 1
    times = np.linspace(0.0, T, num_steps)

    # For numerical solution, we use forward Euler:
    #   x_{k+1} = x_k - dt * A x_k
    # This is stable when dt * λ_max < 2.
    lam_max = evals[-1]
    if dt * lam_max >= 2.0:
        raise RuntimeError(
            "Time step too large for stable forward Euler; "
            "reduce dt or n."
        )

    # For each mode, simulate and compare to e^{-λ t}
    fig, axes = plt.subplots(
        1, len(mode_indices), figsize=(10, 3), sharey=True
    )

    for ax, idx in zip(axes, mode_indices):
        lam = evals[idx]
        v = evecs[:, idx]

        # Normalize mode vector for clarity
        v = v / np.linalg.norm(v)

        # Initial condition: x(0) = v
        x = v.copy()
        mode_amp = np.zeros(num_steps)
        mode_amp[0] = np.abs(v @ x)

        # Time stepping
        for k in range(1, num_steps):
            x = x - dt * A.dot(x)
            mode_amp[k] = np.abs(v @ x)

        # Theoretical continuous-time decay of this mode (gradient flow)
        theory = np.exp(-lam * times)

        ax.semilogy(times, mode_amp, label=r"$|x_k(t)|$")
        ax.semilogy(times, theory, "--", label=r"$e^{-\lambda_k t}$")
        ax.set_title(f"Eigenmode {idx}")
        ax.set_xlabel(r"$t$")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Mode amplitude (log scale)")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Mode-wise dynamic response for a cycle graph", y=1.05)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    figure1_lambda_vs_n()
    figure2_gradient_descent()
    figure3_modewise_cycle()
    print("Figures saved as:")
    print("  fig1_lambda_vs_n.png")
    print("  fig2_gd_grid50.png")
    print("  fig3_modes_cycle.png")


if __name__ == "__main__":
    main()

# module main.py
"""Uses the Rydberg-Klein-Rees method to obtain the internuclear potential V(r)."""

# Copyright (C) 2025-2026 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.sparse import diags_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

plt.style.use(["science", "grid"])
plt.rcParams.update({"font.size": 18})

# Equal to 0.5 * ћ^2 in the appropriate units. Given in "RKR1" by LeRoy.
hbar2_over_2: float = 16.857629206  # [amu * Å^2 * cm^-1]

m_oxygen: float = 15.999  # [amu]

mass: float = (m_oxygen * m_oxygen) / (m_oxygen + m_oxygen)

# Constants for O2 from the NIST Chemistry WebBook.

# [T_e, ω_e, ω_ex_e, ω_ey_e, ...]
g_consts_up: list[float] = [49793.28, 709.31, -10.65, -0.139]
# [B_e, α_e, γ_e, ...]
b_consts_up: list[float] = [0.81902, -0.01206, -5.56e-4]

g_consts_lo: list[float] = [0, 1580.193, -11.981, 0.04747]
b_consts_lo: list[float] = [1.4376766, -0.01593]


def g(v: int | float, g_consts: list[float]) -> float:
    x = v + 0.5

    return sum(val * x**idx for idx, val in enumerate(g_consts))


def b(v: int | float, b_consts: list[float]) -> float:
    x = v + 0.5

    return sum(val * x**idx for idx, val in enumerate(b_consts))


def weight_fn(v_prime: float, v: int, g_consts: list[float]):
    # The weight function proposed by Tellinghuisen.
    # w(v') = sqrt(v - v') / sqrt(G(v) - G(v'))
    return math.sqrt(v - v_prime) / math.sqrt(g(v, g_consts) - g(v_prime, g_consts))


def f_integral_weighted(v: int, v_min: float, g_consts: list[float]) -> float:
    # f(v) = 2 * sqrt{hbar^2 / 2μ} * ∫_0^sqrt{v - v_min} w(v - u^2) du
    upper_bound = math.sqrt(v - v_min)

    def integrand_f(u: float) -> float:
        # u = sqrt(v - v')
        v_prime = v - u**2
        return weight_fn(v_prime, v, g_consts)

    return 2.0 * math.sqrt(hbar2_over_2 / mass) * quad(integrand_f, 0.0, upper_bound)[0]


def g_integral_weighted(
    v: int, v_min: float, g_consts: list[float], b_consts: list[float]
) -> float:
    # g(v) = 2 * sqrt{2μ / hbar^2} * ∫_0^sqrt{v - v_min} B(v - u^2) * w(v - u^2) du
    upper_bound = math.sqrt(v - v_min)

    def integrand_g(u: float) -> float:
        # u = sqrt(v - v')
        v_prime = v - u**2
        return b(v_prime, b_consts) * weight_fn(v_prime, v, g_consts)

    return 2.0 * math.sqrt(mass / hbar2_over_2) * quad(integrand_g, 0.0, upper_bound)[0]


def rkr(v: int, g_consts: list[float], b_consts: list[float]) -> tuple[float, float]:
    v_min = -0.5

    # With epsabs=1e-12 and epsrel=1e-12, it takes between 300 and 600 evaluations for the
    # non-weighted integrals to converge. With the same tolerances, the weighted integrals take only
    # around 20 iterations.
    f = f_integral_weighted(v, v_min, g_consts)
    g = g_integral_weighted(v, v_min, g_consts, b_consts)

    sqrt_term = math.sqrt(f**2 + f / g)

    r_min = sqrt_term - f  # [Å]
    r_max = sqrt_term + f  # [Å]

    return r_min, r_max


def radial_schrodinger(
    r: NDArray[np.float64], v_max: int, potential_term: NDArray, dim: int
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    dr: float = r[1] - r[0]

    # Construct the kinetic energy operator via a second-order central finite difference. A sparse
    # array is used to save space.
    kinetic_term: NDArray[np.float64] = (-hbar2_over_2 / (mass * dr**2)) * diags_array(
        [1.0, -2.0, 1.0],
        offsets=[-1, 0, 1],  # pyright: ignore[reportArgumentType]
        shape=(dim, dim),
    ).toarray()

    hamiltonian = kinetic_term + np.diag(potential_term)

    # The Hamiltonian will always be Hermitian, so the use of eigh is warranted here.
    eigvals, eigvecs = eigh(hamiltonian)

    norm_wavefns: list[NDArray[np.float64]] = []

    # Normalize the wavefunctions ψ(r) such that ∫ ψ'ψ dr = 1.
    for i in range(v_max):
        wavefn: NDArray[np.float64] = eigvecs[:, i]
        norm: float = float(simpson(wavefn**2, r))
        norm_wavefns.append(wavefn / math.sqrt(norm))

    return eigvals[:v_max], norm_wavefns


def plot_extrapolation(
    fit_fn: Callable, xdata: NDArray[np.float64], ydata: NDArray[np.float64], fit_type: str
) -> NDArray[np.float64]:
    params, _, info, _, _ = curve_fit(fit_fn, xdata, ydata, maxfev=100000, full_output=True)
    print(f"The {fit_type} fit took {info['nfev']} iterations.")

    return params


def extrapolate_inner(
    rkr_sorted: NDArray[np.float64], energies_sorted: NDArray[np.float64], fn_type: str = "exp"
) -> tuple[NDArray[np.float64], Callable]:
    inner_points: NDArray[np.float64] = rkr_sorted[0:3]
    inner_energy: NDArray[np.float64] = energies_sorted[0:3]

    # LeRoy's LEVEL extrapolates the potential inward with an exponential function fitted to the
    # first three points.
    def fit(x, a, b):
        match fn_type:
            case "exp":
                return a * np.exp(-b * x)

            case "inv":
                return a / x**b

    params: NDArray[np.float64] = plot_extrapolation(fit, inner_points, inner_energy, "inner")

    return params, fit


def extrapolate_outer(
    rkr_sorted: NDArray[np.float64],
    energies_sorted: NDArray[np.float64],
    g_consts: list[float],
    fn_type: str = "inv",
) -> tuple[NDArray[np.float64], Callable]:
    outer_points: NDArray[np.float64] = rkr_sorted[-3:]
    outer_energy: NDArray[np.float64] = energies_sorted[-3:]

    # The dissociation limit given in Herzberg is D_e = ω_e^2 / (4ω_ex_e), but I'm not sure how
    # accurate this is when the potential is solved to high vibrational quantum numbers.
    dissociation: float = g_consts[1] ** 2 / (4 * abs(g_consts[2]))

    # All three of these fit functions are given in the documentation for LeRoy's LEVEL.
    def fit(x, a, b, c):
        match fn_type:
            case "exp":
                return dissociation - a * np.exp(-b * (x - c) ** 2)
            case "inv":
                return dissociation - a / x**b
            case "mix":
                return dissociation - a * x**b * np.exp(-c * x)

    params: NDArray[np.float64] = plot_extrapolation(fit, outer_points, outer_energy, "outer")

    return params, fit


def get_bounds(
    v_max: int, g_consts: list[float], b_consts: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rkr_mins: NDArray[np.float64] = np.empty(v_max)
    rkr_maxs: NDArray[np.float64] = np.empty(v_max)
    energies: NDArray[np.float64] = np.empty(v_max)

    for v in range(0, v_max):
        r_min, r_max = rkr(v, g_consts, b_consts)

        rkr_mins[v] = r_min
        rkr_maxs[v] = r_max
        energies[v] = g(v, g_consts)

    plt.scatter(rkr_mins, energies)
    plt.scatter(rkr_maxs, energies)

    return rkr_mins, rkr_maxs, energies


def get_potential(
    r: NDArray[np.float64],
    rkr_mins: NDArray[np.float64],
    rkr_maxs: NDArray[np.float64],
    energies: NDArray[np.float64],
    g_consts: list[float],
    j_qn: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rkr_all: NDArray[np.float64] = np.concatenate((rkr_mins, rkr_maxs))
    energies_all: NDArray[np.float64] = np.concatenate((energies, energies))

    # The x values in CubicSpline must be listed in increasing order, so sort to ensure this.
    sorted_indices: NDArray[np.int64] = np.argsort(rkr_all)

    rkr_sorted: NDArray[np.float64] = rkr_all[sorted_indices]
    energies_sorted: NDArray[np.float64] = energies_all[sorted_indices]

    cubic_spline: CubicSpline = CubicSpline(rkr_sorted, energies_sorted)

    params_inner, fit_inner = extrapolate_inner(rkr_sorted, energies_sorted)
    params_outer, fit_outer = extrapolate_outer(rkr_sorted, energies_sorted, g_consts)

    rkr_min: float = rkr_sorted[0]
    rkr_max: float = rkr_sorted[-1]

    lmask: NDArray[np.bool] = r < rkr_min
    mmask: NDArray[np.bool] = (r >= rkr_min) & (r <= rkr_max)
    rmask: NDArray[np.bool] = r > rkr_max

    potential: NDArray[np.float64] = np.empty_like(r)

    potential[lmask] = fit_inner(r[lmask], *params_inner)
    potential[mmask] = cubic_spline(r[mmask])
    potential[rmask] = fit_outer(r[rmask], *params_outer)

    # Add centrifugal J dependence to the rotationless potential.
    potential += (hbar2_over_2 / (mass * r**2)) * j_qn * (j_qn + 1)

    plt.plot(r[lmask], potential[lmask], color="blue")
    plt.plot(r[mmask], potential[mmask], color="black")
    plt.plot(r[rmask], potential[rmask], color="red")

    return r, potential


def main() -> None:
    v_max_up: int = 16
    v_max_lo: int = 19
    j_qn_up: int = 0
    j_qn_lo: int = 0

    dim: int = 1000

    rkr_mins_up, rkr_maxs_up, energies_up = get_bounds(v_max_up, g_consts_up, b_consts_up)
    rkr_mins_lo, rkr_maxs_lo, energies_lo = get_bounds(v_max_lo, g_consts_lo, b_consts_lo)

    r_min: float = min(rkr_mins_up.min(), rkr_mins_lo.min())
    r_max: float = max(rkr_maxs_up.max(), rkr_maxs_lo.max())

    r: NDArray[np.float64] = np.linspace(r_min, r_max, dim)

    r_up, potential_up = get_potential(
        r, rkr_mins_up, rkr_maxs_up, energies_up, g_consts_up, j_qn_up
    )
    r_lo, potential_lo = get_potential(
        r, rkr_mins_lo, rkr_maxs_lo, energies_lo, g_consts_lo, j_qn_lo
    )

    eigvals_up, wavefns_up = radial_schrodinger(r_up, v_max_up, potential_up, dim)
    eigvals_lo, wavefns_lo = radial_schrodinger(r_lo, v_max_lo, potential_lo, dim)

    scaling_factor: int = 500

    for i, psi in enumerate(wavefns_up):
        plt.plot(r_up, psi * scaling_factor + eigvals_up[i])

    for i, psi in enumerate(wavefns_lo):
        plt.plot(r_lo, psi * scaling_factor + eigvals_lo[i])

    plt.xlabel(r"Internuclear Distance, $r$ [$\AA$]")
    plt.ylabel(r"Energy, $E$ [cm$^{-1}$]")
    plt.show()

    # Compute Franck-Condon factors and compare with known data from Cheung.
    fcfs: NDArray[np.float64] = np.zeros((v_max_up, v_max_lo))

    for i in range(v_max_up):
        for j in range(v_max_lo):
            fcfs[i][j] = np.abs(simpson(wavefns_up[i] * wavefns_lo[j], r_up)) ** 2

    cheung: NDArray[np.float64] = np.genfromtxt("../data/cheung.csv", delimiter=",")

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Simulation")
    axs[0].imshow(fcfs, origin="lower")

    axs[1].set_title("Cheung")
    im = axs[1].imshow(cheung, origin="lower")

    for ax, data in zip(axs, [fcfs, cheung]):
        x_range: range = range(data.shape[1])
        y_range: range = range(data.shape[0])
        ax.set_xticks(x_range, labels=x_range)
        ax.set_yticks(y_range, labels=y_range)
        ax.set(xlabel="$v''$", ylabel="$v'$")

    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, label="Overlap Integral")
    plt.show()


if __name__ == "__main__":
    main()

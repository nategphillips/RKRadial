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

# Equal to 0.5 * ћ^2 [amu * Å^2 * cm^-1]. Given in "RKR1" by LeRoy.
hbar2_over_2 = 16.857629206
# Avodagro constant [1/mol]
avogadro = 6.02214076e23
# Mass of oxygen [amu].
m_oxygen = 15.999
# Total molecular charge.
charge = 0
# Mass of electron [amu].
m_e = 9.1093837139e-31 * 1e3 * avogadro
# Charge-modified reduced mass [amu].
reduced_mass = (m_oxygen * m_oxygen) / (m_oxygen + m_oxygen - charge * m_e)

# Constants for the Schumann-Runge transition of O2 from the NIST Chemistry WebBook.

# Vibrational constants for G(v): [T_e, ω_e, ω_ex_e, ω_ey_e, ...]
g_consts_up = [49793.28, 709.31, -10.65, -0.139]
g_consts_lo = [0, 1580.193, -11.981, 0.04747]
# Rotational constants for B(v): [B_e, α_e, γ_e, ...]
b_consts_up = [0.81902, -0.01206, -5.56e-4]
b_consts_lo = [1.4376766, -0.01593]


def g(v_qn: int | float, g_consts: list[float]) -> float:
    """Return the vibrational term value G(v) for a given vibrational quantum number.

    Args:
        v_qn: Vibrational quantum number v.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].

    Returns:
        The vibrational term value G(v).
    """
    x = v_qn + 0.5

    return sum(val * x**idx for idx, val in enumerate(g_consts))


def b(v_qn: int | float, b_consts: list[float]) -> float:
    """Return the rotational term value B(v) for a given vibrational quantum number.

    Args:
        v_qn: Vibrational quantum number v.
        b_consts: Rotational constants [B_e, α_e, γ_e, ...].

    Returns:
        The rotational term value B(v)
    """
    x = v_qn + 0.5

    return sum(val * x**idx for idx, val in enumerate(b_consts))


def weight_fn(v_prime: float, v: int, g_consts: list[float]):
    """The weight function proposed by Tellinghuisen and given in Eq. (20) of Le Roy's RKR1 paper.

    w(v') = sqrt[(v - v') / (G(v) - G(v'))]

    Args:
        v_prime: The variable of integration.
        v: Vibrational quantum number of the desired level.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
    """
    return math.sqrt((v - v_prime) / (g(v, g_consts) - g(v_prime, g_consts)))


def f_integral_weighted(v: int, v_min: float, g_consts: list[float]) -> float:
    """The Klein integral f(v) computed using Tellinghuisen's weight function.

    f(v) = 2 * sqrt(ħ^2 / 2μ) * ∫_0^sqrt(v - v_min) w(v - u^2) du

    Args:
        v: Vibrational quantum number of the desired level.
        v_min: Minimum vibrational quantum number.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].

    Returns:
        The Klein integral f(v).
    """
    upper_bound = math.sqrt(v - v_min)

    def integrand_f(u: float) -> float:
        # The variable of substitution is u = sqrt(v - v').
        v_prime = v - u**2
        return weight_fn(v_prime, v, g_consts)

    return 2.0 * math.sqrt(hbar2_over_2 / reduced_mass) * quad(integrand_f, 0.0, upper_bound)[0]


def g_integral_weighted(
    v: int, v_min: float, g_consts: list[float], b_consts: list[float]
) -> float:
    """The Klein integral g(v) computed using Tellinghuisen's weight function.

    g(v) = 2 * sqrt(2μ / ħ^2) * ∫_0^sqrt(v - v_min) B(v - u^2) * w(v - u^2) du

    Args:
        v: Vibrational quantum number of the desired level.
        v_min: Minimum vibrational quantum number.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
        b_consts: Rotational constants [B_e, α_e, γ_e, ...].

    Returns:
        The Klein integral g(v).
    """
    upper_bound = math.sqrt(v - v_min)

    def integrand_g(u: float) -> float:
        # The variable of substitution is u = sqrt(v - v').
        v_prime = v - u**2
        return b(v_prime, b_consts) * weight_fn(v_prime, v, g_consts)

    return 2.0 * math.sqrt(reduced_mass / hbar2_over_2) * quad(integrand_g, 0.0, upper_bound)[0]


def rkr(v: int, g_consts: list[float], b_consts: list[float]) -> tuple[float, float]:
    """Return the minimum and maximum interatomic distances for the selected vibrational level [Å].

    Args:
        v: Vibrational quantum number of the desired level.
        v_min: Minimum vibrational quantum number.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
        b_consts: Rotational constants [B_e, α_e, γ_e, ...].

    Returns:
        The minimum and maximum interatomic distances for the selected vibrational level [Å].
    """
    v_min = -0.5

    # NOTE: 26/02/11 - With epsabs=1e-12 and epsrel=1e-12, it took between 300 and 600 evaluations
    # for the non-weighted integrals to converge. With the same tolerances, the weighted integrals
    # took only around 20 iterations.
    f = f_integral_weighted(v, v_min, g_consts)
    g = g_integral_weighted(v, v_min, g_consts, b_consts)

    sqrt_term = math.sqrt(f**2 + f / g)

    # Minimum and maximum interatomic distances [Å].
    r_min = sqrt_term - f
    r_max = sqrt_term + f

    return r_min, r_max


def radial_schrodinger(
    r: NDArray[np.float64], v_max: int, effective_potential: NDArray, resolution: int
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Solve the radial Schrödinger equation using second-order central finite differencing.

    Args:
        r: The domain on which to solve.
        v_max: Maximum vibrational level to compute.
        effective_potential: Sum of the rotationless and centrifugal potentials.
        resolution: Dimension of the computational domain.

    Returns:
        Eigenvalues and normalized eigenfunctions of the radial Schrödinger equation.
    """
    dr: float = r[1] - r[0]

    # Construct the kinetic energy operator via a second-order central finite difference. A sparse
    # array is used to save space.
    kinetic_term: NDArray[np.float64] = (-hbar2_over_2 / (reduced_mass * dr**2)) * diags_array(
        [1.0, -2.0, 1.0],
        offsets=[-1, 0, 1],  # pyright: ignore[reportArgumentType]
        shape=(resolution, resolution),
    ).toarray()

    hamiltonian = kinetic_term + np.diag(effective_potential)

    # The Hamiltonian will always be Hermitian, so the use of eigh is warranted here.
    eigvals, eigfns = eigh(hamiltonian)

    norm_wavefns: list[NDArray[np.float64]] = []

    # Normalize the eigenfunctions ψ(r) such that ∫ ψ'ψ dr = 1.
    for i in range(v_max):
        wavefn: NDArray[np.float64] = eigfns[:, i]
        norm = simpson(wavefn**2, r)
        norm_wavefns.append(wavefn / math.sqrt(norm))

    return eigvals[:v_max], norm_wavefns


def perform_curve_fit(
    fit_fn: Callable, x_data: NDArray[np.float64], y_data: NDArray[np.float64], fit_type: str
) -> NDArray[np.float64]:
    """Perform a non-linear least squares curve fit for the inner and outer extrapolation functions.

    Args:
        fit_fn: Either an inner or outer extrapolation function.
        x_data: Data points for x.
        y_data: Data points for y.
        fit_type: Which fit was performed, used for debugging.

    Returns:
        The parameters obtained for the curve fit.
    """
    params, _, info, _, _ = curve_fit(fit_fn, x_data, y_data, maxfev=100000, full_output=True)
    print(f"The {fit_type} fit took {info['nfev']} iterations.")

    return params


def extrapolate_inner(
    rkr_sorted: NDArray[np.float64], energies_sorted: NDArray[np.float64], fn_type: str = "exp"
) -> tuple[NDArray[np.float64], Callable]:
    """Extrapolate the inner potential well with a selected function.

    The same functions are explained in Le Roy's LEVEL, p. 172.

    Args:
        rkr_sorted: Sorted (min to max) array of RKR turning points.
        energies_sorted: Sorted array of energies matching the RKR turning points.
        fn_type: Which fit to perform.

    Returns:
        The fit parameters and function for inner well extrapolation.
    """
    inner_points: NDArray[np.float64] = rkr_sorted[0:3]
    inner_energy: NDArray[np.float64] = energies_sorted[0:3]

    # Both of these fit functions are explained in Le Roy's LEVEL, p. 172.
    def fit_fn(x, a, b):
        match fn_type:
            case "exp":
                return a * np.exp(-b * x)

            case "inv":
                return a / x**b

    params = perform_curve_fit(fit_fn, inner_points, inner_energy, "inner")

    return params, fit_fn


def extrapolate_outer(
    rkr_sorted: NDArray[np.float64],
    energies_sorted: NDArray[np.float64],
    g_consts: list[float],
    fn_type: str = "inv",
) -> tuple[NDArray[np.float64], Callable]:
    """Extrapolate the outer potential well with a selected function.

    The same functions are explained in Le Roy's LEVEL, Appendix B, p. 5.

    Args:
        rkr_sorted: Sorted (min to max) array of RKR turning points.
        energies_sorted: Sorted array of energies matching the RKR turning points.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
        fn_type: Which fit to perform.

    Returns:
        The fit parameters and function for outer well extrapolation.
    """
    outer_points: NDArray[np.float64] = rkr_sorted[-3:]
    outer_energy: NDArray[np.float64] = energies_sorted[-3:]

    # The dissociation limit given in Herzberg is D_e = ω_e^2 / (4 * ω_ex_e).
    dissociation = g_consts[1] ** 2 / (4.0 * abs(g_consts[2]))

    # All three of these fit functions are given in Le Roy's LEVEL, Appendix B, p. 5.
    def fit_fn(x, a, b, c):
        match fn_type:
            case "exp":
                return dissociation - a * np.exp(-b * (x - c) ** 2)
            case "inv":
                return dissociation - a / x**b
            case "mix":
                return dissociation - a * x**b * np.exp(-c * x)

    params = perform_curve_fit(fit_fn, outer_points, outer_energy, "outer")

    return params, fit_fn


def get_rkr_points(
    v_max: int, g_consts: list[float], b_consts: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the inner and outer RKR turning points and their energies for each vibrational level.

    Args:
        v_max: Maximum vibrational level to compute.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
        b_consts: Rotational constants [B_e, α_e, γ_e, ...].

    Returns:
        Minimum and maximum internuclear distances and their energies for each vibrational level.
    """
    rkr_mins = np.empty(v_max)
    rkr_maxs = np.empty(v_max)
    energies = np.empty(v_max)

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
    """Return the effective potential, i.e., the sum of the rotationless and centrifugal potentials.

    Args:
        r: The domain on which to solve.
        rkr_mins: All "left side" RKR points.
        rkr_maxs: All "right side" RKR points.
        energies: Energies for all RKR points.
        g_consts: Vibrational constants [T_e, ω_e, ω_ex_e, ω_ey_e, ...].
        j_qn: Rotational quantum number J.

    Returns:
        The effective internuclear potential.
    """
    rkr_all = np.concatenate((rkr_mins, rkr_maxs))
    energies_all = np.concatenate((energies, energies))

    # The x values in CubicSpline must be listed in increasing order, so sort to ensure this.
    sorted_indices = np.argsort(rkr_all)

    rkr_sorted = rkr_all[sorted_indices]
    energies_sorted = energies_all[sorted_indices]

    cubic_spline = CubicSpline(rkr_sorted, energies_sorted)

    params_inner, fit_inner = extrapolate_inner(rkr_sorted, energies_sorted)
    params_outer, fit_outer = extrapolate_outer(rkr_sorted, energies_sorted, g_consts)

    # The absolute minimum and maximum internuclear distances computed from the RKR routine.
    rkr_min: float = rkr_sorted[0]
    rkr_max: float = rkr_sorted[-1]

    l_mask = r < rkr_min
    m_mask = (r >= rkr_min) & (r <= rkr_max)
    r_mask = r > rkr_max

    effective_potential = np.empty_like(r)

    # Everything within the RKR points must be interpolated, and everything outside must be
    # extrapolated.
    effective_potential[l_mask] = fit_inner(r[l_mask], *params_inner)
    effective_potential[m_mask] = cubic_spline(r[m_mask])
    effective_potential[r_mask] = fit_outer(r[r_mask], *params_outer)

    # Add centrifugal J dependence to the rotationless potential to get the effective potential.
    effective_potential += (hbar2_over_2 / (reduced_mass * r**2)) * j_qn * (j_qn + 1)

    plt.plot(r[l_mask], effective_potential[l_mask], color="blue")
    plt.plot(r[m_mask], effective_potential[m_mask], color="black")
    plt.plot(r[r_mask], effective_potential[r_mask], color="red")

    return r, effective_potential


def main() -> None:
    """Entry point."""
    v_max_up = 16
    v_max_lo = 19
    j_qn_up = 0
    j_qn_lo = 0

    dim = 1000

    rkr_mins_up, rkr_maxs_up, energies_up = get_rkr_points(v_max_up, g_consts_up, b_consts_up)
    rkr_mins_lo, rkr_maxs_lo, energies_lo = get_rkr_points(v_max_lo, g_consts_lo, b_consts_lo)

    r_min: float = min(rkr_mins_up.min(), rkr_mins_lo.min())
    r_max: float = max(rkr_maxs_up.max(), rkr_maxs_lo.max())

    r = np.linspace(r_min, r_max, dim)

    r_up, potential_up = get_potential(
        r, rkr_mins_up, rkr_maxs_up, energies_up, g_consts_up, j_qn_up
    )
    r_lo, potential_lo = get_potential(
        r, rkr_mins_lo, rkr_maxs_lo, energies_lo, g_consts_lo, j_qn_lo
    )

    eigvals_up, wavefns_up = radial_schrodinger(r_up, v_max_up, potential_up, dim)
    eigvals_lo, wavefns_lo = radial_schrodinger(r_lo, v_max_lo, potential_lo, dim)

    scaling_factor = 500

    for i, psi in enumerate(wavefns_up):
        plt.plot(r_up, psi * scaling_factor + eigvals_up[i])

    for i, psi in enumerate(wavefns_lo):
        plt.plot(r_lo, psi * scaling_factor + eigvals_lo[i])

    plt.xlabel(r"Internuclear Distance, $r$ [$\AA$]")
    plt.ylabel(r"Energy, $E$ [cm$^{-1}$]")
    plt.show()

    # Compute Franck-Condon factors and compare with known data from Cheung.
    fcfs = np.zeros((v_max_up, v_max_lo))

    for i in range(v_max_up):
        for j in range(v_max_lo):
            fcfs[i][j] = np.abs(simpson(wavefns_up[i] * wavefns_lo[j], r_up)) ** 2

    cheung = np.genfromtxt("../data/cheung.csv", delimiter=",", dtype=np.float64)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Simulation")
    axs[0].imshow(fcfs, origin="lower")

    axs[1].set_title("Cheung")
    im = axs[1].imshow(cheung, origin="lower")

    for ax, data in zip(axs, [fcfs, cheung]):
        x_range = range(data.shape[1])
        y_range = range(data.shape[0])
        ax.set_xticks(x_range, labels=x_range)
        ax.set_yticks(y_range, labels=y_range)
        ax.set(xlabel="$v''$", ylabel="$v'$")

    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, label="Overlap Integral")
    plt.show()


if __name__ == "__main__":
    main()

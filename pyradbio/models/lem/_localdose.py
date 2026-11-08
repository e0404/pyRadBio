"""Local-dose compound-Poisson machinery for the LEM IV full simulation.

Numerical core of the Friedrich et al. 2012 (Int. J. Radiat. Biol. 88:103-107,
doi:10.3109/09553002.2011.611213, Fig. 4) full-simulation ``beta_I``. It holds no
biophysics: [`pyradbio.models.lem.lem4`][] supplies the radial dose profile and
the per-subvolume response kernel, this module supplies the statistics.

**The model.** Following the local-effect principle of Elsasser & Scholz 2007
(Radiat. Res. 167:319-329, doi:10.1667/RR0467.1, eq. 5), the multi-track effect
is a sum over subvolumes, each responding to its own *accumulated local dose* --
not one global whole-nucleus cluster index. Tracks arrive as a Poisson process
in the plane transverse to the beam with areal intensity

    phi = D / I_track ,     I_track = integral D_track(r) 2 pi r dr   [Gy um^2]

(``D`` the macroscopic dose), so the local dose at a random nucleus point is a
*compound-Poisson* (shot-noise) sum: a track whose core passes at distance ``r``
contributes a jump ``g(r)``, and the mean number of tracks contributing a jump
from ``[r, r+dr]`` is ``phi 2 pi r dr``. Hence the local-dose density ``f_D(d)``
has the exact characteristic function

    log E[e^{-i theta Z}] = integral (e^{-i theta g(r)} - 1) phi 2 pi r dr ,

evaluated here by **direct summation over the radial nodes** and inverted by one
FFT (`CompoundPoissonSolver`). The lethal-event count then follows as
``N_ion(D) = N_sv <rho(d)>_{f_D}`` for a per-subvolume response ``rho``.

Two properties make this the right formulation, and both are exercised by the
tests:

* The track plane is **infinite**, so the far penumbra is folded in analytically.
  No ``r_sim`` cap is needed -- at LET 10 keV/um the delta-ray range reaches
  ``r_max ~ 2200 um``, which is hopeless for an explicit track-placement Monte
  Carlo but costs nothing here.
* As the tracks become dense (low LET) the jumps become many and small, so
  ``f_D -> delta(d - D)`` and the response collapses to the photon response --
  the ``beta_I -> beta_x`` limit, recovered exactly rather than by calibration.

**Numerical warnings** (each cost a prototype round; do not "simplify" them):

* The jump-size range spans ~6 decades (1e-4 .. 1e2 Gy), so the many tiny
  penumbra jumps are tempting to aggregate into a Gaussian or Gamma small-jump
  term. **Do not.** Their aggregate is strictly non-negative and skewed, so a
  Gaussian's spurious negative tail *wraps around the periodic FFT grid* and
  lands at high dose, inflating the mean by orders of magnitude (observed:
  mean 148 Gy instead of 0.4 Gy). A Gamma match is better but still leaks at
  very low dose. The exact characteristic function summed directly over the
  radial nodes has no such failure mode and is cheap enough.
* The subvolume-averaged single-track profile ``g_B`` must **conserve the
  dose-area integral** ``integral g_B 2 pi r dr = I_track``. Box-averaging is a
  convolution with a unit-mass kernel, so this holds exactly in the continuum;
  a crude near-core box quadrature breaks it by ~35% (over-counting the singular
  1/r^2 core) and biases every downstream ``beta``. `box_averaged_profile`
  therefore integrates in *polar* coordinates against an exact circle-in-square
  angular overlap, so the core is resolved by the radial nodes themselves.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import fit_lq_linearised
from pyradbio.models.lem import _localdose_kernel
from pyradbio.models.lem.engine import ENGINES, Engine

# Execution engine for `box_averaged_profile` (the local-dose hot spot).
#   "auto" (default) -- the numba kernel when the ``fast`` extra is installed, else NumPy.
#   "numpy" -- the vectorised reference (also the fallback and the accuracy anchor).
#   "numba" -- force the compiled kernel; errors if numba is unavailable.
# The two agree to machine precision (same quadrature), so "auto" is safe by default.
BoxEngine = Engine  # one definition: pyradbio.models.lem.engine

# Chunk size (frequencies per block) for the characteristic-function summation.
# Bounds peak memory at ``_CF_CHUNK * n_jumps * 16`` bytes without measurably
# slowing the (exp-dominated) inner loop.
_CF_CHUNK: int = 4096

FloatArray = npt.NDArray[np.float64]


def trapezoid_weights(x: FloatArray) -> FloatArray:
    """Quadrature weights ``w`` with ``sum(w * f) == trapezoid(f, x)``.

    Lets a non-uniform (log-spaced) radial grid be reused as a set of discrete
    jump intensities without re-integrating for every dose point.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        raise ValueError(f"need at least 2 nodes, got {x.size}")
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def _centred_square_arc(s: FloatArray, half: float) -> FloatArray:
    """Angular measure [rad] of the circle ``|p| = s`` inside a square of
    half-side ``half`` **centred on the origin**.

    Closed form: the circle is wholly inside up to the inradius, wholly outside
    beyond the circumradius, and in between loses the four arcs that stick out
    past the edges:

    ```text
    s <= half            ->  2 pi
    half < s < half*sqrt2 ->  2 pi - 8 arccos(half / s)
    s >= half*sqrt2      ->  0
    ```
    """
    s = np.asarray(s, dtype=np.float64)
    safe = np.maximum(s, 1.0e-300)
    cut = np.clip(half / safe, -1.0, 1.0)
    return np.asarray(
        np.where(
            s <= half,
            2.0 * math.pi,
            np.maximum(2.0 * math.pi - 8.0 * np.arccos(cut), 0.0),
        ),
        dtype=np.float64,
    )


def _log_log_interpolator(r: FloatArray, dose: FloatArray) -> Callable[[FloatArray], FloatArray]:
    """Log-log interpolation of a radial dose profile, flat inside ``r[0]`` and
    zero beyond ``r[-1]`` (the delta-ray range)."""
    log_r = np.log(r)
    log_d = np.log(np.maximum(dose, 1.0e-300))
    r_lo, r_hi = float(r[0]), float(r[-1])
    inner = float(dose[0])

    def dose_at(x: FloatArray) -> FloatArray:
        x = np.asarray(x, dtype=np.float64)
        out = np.exp(np.interp(np.log(np.clip(x, r_lo, r_hi)), log_r, log_d))
        out = np.where(x < r_lo, inner, out)
        return np.asarray(np.where(x > r_hi, 0.0, out), dtype=np.float64)

    return dose_at


def box_averaged_profile(
    r: FloatArray,
    dose: FloatArray,
    rho: FloatArray,
    *,
    ell: float,
    n_nodes: int = 512,
    engine: BoxEngine = "auto",
) -> FloatArray:
    """Mean dose in a square subvolume of side ``ell`` whose centre sits at
    distance ``rho`` from a track core with radial profile ``(r, dose)``.

    This is the single-track jump size of the compound-Poisson local dose, for
    *either* response kernel. LEM IV resolves damage on the 540 nm giant-loop
    scale, so a subvolume responds to its *average* dose, not to the pointwise
    dose (Pfuhl, Friedrich & Scholz 2020, Radiat. Res. 193:130: "the number of DSB
    in each subvolume is determined according to the local dose distribution
    within the subvolume").

    Using the pointwise dose instead is not a refinement but a bug: sampling the
    singular ``1/r^2`` profile at subvolume centres over-reads the core by orders
    of magnitude whenever a centre lands near a track, which manufactures a
    spurious high-LET ``beta`` collapse.

    The subvolume lattice is fixed while the track lands at a random position
    relative to it, so the physically meaningful ``g_B`` is the average over the
    box centre's *direction* at fixed ``rho``. Writing the source point at radius
    ``r`` and angle ``delta`` from the box centre, the offset ``q - c`` has length
    ``s = sqrt(r^2 + rho^2 - 2 r rho cos delta)`` and a direction that sweeps
    uniformly as the pair rotates, so the direction average turns into the
    *centred* circle-in-square measure and

        g_B(rho) = (1/ell^2) integral D(r) K(r, rho) r dr ,
        K(r, rho) = (1/pi) integral_0^pi Theta_0(s(delta)) ddelta ,

    with ``Theta_0`` = `_centred_square_arc`. ``K`` is symmetric in
    ``r <-> rho`` and satisfies ``integral K(r, rho) rho drho = ell^2``, so
    dose-area conservation ``integral g_B 2 pi rho drho == integral D 2 pi r dr``
    holds *identically* rather than by renormalisation. (Skipping the direction
    average -- evaluating only with the box centre on the ``+x`` axis -- silently
    loses ~8% of the dose-area, because a square offset along the radius sees a
    different average than one offset diagonally.)

    Working in polar coordinates about the singularity is also what keeps the
    near-core average honest: a 2-D sub-sampling grid would have to be
    impractically fine there, and a crude one over-counts the ``1/r^2`` core by
    ~35%. The radial quadrature runs on a per-``rho`` log grid covering only the
    band that can contribute (``|r - rho| < ell/sqrt2``, the square's
    circumradius), and the ``delta`` range is likewise narrowed to where
    ``Theta_0`` is non-zero -- ``sin delta < ell/(sqrt2 rho)`` -- so far-field
    radii stay as well resolved as the core.

    ``engine`` selects the execution backend (`BoxEngine`): ``"auto"`` uses the
    compiled numba kernel (`_localdose_kernel`) when the ``fast`` extra is
    installed, else the NumPy path here. The two run the identical quadrature and agree
    to machine precision, so the choice is purely performance (~20-25x); this function
    is ~94% of a ``LEM4.local_dose_beta`` call.
    """
    r = np.asarray(r, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES!r}, got {engine!r}")
    if engine == "numba" and not _localdose_kernel.HAVE_NUMBA:
        raise ValueError(
            "engine='numba' requires the 'fast' extra (pip install pyradbio[fast]); "
            "use engine='numpy' or 'auto'."
        )
    if engine != "numpy" and _localdose_kernel.HAVE_NUMBA:
        # Compiled path: numerically identical (same quadrature), ~20-25x faster.
        return _localdose_kernel.box_average(
            np.log(r),
            np.log(np.maximum(dose, 1.0e-300)),
            float(r[0]),
            float(r[-1]),
            float(dose[0]),
            rho,
            float(ell),
            int(n_nodes),
        )
    half = 0.5 * ell
    reach = half * math.sqrt(2.0)  # circumradius: beyond this the square misses
    dose_at = _log_log_interpolator(r, dose)
    r_floor = float(r[0])
    out = np.empty(rho.shape, dtype=np.float64)
    for i, rc in enumerate(rho):
        centre = float(rc)
        r_lo = max(centre - reach, r_floor)
        nodes = np.logspace(math.log10(r_lo), math.log10(centre + reach), n_nodes, dtype=np.float64)
        # Theta_0 vanishes unless the pair can come within the circumradius; the
        # closest approach over r at angle delta is rho sin(delta).
        d_max = math.pi if centre <= reach else math.asin(min(1.0, reach / centre))
        delta = np.linspace(0.0, d_max, n_nodes, dtype=np.float64)
        # Law of cosines. Clamp at zero: for nodes ~ centre and delta ~ 0 this is
        # the catastrophically-cancelling form of (nodes - centre)^2, which can
        # round to a small negative and turn the whole profile into NaN.
        s = np.sqrt(
            np.maximum(
                nodes[:, None] ** 2
                + centre * centre
                - 2.0 * centre * nodes[:, None] * np.cos(delta)[None, :],
                0.0,
            )
        )
        kernel = np.trapezoid(_centred_square_arc(s, half), delta, axis=1) / math.pi
        out[i] = float(np.trapezoid(dose_at(nodes) * kernel * nodes, nodes)) / (ell * ell)
    return out


def dose_area_histogram(
    x: FloatArray,
    w: FloatArray,
    *,
    bins_per_decade: int = 48,
) -> tuple[FloatArray, FloatArray]:
    """Compress jumps ``(x, w)`` onto a log-dose grid, exactly preserving both
    ``sum(w)`` and the dose-area ``sum(w * x)``.

    The single-track profile supplies ~1500 radial nodes whose jump sizes span
    many decades but vary smoothly, so the characteristic-function sum (whose
    cost is ``n_frequencies * n_jumps``) can be shortened a long way at no
    accuracy cost. Each occupied bin keeps its total intensity and its
    *intensity-weighted mean* jump size, so the compound-Poisson mean
    ``sum(w x)`` -- which must equal the macroscopic dose -- is exact, and the
    characteristic function (smooth in ``x``) is accurate to the bin width.

    Note this is a histogram of the *jump sizes*, distinct from binning onto the
    linear FFT dose grid, which is what wraps and corrupts the density.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    keep = (x > 0.0) & (w > 0.0)
    x, w = x[keep], w[keep]
    if x.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    log_x = np.log10(x)
    lo = math.floor(float(log_x.min()) * bins_per_decade)
    idx = np.maximum(np.floor(log_x * bins_per_decade).astype(np.int64) - lo, 0)
    n_bins = int(idx.max()) + 1
    # np.bincount is typed as integer-valued; with ``weights`` it is float64.
    w_bin = np.asarray(np.bincount(idx, weights=w, minlength=n_bins), dtype=np.float64)
    xw_bin = np.asarray(np.bincount(idx, weights=w * x, minlength=n_bins), dtype=np.float64)
    occupied = w_bin > 0.0
    return xw_bin[occupied] / w_bin[occupied], w_bin[occupied]


class CompoundPoissonSolver:
    """Density of a compound-Poisson sum ``Z = sum_k X_k`` via its exact
    characteristic function, reusable across intensities.

    ``x_jump`` are the distinct jump sizes and ``w_jump`` their Poisson
    intensities (expected counts) at unit scale. The log-characteristic function

        psi(theta) = sum_j w_j (e^{-i theta x_j} - 1)

    is summed *directly* over the jump nodes -- no parametric small-jump
    approximation, which is what keeps the periodic FFT grid clean (see the
    module docstring). It is linear in the overall intensity, so the expensive
    pass is done once in ``__init__`` and every `density` call is a single
    ``exp`` plus one inverse FFT. That matters because the physical scale is
    ``phi = D / I_track``: a whole dose-response curve costs one setup.

    The reconstruction grid spans ``[0, 2 d_hi)`` but only its lower half is
    reported: the upper half represents *negative* doses under the FFT's periodic
    wrap, so ``d_hi`` must bound the true support (jumps included) with room to
    spare. See `density` for why that split is needed and what is done with
    the wrapped mass. ``n_grid`` sets the resolution ``dd = 2 d_hi / n_grid``.
    """

    __slots__ = ("_dd", "_half", "_kernel", "_n_grid", "_psi_unit")

    def __init__(
        self,
        x_jump: FloatArray,
        w_jump: FloatArray,
        *,
        d_hi: float,
        n_grid: int,
        smoothing: float = 1.0,
    ) -> None:
        x = np.asarray(x_jump, dtype=np.float64)
        w = np.asarray(w_jump, dtype=np.float64)
        if x.shape != w.shape:
            raise ValueError(f"x_jump and w_jump must have equal shape, got {x.shape} {w.shape}")
        if x.size and float(x.max()) > d_hi:
            raise ValueError(
                f"largest jump {float(x.max()):.4g} Gy exceeds d_hi={d_hi:.4g} Gy; it would "
                "alias around the periodic grid -- raise d_hi."
            )
        if n_grid < 4 or n_grid % 2:
            raise ValueError(f"n_grid must be even and >= 4, got {n_grid}")
        self._n_grid = int(n_grid)
        self._half = int(n_grid) // 2
        self._dd = 2.0 * float(d_hi) / n_grid
        n_freq = n_grid // 2 + 1
        # theta_m = 2 pi m / (n_grid * dd) matches numpy's fft sign convention,
        # under which fft(L)[m] = sum_k L[k] exp(-i theta_m k dd).
        theta = 2.0 * math.pi * np.arange(n_freq, dtype=np.float64) / (n_grid * self._dd)
        psi = np.empty(n_freq, dtype=np.complex128)
        total_w = float(w.sum())
        for start in range(0, n_freq, _CF_CHUNK):
            block = theta[start : start + _CF_CHUNK]
            psi[start : start + _CF_CHUNK] = np.exp(-1j * np.outer(block, x)) @ w - total_w
        self._psi_unit = psi
        # Jump sizes are discrete and generally off-grid, so the exact density is
        # a comb of deltas at non-grid points; inverting its characteristic
        # function straight onto the grid reconstructs that comb through a
        # Dirichlet kernel, i.e. with large oscillations (observed: a mean of
        # -0.87 Gy where the true mean is 2.21 Gy). Convolving with a narrow
        # Gaussian resolves the comb onto the grid: being symmetric and of unit
        # mass it leaves the first moment exact, and it costs only ``sigma^2``
        # of spurious variance -- negligible against the physical spread, and the
        # smooth, bounded response averages this feeds are insensitive to it.
        # A triangular (linear-interpolation) kernel is *not* enough: its sinc^2
        # transfer function is still 0.41 at the Nyquist frequency, so the comb
        # survives and ~1.5% of the mass wraps past zero.
        sigma = smoothing * self._dd
        theta_all = 2.0 * math.pi * np.arange(n_freq, dtype=np.float64) / (n_grid * self._dd)
        self._kernel = np.exp(-0.5 * (sigma * theta_all) ** 2)

    @property
    def dd(self) -> float:
        """Dose-grid spacing [Gy]."""
        return self._dd

    @property
    def dose_grid(self) -> FloatArray:
        """Local-dose grid ``d_k = k * dd`` [Gy] the density is sampled on."""
        return np.arange(self._half, dtype=np.float64) * self._dd

    def density(self, scale: float) -> FloatArray:
        """Probability mass on `dose_grid` when every intensity is
        multiplied by ``scale`` (physically ``scale = phi = D / I_track``).

        The local dose is non-negative and has an *atom at zero* (the probability
        that no track comes close enough), which the resolving kernel smears
        symmetrically -- so part of that atom lands at negative dose and, on a
        periodic grid, reappears at the very top. Left alone this is not a
        rounding error but a sign flip in meaning: the wrapped mass would be
        scored at the maximum local dose, i.e. as maximally clustered damage,
        precisely inverting its physical content. The upper half of the grid is
        therefore read as negative dose and folded back onto ``d = 0``, which is
        exact as long as ``d_hi`` bounds the true support (`wrapped_mass`
        checks the assumption).

        Masses are *not* clipped at zero: residual ripple is symmetric, and
        one-sided clipping would turn it into a systematic bias in the smooth
        response averages this feeds. Total mass is ``exp(psi(0)) = 1``.
        """
        f = np.fft.irfft(np.exp(scale * self._psi_unit) * self._kernel, n=self._n_grid)
        out = np.array(f[: self._half], dtype=np.float64)
        out[0] += float(f[self._half :].sum())
        return out

    def wrapped_mass(self, scale: float) -> float:
        """Probability mass the reconstruction places at dose ``> d_hi``.

        Physically this must be ~0: it is the tail `density` folds back to
        zero, valid only while ``d_hi`` really does bound the support. A value
        that is not small means the dose grid is too short and the folded mass is
        genuine high-dose damage being scored as none.
        """
        f = np.fft.irfft(np.exp(scale * self._psi_unit) * self._kernel, n=self._n_grid)
        upper = f[self._half :]
        # The wrap is symmetric about the top of the grid; the half nearest d_hi
        # is real high-dose mass, the half nearest 2*d_hi is the negative-dose
        # image of the atom at zero.
        return float(upper[: upper.size // 2].sum())

    def response_mean(self, scale: float, response: FloatArray) -> float:
        """``<rho(Z)>`` for a response tabulated on `dose_grid`."""
        return float(np.dot(self.density(scale), response))


def subvolume_centres(
    r_eff: float, ell: float, *, dx: float = 0.0, dy: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Transverse centres of the ``ell``-sided subvolumes tiling a nucleus of
    effective radius ``r_eff`` (those whose centre falls inside the disk).

    ``dx``/``dy`` shift the lattice's **phase** before the centre-inside-disk cut,
    so the disk boundary is marginalised along with the tiling. The tiling has no
    preferred origin, and pinning it to one is not harmless: the default phase puts
    a loop centre exactly on the origin whenever ``ceil(2 r_eff / ell)`` is odd
    (``r_eff = 5.0 um`` at ``ell = 540 nm``) and straddles it otherwise, which moved
    ``1 - C_I`` by 4x over a 6% change in ``r_eff``. Callers that need a
    registration-independent answer average over a period; see
    [`LEM4.analytic_discrete_cluster_index`][pyradbio.models.lem.lem4.LEM4.analytic_discrete_cluster_index].

    The candidate grid is widened by two extra rings when a phase is applied, so a
    shifted lattice still covers the disk. That cannot change the ``dx = dy = 0``
    result -- the extra rings lie outside the disk and are cut.
    """
    pad = 2 if dx == 0.0 and dy == 0.0 else 4
    n = math.ceil(2.0 * r_eff / ell) + pad
    idx = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * ell
    gx, gy = np.meshgrid(idx + dx, idx + dy)
    inside = gx * gx + gy * gy <= r_eff * r_eff
    return np.asarray(gx[inside], dtype=np.float64), np.asarray(gy[inside], dtype=np.float64)


def far_track_background_per_Gy(
    jump_at: Callable[[FloatArray], FloatArray],
    *,
    b_cut: float,
    r_sim: float,
    tracks_per_Gy: float,
    n_nodes: int = 1024,
) -> float:
    """Mean local dose per Gy contributed by tracks beyond impact parameter ``b_cut``.

    Tracks are Poisson with areal density ``tracks_per_Gy * D / (pi r_sim^2)``, so the
    expected dose they deposit at the nucleus centre is that density times
    ``int_{b_cut}^{r_sim} g(b) 2 pi b db``, linear in ``D``.

    This is the *mean-field* half of the track/background split used by
    `monte_carlo_effect`: far tracks are replaced by their expectation rather
    than sampled, which is exactly the approximation the discrete engine already makes
    through ``DiscreteSetup.retained_dose_fraction``. It evaluates ``g`` at the
    distance to the nucleus **centre**, i.e. treats the nucleus as a point -- the
    leading approximation, and the reason ``b_cut`` must stay well outside ``r_eff``.
    """
    if not (0.0 < b_cut < r_sim):
        raise ValueError(f"need 0 < b_cut < r_sim, got b_cut={b_cut}, r_sim={r_sim}")
    b = np.linspace(b_cut, r_sim, n_nodes, dtype=np.float64)
    integral = float(np.trapezoid(jump_at(b) * 2.0 * math.pi * b, b))
    return tracks_per_Gy * integral / (math.pi * r_sim * r_sim)


def monte_carlo_effect(
    jump_at: Callable[[FloatArray], FloatArray],
    score: Callable[[FloatArray], float],
    *,
    doses: FloatArray,
    tracks_per_Gy: float,
    r_eff: float,
    r_sim: float,
    ell: float,
    n_real: int,
    rng: np.random.Generator,
    b_cut: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Explicit-track Monte-Carlo reference for the convolution.

    Places ``Poisson(tracks_per_Gy * D)`` tracks uniformly over a disk of radius
    ``r_sim``, accumulates each subvolume's local dose from ``jump_at(distance)``,
    and scores the resulting pattern. Returns ``(mean_effect, mean_survival)`` over
    realisations -- i.e. both ``<E>`` and ``<exp(-E)>``, which differ by the
    overkill (a cell already killed cannot be killed again). The deterministic
    convolution computes ``<E>``, so that column is the like-for-like check;
    ``<exp(-E)>`` is the convention the convolution cannot reach analytically.

    ``jump_at`` **must be the subvolume-averaged single-track profile**, not the
    pointwise radial dose. Sampling the pointwise ``1/r^2`` dose at subvolume
    *centres* over-reads the core by orders of magnitude whenever a centre lands
    near a track, which manufactures a spurious high-LET ``beta`` collapse; that
    artifact, not any physics, is what sampling at centres produces.

    ``score`` maps the vector of subvolume local doses to **one** cell effect, so
    it carries the whole kernel-specific aggregation: a sum over loops for a
    per-loop response (GLOBLE), a mean for a response that is already the
    whole-cell effect at uniform dose (``-ln S_x``), or a genuinely non-additive
    functional of the entire pattern (LEM IV's cluster index, which forms
    ``C_I = N_cDSB/N_dmg`` over the whole nucleus before mapping to an effect).
    That last case is why this takes a pattern scorer rather than a per-subvolume
    response: the cluster index has no per-subvolume decomposition.

    ``b_cut`` enables the **track/background split**. Cost here is
    ``n_tracks x n_subvolumes`` and ``n_tracks`` grows as ``(r_sim/r_eff)^2`` with
    ``r_sim = r_eff + r_max ~ E^1.7``, so low LET -- meaning high energy -- is the
    expensive end: carbon at LET 20 puts a 293 um domain around a 5 um nucleus, and
    nearly every sampled track passes hundreds of um away to deposit a vanishing,
    near-uniform dose. With ``b_cut`` set, only tracks inside that impact parameter are
    sampled and everything beyond is added as the deterministic mean
    (`far_track_background_per_Gy`), cutting cost by ``(r_sim/b_cut)^2``.

    The approximation is that far tracks are replaced by their expectation, so their
    *fluctuation* is dropped. That is the favourable direction here: the far population
    is large, so its relative fluctuation is small, and it gets larger -- hence the
    approximation better -- exactly as the domain grows. ``b_cut`` must still sit well
    outside ``r_eff``, since the background is evaluated as if the nucleus were a
    point. ``None`` (the default) samples everything and is bit-identical to the
    pre-split engine.
    """
    doses = np.asarray(sorted(doses), dtype=np.float64)
    sx, sy = subvolume_centres(r_eff, ell)
    n_sv = sx.size
    d_max = float(doses[-1])
    sample_radius = r_sim if b_cut is None else float(b_cut)
    mean_tracks = tracks_per_Gy * d_max * (sample_radius / r_sim) ** 2
    background_per_Gy = (
        0.0
        if b_cut is None
        else far_track_background_per_Gy(
            jump_at, b_cut=float(b_cut), r_sim=r_sim, tracks_per_Gy=tracks_per_Gy
        )
    )
    effect = np.zeros(doses.size, dtype=np.float64)
    survival = np.zeros(doses.size, dtype=np.float64)
    for _ in range(n_real):
        n_tracks = int(rng.poisson(mean_tracks))
        if n_tracks == 0:
            if background_per_Gy == 0.0:
                survival += 1.0
                continue
            # With a split, no sampled track still leaves the far background.
            for i, dose in enumerate(doses):
                e = score(np.full(n_sv, background_per_Gy * float(dose)))
                effect[i] += e
                survival[i] += math.exp(-min(e, 700.0))
            continue
        radius = sample_radius * np.sqrt(rng.random(n_tracks))
        angle = 2.0 * math.pi * rng.random(n_tracks)
        tx, ty = radius * np.cos(angle), radius * np.sin(angle)
        # Give each track an "arrival dose" so one realisation serves every dose
        # point (common random numbers): the pattern at D is the tracks with
        # arrival <= D, so the dose response is monotone within a realisation.
        arrival = d_max * rng.random(n_tracks)
        order = np.argsort(arrival)
        tx, ty, arrival = tx[order], ty[order], arrival[order]
        dist = np.sqrt((sx[None, :] - tx[:, None]) ** 2 + (sy[None, :] - ty[:, None]) ** 2)
        cumulative = np.cumsum(jump_at(dist), axis=0)
        for i, dose in enumerate(doses):
            last = int(np.searchsorted(arrival, dose, side="right")) - 1
            local = cumulative[last] if last >= 0 else np.zeros(n_sv, dtype=np.float64)
            if background_per_Gy != 0.0:
                local = local + background_per_Gy * float(dose)
            e = score(local)
            effect[i] += e
            survival[i] += math.exp(-min(e, 700.0))
    return effect / n_real, survival / n_real


def lq_coefficients(doses: FloatArray, effect: FloatArray) -> tuple[float, float]:
    """Least-squares ``(alpha, beta)`` of ``effect = alpha D + beta D^2``.

    Fitted as ``effect / D = alpha + beta D``, the linearisation Friedrich et al.
    2012 use to read ``alpha_I`` and ``beta_I`` off the simulated dose response.
    ``beta`` is returned unclamped: clamping negative fits to zero biases any
    subsequent averaging upward, and a negative fit is diagnostic (it means the
    window has reached the single-track overkill saturation, where the response
    is concave and the LQ form no longer applies).

    Delegates to [`pyradbio.core.lq.fit_lq_linearised`][], whose arithmetic is
    this function's former body in the same order (bit-equality on recorded LEM IV
    inputs is pinned in ``tests/core/test_lq_fits.py``). It raises ``ValueError``
    on degenerate input -- fewer than 2 doses, a non-positive dose, non-finite
    ``effect`` or a single repeated dose -- where the former body returned NaN or
    a minimum-norm solution.
    """
    return fit_lq_linearised(doses, effect)


__all__ = [
    "CompoundPoissonSolver",
    "box_averaged_profile",
    "dose_area_histogram",
    "lq_coefficients",
    "monte_carlo_effect",
    "subvolume_centres",
    "trapezoid_weights",
]

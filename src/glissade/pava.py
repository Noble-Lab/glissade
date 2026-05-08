import warnings
import math
import numpy as np
from math import log, sqrt

def _ecdf_on_grid(sample_sorted, grid):
	"""
	Returns ECDF values evaluated on 'grid':
	F_n(x) = (1/n) * #{i: sample_i <= x}, using 'right' side for step function.
	Assumes sample_sorted is sorted ascending.
	"""
	n = len(sample_sorted)
	idx = np.searchsorted(sample_sorted, grid, side='right')
	return idx / float(n)

def _lcm_on_interval(x, y, a, b):
	"""
	Least Concave Majorant (LCM) of y(x) on [a,b], computed on the subgrid x_a..x_b.
	Returns lcm_y on the *full* grid x, replacing y on [a,b] with its LCM and leaving outside unchanged.

	Method: PAVA on slopes (isotonic regression with a nonincreasing constraint) on the restricted interval.
	"""
	# Identify the subinterval indices
	if a > b:
		a, b = b, a
	ia = np.searchsorted(x, a, side='left')
	ib = np.searchsorted(x, b, side='right') - 1
	ia = max(0, min(ia, len(x)-1))
	ib = max(0, min(ib, len(x)-1))
	if ia >= ib:
		return y.copy()

	x_sub = x[ia:ib+1]
	y_sub = y[ia:ib+1]

	# Slopes and widths on the subinterval
	dx = np.diff(x_sub)
	# Guard against zero-width (duplicate x); if any, collapse by unique
	if np.any(dx <= 0):
		# Enforce strictly increasing x_sub by uniquifying
		us, idxu = np.unique(x_sub, return_index=True)
		y_sub = y_sub[idxu]
		x_sub = us
		if len(x_sub) < 2:
			out = y.copy()
			return out
		dx = np.diff(x_sub)

	sl = np.diff(y_sub) / dx
	w = dx.copy()

	# PAVA for nonincreasing slopes
	S, W, C = [], [], []  # block means, weights, counts
	for i in range(len(sl)):
		S.append(sl[i]); W.append(w[i]); C.append(1)
		# enforce S[-2] >= S[-1] (nonincreasing)
		while len(S) >= 2 and (S[-2] < S[-1] - 1e-15):
			newS = (S[-2]*W[-2] + S[-1]*W[-1]) / (W[-2] + W[-1])
			newW = W[-2] + W[-1]
			newC = C[-2] + C[-1]
			S[-2] = newS; W[-2] = newW; C[-2] = newC
			S.pop(); W.pop(); C.pop()

	# Expand block slopes back to per-segment slopes
	sl_hat = np.concatenate([np.full(c, s, dtype=float) for s, c in zip(S, C)])

	# Reconstruct the concave majorant values on x_sub
	lcm_sub = np.empty_like(y_sub)
	lcm_sub[0] = y_sub[0]
	for j in range(len(sl_hat)):
		lcm_sub[j+1] = lcm_sub[j] + sl_hat[j] * (x_sub[j+1] - x_sub[j])

	# Splice back into full vector
	out = y.copy()
	out[ia:ib+1] = lcm_sub
	return out

def _gap_T(x, y, a, b):
	"""
	Integrated squared LCM gap on [a,b]:
	T = ∫_{a}^{b} [LCM(y) - y]^2 dx, computed by trapezoidal rule on the grid.
	"""
	lcm_y = _lcm_on_interval(x, y, a, b)
	mask = (x >= min(a, b)) & (x <= max(a, b))
	xs = x[mask]
	if len(xs) < 2:
		return 0.0
	gs = (lcm_y[mask] - y[mask])
	dx = np.diff(xs)
	return float(np.sum(0.5 * (gs[:-1]**2 + gs[1:]**2) * dx))


def _concave_projection_on_interval(x, y, a, b):
	"""
	Concave projection: replace y on [a,b] by its LCM; return the projected vector on full grid.
	"""
	return _lcm_on_interval(x, y, a, b)

# =========================================
# Tolerance providers (DKW and Bootstrap)
# =========================================

def _sample_from_cdf(grid_x, cdf_vals, size, rng):
	"""
	Inverse-CDF sampling from a right-continuous step CDF on grid_x.
	"""
	u = rng.random(size)
	idx = np.searchsorted(cdf_vals, u, side='left')
	idx = np.clip(idx, 0, len(grid_x)-1)
	return grid_x[idx]

def tol_bootstrap(alpha, x_grid, Fn, Gm, x0, x1, x_alt, n, m, B=100, q=0.95, rng=None):
	"""
	Bootstrap-calibrated tolerance for T at the given alpha.
	- Fit H_hat by concave projection of residual CDF on [x0, x1].
	- Simulate mixture samples from (1-alpha)*H_hat + alpha*G_hat (G_hat=alt ECDF),
	  and independent alt samples of size m.
	- Compute T* for each bootstrap and return the empirical q-quantile as tol.

	Note: Keep B modest (e.g., 50-200) if called repeatedly during search.
	"""
	if rng is None:
		rng = np.random.default_rng()

	# Residual CDF and its concave projection to get H_hat
	R = (Fn - alpha*Gm) / max(1e-12, (1.0 - alpha))
	H_hat = _concave_projection_on_interval(x_grid, R, x0, x1)
	# Ensure H_hat is a valid CDF on the grid (clip tiny violations)
	H_hat = np.clip(H_hat, 0.0, 1.0)
	H_hat = np.maximum.accumulate(H_hat)  # enforce monotonicity

	# Precompute for inverse-CDF sampling
	Hx, Hcdf = x_grid, H_hat
	alt_sorted = np.sort(x_alt)

	Tstar = np.empty(B, dtype=float)
	for b in range(B):
		# Simulate mixture of size n
		u = rng.random(n)
		k_alt = np.sum(u < alpha)
		k_bg = n - k_alt
		x_bg = _sample_from_cdf(Hx, Hcdf, k_bg, rng)
		if k_alt > 0:
			idx_alt = rng.integers(low=0, high=len(alt_sorted), size=k_alt)
			x_as = alt_sorted[idx_alt]
			x_mix_b = np.concatenate([x_bg, x_as])
		else:
			x_mix_b = x_bg
		x_mix_b.sort(kind='mergesort')

		# Independent alt sample of size m (for Gm*)
		idx_alt2 = rng.integers(low=0, high=len(alt_sorted), size=m)
		x_alt_b = np.sort(alt_sorted[idx_alt2])

		# ECDFs on the same grid
		Fn_b = _ecdf_on_grid(x_mix_b, x_grid)
		Gm_b = _ecdf_on_grid(x_alt_b, x_grid)

		# Residual and its LCM gap on [x0,x1]
		R_b = (Fn_b - alpha*Gm_b) / max(1e-12, (1.0 - alpha))
		Tstar[b] = _gap_T(x_grid, R_b, x0, x1)

	# Empirical quantile as tolerance
	q = float(q)
	q = min(max(q, 0.0), 1.0)
	return float(np.quantile(Tstar, q, method='higher'))

# # =========================================
# # Feasibility, bracketing, and bisection
# # =========================================

def _build_grid(x_mix, x_alt, x0, x1, max_points=5000):
	"""
	Build a sorted grid over [min, x1] including all distinct sample points,
	and ensuring x0 and x1 are present. Caps size to max_points by thinning.
	"""
	xmin = float(np.min([x_mix.min(), x_alt.min(), x0]))
	xmax = float(x1)
	base = np.unique(np.concatenate([x_mix, x_alt, np.array([x0, xmax], dtype=float)]))
	base = base[(base >= xmin) & (base <= xmax)]
	if len(base) <= max_points:
		return base
	# Thin the grid by quantiles if too dense
	qs = np.linspace(0.0, 1.0, num=max_points)
	return np.quantile(base, qs, method='linear')

def feasible_at_alpha(alpha, Fn, Gm, x_grid, x0, x1, tol_value, cdf_eps=1e-6, valid_eps=None):
	"""
	Check feasibility at a given alpha:
	- R = (Fn - alpha*Gm)/(1-alpha) is a valid CDF (global), allowing for sampling noise
	  via either a fixed 'cdf_eps' or a principled 'valid_eps' (tuple) derived from DKW or bootstrap.
	- T_alpha = ∫ (LCM(R) - R)^2 dx on [x0,x1] <= tol_value
	Returns (is_feasible, T_alpha, R)

	Parameters
	----------
	alpha : float
	Fn, Gm : np.ndarray
	x_grid : np.ndarray
	x0, x1 : floats
	tol_value : float
	cdf_eps : float
		Legacy single-parameter epsilon (used only if valid_eps is None).
	valid_eps : tuple or None
		If not None, a pair (eps_range, eps_mon) used to judge global CDF validity:
		- range: y[0] >= -eps_range and y[-1] <= 1 + eps_range
		- monotonicity: np.diff(y) >= -eps_mon
	"""
	if alpha >= 1.0 - 1e-12:
		return (False, np.inf, None)

	R = (Fn - alpha*Gm) / max(1e-12, (1.0 - alpha))

	# Global CDF validity, with principled tolerances if provided
	if valid_eps is not None:
		eps_range, eps_mon = valid_eps
		# Range checks (allow ±eps_range)
		if (R[0] < -eps_range) or (R[-1] > 1.0 + eps_range):
			return (False, np.inf, R)
		# Monotonicity (allow small negative dips up to eps_mon)
		if np.any(np.diff(R) < -float(eps_mon)):
			return (False, np.inf, R)
	else:
		# Legacy fixed-eps check
		if not _is_valid_cdf(x_grid, R, eps=cdf_eps):
			return (False, np.inf, R)

	# LCM gap on the test interval
	Ta = _gap_T(x_grid, R, x0, x1)
	return (Ta <= float(tol_value), Ta, R)


def alpha_minimize(
	x_mix,
	x_alt,
	x0,
	alpha_tol=1e-4,
	tol_mode='bootstrap',      # 'bootstrap' or 'dkw'  (for concavity tolerance)
	B=100,                     # bootstraps per feasibility check (bootstrap mode)
	delta_total=0.05,          # overall error budget (only used for union-bound spending)
	per_check_delta=None,      # fixed per-check tail prob (bypasses union-bound if set)
	max_checks=32,             # hard cap to prevent runaway loops
	random_state=None,
	grid=None,
	initial_bracket=None,      # optional tuple (alpha_lo, alpha_hi) to start from (option b)
	verbose=False,             # print bracketing/bisection progress
	validity_mode='dkw'        # NEW: validity tolerance for CDF check: 'dkw' (fast, default) or 'bootstrap'
):
	"""
	Main entry: compute the smallest alpha such that the residual CDF is concave on [x0,0]
	(up to a tolerance and possibly accounting for sampling noise), with either coarse-grid
	bracketing (a) or an optional initial bracket (b) supplied by conservative bounds.

	Parameters
	----------
	x_mix : array_like
		i.i.d. mixture sample (scores on (-inf, 0])
	x_alt : array_like
		i.i.d. alternative sample (scores on (-inf, 0])
	x0    : float
		left endpoint of the interval [x0, 0] where background density must be decreasing
	alpha_tol : float
		bisection tolerance on alpha
	tol_mode : {'bootstrap','dkw'}
		method to set per-check tolerance for the LCM gap statistic (concavity)
	B : int
		number of bootstrap replicates per alpha check (bootstrap mode)
	delta_total : float
		overall error budget if you want finite-sample family-wise control via a union bound
	per_check_delta : float or None
		if not None, use a fixed per-check tail level at every feasibility check:
		- tol_mode='dkw'       → per-check delta = per_check_delta (S=1)
		- tol_mode='bootstrap' → per-check quantile q = 1 - per_check_delta
	max_checks : int
		safety cap on the total number of feasibility checks (loop guard only; ignored by fixed-delta)
	random_state : int or numpy.random.Generator or None
		RNG seed/control for reproducible bootstrap
	grid : array or None
		if provided, use this grid; else auto-build from samples and {x0,0}
	initial_bracket : tuple or None
		if provided, (lo, hi) to initialize the search region (option b)
	verbose : bool
		if True, print bracketing and bisection progress
	validity_mode : {'dkw','bootstrap'}
		NEW: how to gauge sampling deviations in the global CDF validity check:
		- 'dkw'       → allow violations within DKW-based epsilons (fast, default)
		- 'bootstrap' → allow violations within a bootstrap-quantile of the max violation magnitude

	Returns
	-------
	alpha_hat : float
	H_hat     : np.ndarray
		concave-projected background CDF on the grid (full support), values in [0,1]
	grid_x    : np.ndarray
		grid used for all calculations
	info      : dict
		diagnostics including 'T_at_alpha', 'tol_at_alpha', 'mode', and 'checks_budget' (if union-bound spending is used)
	"""
	rng = np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))

	x1 = 0.0
	L = abs(x1 - x0)

	if grid is None:
		grid_x = _build_grid(x_mix, x_alt, x0, x1)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		if grid_x[0] > x0:
			grid_x = np.concatenate([[x0], grid_x])
		if grid_x[-1] < x1:
			grid_x = np.concatenate([grid_x, [x1]])

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	n, m = len(x_mix), len(x_alt)
	
	# -----------------------------
	# Validity tolerances per alpha
	# -----------------------------
	def _per_check_delta(S):
		"""Per-check tail level: fixed per_check_delta if provided; else delta_total / S."""
		return float(per_check_delta) if (per_check_delta is not None) else (float(delta_total) / max(1, S))

	def _valid_eps_dkw(a, S):
		"""DKW-based epsilons for validity: (eps_range, eps_mon) = (eps_R, 2*eps_R)."""
		delta = _per_check_delta(S)
		eps_n = sqrt( max(0.0, log(2.0/delta)) / (2.0*max(1, n)) )
		eps_m = sqrt( max(0.0, log(2.0/delta)) / (2.0*max(1, m)) )
		den = max(1e-12, (1.0 - a))
		eps_R = (eps_n + a*eps_m) / den
		return (eps_R, 2.0*eps_R)

	def _valid_eps_bootstrap(a, S):
		"""
		Bootstrap validity tolerances for global CDF checks.
		Returns a pair (eps_range, eps_mon), where:
		- eps_range is the q-quantile of range violations (max of below-0 and above-1),
		- eps_mon   is the q-quantile of the maximum negative adjacent difference.
		"""
		# Quantile level q consistent with concavity side (and estimable with B)
		q_raw = 1.0 - _per_check_delta(S)
		q_max = 1.0 - 1.0 / (B + 1.0)
		q = min(q_raw, q_max)
	
		# Fit H_hat from current residual (on [x0,x1]) to simulate background
		R_now = (Fn - a*Gm) / max(1e-12, (1.0 - a))
		H_hat = _concave_projection_on_interval(grid_x, R_now, x0, x1)
		H_hat = np.maximum.accumulate(np.clip(H_hat, 0.0, 1.0))
	
		alt_sorted = np.sort(x_alt)
	
		# Collect separate violation magnitudes across replicates
		neg_dips  = np.empty(B, dtype=float)  # monotonicity: max negative diff
		low_viol  = np.empty(B, dtype=float)  # range: amount below 0
		high_viol = np.empty(B, dtype=float)  # range: amount above 1
	
		for b in range(B):
			# Simulate mixture of size n at this alpha
			u = rng.random(n)
			k_alt = np.sum(u < a)
			k_bg  = n - k_alt
			x_bg  = _sample_from_cdf(grid_x, H_hat, k_bg, rng)
			if k_alt > 0:
				idx_alt = rng.integers(low=0, high=len(alt_sorted), size=k_alt)
				x_as = alt_sorted[idx_alt]
				x_mix_b = np.concatenate([x_bg, x_as])
			else:
				x_mix_b = x_bg
			x_mix_b.sort(kind='mergesort')
	
			# Independent alt sample of size m (for Gm*)
			idx_alt2 = rng.integers(low=0, high=len(alt_sorted), size=m)
			x_alt_b = np.sort(alt_sorted[idx_alt2])
	
			# ECDFs on the same grid
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
	
			# Residual and its validity violations
			R_b = (Fn_b - a*Gm_b) / max(1e-12, (1.0 - a))
			diffs = np.diff(R_b)
			neg_dips[b]  = max(0.0, -float(np.min(diffs))) if len(diffs) else 0.0
			low_viol[b]  = max(0.0, -float(np.min(R_b)))
			high_viol[b] = max(0.0, float(np.max(R_b) - 1.0))
	
		# Per-check bootstrap quantiles for validity
		eps_mon   = float(np.quantile(neg_dips,  q, method='higher'))
		eps_low   = float(np.quantile(low_viol,  q, method='higher'))
		eps_high  = float(np.quantile(high_viol, q, method='higher'))
		eps_range = max(eps_low, eps_high)
	
		return (eps_range, eps_mon)

	def _get_valid_eps(a, S):
		return _valid_eps_bootstrap(a, S)

	# -----------------------------
	# Concavity tolerance per alpha
	# -----------------------------
	def _get_tol(a, checks_budget):
		if tol_mode == 'bootstrap':
			if per_check_delta is not None:
				q = 1.0 - float(per_check_delta)
				q_max = 1.0 - 1.0 / (B + 1.0)  # largest estimable quantile with B draws
				if q > q_max + 1e-12:
					warnings.warn(
						f"[alpha_minimize] per_check_delta={per_check_delta:.6f} "
						f"implies q={q:.6f} > q_max(B)={q_max:.6f}; clamping to q_max."
					)
					q = q_max
				return tol_bootstrap(a, grid_x, Fn, Gm, x0, x1, x_alt, n, m, B=B, q=q, rng=rng)
			# union-bound spending
			q_raw = 1.0 - float(delta_total) / max(1, checks_budget)
			q_max = 1.0 - 1.0 / (B + 1.0)  # largest estimable quantile with B draws
			if q_raw > q_max + 1e-12:
				B_min_required = int(math.ceil(1.0 / (1.0 - q_raw)) - 1.0)
				warnings.warn(
					f"[alpha_minimize] Per-check quantile q_raw={q_raw:.6f} exceeds "
					f"the maximum estimable with B={B} (q_max={q_max:.6f}). "
					f"Clamping to q_max. To attain q_raw, increase B to at least {B_min_required}."
				)
			q = min(q_raw, q_max)
			return tol_bootstrap(a, grid_x, Fn, Gm, x0, x1, x_alt, n, m, B=B, q=q, rng=rng)
		else:
			raise ValueError("tol_mode must be 'bootstrap' or 'dkw'.")

	# --- First feasibility check at alpha = 0 ---
	# Use a SAFE temporary budget to avoid a spurious clamp warning on the first call.
	# Choose S_temp so that delta_total / S_temp >= 1/(B+1) (i.e., q_raw <= q_max), bounded by max_checks.
	if per_check_delta is None:
		S_temp = max(1, min(max_checks, int(math.floor(delta_total * (B + 1)))))
	else:
		S_temp = 1
	tol0 = _get_tol(0.0, checks_budget=S_temp)
	valid0 = _get_valid_eps(0.0, S_temp)
	ok0, T0, R0 = feasible_at_alpha(0.0, Fn, Gm, grid_x, x0, x1, tol0, valid_eps=valid0)
	if ok0:
		alpha_hat = 0.0
		H_hat = _concave_projection_on_interval(grid_x, R0, x0, x1)
		H_hat = np.clip(np.maximum.accumulate(H_hat), 0.0, 1.0)
		return alpha_hat, H_hat, grid_x, {
			'T_at_alpha': float(T0),
			'tol_at_alpha': float(tol0),
			'mode': tol_mode,
			'checks_budget': 1
		}

	# --- Bracketing ---
	alpha_lo, alpha_hi = None, None
	bracket_checks = 0

	if initial_bracket is not None:
		# Option (b): use provided bracket [lo, hi] and scan inside it to find first feasible point
		lo, hi = initial_bracket
		lo = max(0.0, float(lo))
		hi = min(0.999999, float(hi))
		if not (0.0 <= lo < hi < 1.0):
			raise ValueError("initial_bracket must satisfy 0 <= lo < hi < 1.")
		# Coarse probes inside [lo, hi] to locate first feasible alpha
		probes = np.linspace(lo, hi, num=9)
		last_infeasible, found = lo, False
		for a in probes:
			tol_a = _get_tol(a, checks_budget=S_temp)
			valid_a = _get_valid_eps(a, S_temp)
			ok_a, T_a, R_a = feasible_at_alpha(a, Fn, Gm, grid_x, x0, x1, tol_a, valid_eps=valid_a)
			bracket_checks += 1
			if ok_a:
				alpha_lo, alpha_hi = last_infeasible, a
				found = True
				break
			else:
				last_infeasible = a
		if not found:
			# If no feasible point found inside the bracket, try the right endpoint
			tol_hi = _get_tol(hi, checks_budget=S_temp)
			valid_hi = _get_valid_eps(hi, S_temp)
			ok_hi, T_hi, R_hi = feasible_at_alpha(hi, Fn, Gm, grid_x, x0, x1, tol_hi, valid_eps=valid_hi)
			bracket_checks += 1
			if ok_hi:
				alpha_lo, alpha_hi = last_infeasible, hi
			else:
				# Could not find a feasible point—return diagnostic
				return np.nan, None, grid_x, {
					'T_at_alpha': float(T_hi),
					'tol_at_alpha': float(tol_hi),
					'mode': tol_mode,
					'checks_budget': bracket_checks
				}

	# --- Single (constant) budget for remaining checks (unused if per_check_delta is set) ---
	exp_bisect = int(math.ceil(math.log2(max((alpha_hi - alpha_lo)/max(alpha_tol, 1e-12), 1.0))))
	checks_budget = min(max_checks, bracket_checks + exp_bisect + 2)

	# --- Bisection ---
	checks = bracket_checks
	iter_idx = 0
	last_T, last_tol = np.nan, np.nan
	while (alpha_hi - alpha_lo) > alpha_tol and checks < max_checks:
		iter_idx += 1
		mid = 0.5 * (alpha_lo + alpha_hi)
		tol_mid = _get_tol(mid, checks_budget=checks_budget)
		valid_mid = _get_valid_eps(mid, checks_budget)
		ok_mid, T_mid, R_mid = feasible_at_alpha(mid, Fn, Gm, grid_x, x0, x1, tol_mid, valid_eps=valid_mid)
		if ok_mid:
			alpha_hi = mid
			last_T, last_tol = T_mid, tol_mid
		else:
			alpha_lo = mid
			last_T, last_tol = T_mid, tol_mid
		checks += 1

	alpha_hat = alpha_hi
	Rf = (Fn - alpha_hat*Gm) / max(1e-12, (1.0 - alpha_hat))
	H_hat = _concave_projection_on_interval(grid_x, Rf, x0, x1)
	H_hat = np.clip(np.maximum.accumulate(H_hat), 0.0, 1.0)
	return alpha_hat, H_hat, grid_x, {
		'T_at_alpha': float(last_T),
		'tol_at_alpha': float(last_tol),
		'mode': tol_mode,
		'checks_budget': checks_budget
	}


def mixture_sanity_check(Fn, Gm, H_hat, alpha_hat, x_grid,
	mode='dkw', delta=0.05, B=400, rng=None, return_sim=False):
	"""
	Post-fit sanity check for the mixture:
	    M_hat = (1 - alpha_hat) * H_hat + alpha_hat * Gm
	Compare the empirical mixture ECDF F_n to M_hat on the same grid.

	Returns dict:
	  - 'D_ks'   : sup_x |F_n - M_hat|
	  - 'ISE'    : ∫ (F_n - M_hat)^2 dx  (trapezoid on x_grid)
	  - 'crit'   : DKW ε_n (mode='dkw') or bootstrap (1-δ)-quantile (mode='bootstrap')
	  - 'p_value': bootstrap Monte Carlo p-value (None in 'dkw')
	  - 'pass_test'   : D_ks <= crit
	  - 'M_hat'  : fitted mixture CDF on x_grid
	"""
	if rng is None:
		rng = np.random.default_rng()

	# Fitted mixture CDF (guard against tiny numeric drifts)
	M_hat = (1.0 - float(alpha_hat)) * np.asarray(H_hat, dtype=float) + float(alpha_hat) * np.asarray(Gm, dtype=float)
	M_hat = np.minimum(1.0, np.maximum.accumulate(np.maximum(M_hat, 0.0)))

	diff = np.asarray(Fn, dtype=float) - M_hat
	D_ks = float(np.max(np.abs(diff)))

	# L2 over x via trapezoid
	if len(x_grid) >= 2:
		ISE = float(np.sum(0.5 * (diff[:-1]**2 + diff[1:]**2) * np.diff(x_grid)))
	else:
		ISE = 0.0

	if mode == 'dkw':
		n = max(1, int(round(1.0 / (Fn[1] - Fn[0]))) ) if len(Fn) > 1 else 1  # crude but OK: n only enters ε_n
		eps_n = sqrt(max(0.0, log(2.0 / float(delta))) / (2.0 * n))
		return dict(D_ks=D_ks, ISE=ISE, crit=eps_n, p_value=None, pass_test=(D_ks <= eps_n), M_hat=M_hat)

	# Bootstrap calibration (fixed-model; conservative)
	Dstar = np.empty(B, dtype=float)
	for b in range(B):
		x_b = _sample_from_cdf(x_grid, M_hat, size=int(round(1.0 / (Fn[1] - Fn[0]))) if len(Fn) > 1 else len(x_grid), rng=rng)
		x_b.sort(kind='mergesort')
		Fn_b = _ecdf_on_grid(x_b, x_grid)
		Dstar[b] = float(np.max(np.abs(Fn_b - M_hat)))

	q = 1.0 - float(delta)
	crit = float(np.quantile(Dstar, q, method='higher'))
	pv = float((1 + np.sum(Dstar >= D_ks)) / (B + 1))
	out = dict(D_ks=D_ks, ISE=ISE, crit=crit, p_value=pv, pass_test=(D_ks <= crit), M_hat=M_hat)
	if return_sim:
		out['Dstar'] = Dstar
	return out
import warnings
import math
import numpy as np
from dataclasses import dataclass
import warnings

def _random_state_label(random_state):
	"""Human-readable RNG seed/control value for verbose diagnostics."""
	if isinstance(random_state, np.random.Generator):
		return f"Generator({type(random_state.bit_generator).__name__})"
	return repr(random_state)

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

def _convex_projection_on_interval(x, y, a, b):
	"""
	Convex projection: replace y on [a,b] by its greatest convex minorant,
	implemented as the negative LCM of -y; return the projected vector on the full grid.
	"""
	return -_lcm_on_interval(x, -np.asarray(y, dtype=float), a, b)

def _convex_gap_T(x, y, a, b):
	"""
	Integrated squared convex gap on [a,b]:
	T = ∫ [y - GCM(y)]^2 dx, computed via the LCM gap of -y.
	"""
	return _gap_T(x, -np.asarray(y, dtype=float), a, b)

def _as_valid_cdf(y):
	"""
	Clip and monotonize a grid CDF for simulation from a fitted model.
	"""
	out = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
	return np.maximum.accumulate(out)

def _cdf_violation_stats(y):
	"""
	Full-grid CDF violation magnitudes:
	- range: maximum amount below 0 or above 1
	- monotonicity: maximum negative adjacent increment
	"""
	y = np.asarray(y, dtype=float)
	if len(y) == 0:
		return dict(range=0.0, low=0.0, high=0.0, mon=0.0)
	low = max(0.0, -float(np.min(y)))
	high = max(0.0, float(np.max(y) - 1.0))
	if len(y) > 1:
		mon = max(0.0, -float(np.min(np.diff(y))))
	else:
		mon = 0.0
	return dict(range=max(low, high), low=low, high=high, mon=mon)

def _ise_on_grid(x, diff):
	"""
	Trapezoidal integrated squared error for a difference vector on x.
	"""
	x = np.asarray(x, dtype=float)
	diff = np.asarray(diff, dtype=float)
	if len(x) < 2:
		return 0.0
	return float(np.sum(0.5 * (diff[:-1]**2 + diff[1:]**2) * np.diff(x)))

def _chord_on_interval(x, y, a, b):
	"""
	Replace y on [a,b] by the line segment connecting its endpoint values.
	"""
	x = np.asarray(x, dtype=float)
	out = np.asarray(y, dtype=float).copy()
	if a > b:
		a, b = b, a
	ia = np.searchsorted(x, a, side='left')
	ib = np.searchsorted(x, b, side='right') - 1
	ia = max(0, min(ia, len(x)-1))
	ib = max(0, min(ib, len(x)-1))
	if ia >= ib:
		return out
	x0 = float(x[ia])
	x1 = float(x[ib])
	if x1 <= x0:
		return out
	y0 = float(out[ia])
	y1 = float(out[ib])
	idx = np.arange(ia, ib + 1)
	out[idx] = y0 + (y1 - y0) * (x[idx] - x0) / (x1 - x0)
	return out

def _flat_tail_stat(x, h_hat, a, b):
	"""
	Integrated squared departure of a fitted tail CDF from its endpoint chord.
	"""
	chord = _chord_on_interval(x, h_hat, a, b)
	mask = (x >= min(a, b)) & (x <= max(a, b))
	return _ise_on_grid(x[mask], np.asarray(h_hat, dtype=float)[mask] - chord[mask]), chord

def _bootstrap_quantile_level(delta, B, label):
	"""
	Return the per-call bootstrap quantile level, clamped to the largest level
	estimable with B draws.
	"""
	if B < 1:
		raise ValueError("B must be at least 1.")
	if not (0.0 < float(delta) < 1.0):
		raise ValueError("delta must satisfy 0 < delta < 1.")
	q_raw = 1.0 - float(delta)
	q_max = 1.0 - 1.0 / (B + 1.0)
	if q_raw > q_max + 1e-12:
		warnings.warn(
			f"[{label}] delta={delta:.6f} implies q={q_raw:.6f} > "
			f"q_max(B)={q_max:.6f}; clamping to q_max."
		)
	return min(q_raw, q_max)

# =========================================
# Tolerance providers (Bootstrap)
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

def _run_shape_alpha_bracket(
	alpha_left,
	alpha_right,
	n_alpha_probes,
	alpha_tol,
	max_checks,
	total_checks,
	evaluate_shape_path,
	shape_probe_record,
	vprint,
	refine=True
):
	"""
	Shared alpha bracketing/refinement for searches whose lower boundary is
	defined by a shape-passing residual cutoff.

	The caller owns all statistical state through evaluate_shape_path(). This
	helper owns only the alpha probe grid, first-pass bracketing, and optional
	bisection trace so related procedures cannot drift.
	"""
	refine = bool(refine)
	probe_alphas = np.linspace(float(alpha_left), float(alpha_right), num=int(n_alpha_probes))
	shape_probe_table = []
	last_shape_fail = None
	first_shape_pass = None
	vprint(f"[shape-bracket] probing up to {int(n_alpha_probes)} alphas")
	for alpha_probe in probe_alphas:
		if total_checks() >= max_checks:
			break
		ok, pos, diag, meta = evaluate_shape_path(float(alpha_probe), label="shape-probe")
		shape_probe_table.append(shape_probe_record(diag, meta))
		if ok:
			first_shape_pass = {'alpha': float(alpha_probe), 'pos': pos, 'diag': diag, 'meta': meta}
			x0_msg = ''
			if diag is not None and diag.get('x0') is not None:
				x0_msg = f", x0={float(diag['x0']):.6g}"
			vprint(f"[shape-bracket] first shape pass alpha={float(alpha_probe):.6f}{x0_msg}")
			break
		last_shape_fail = {'alpha': float(alpha_probe), 'pos': pos, 'diag': diag, 'meta': meta}

	if first_shape_pass is None:
		return {
			'success': False,
			'reason': 'no_shape_passing_alpha_in_probe_grid',
			'probe_alphas': probe_alphas,
			'shape_probe_table': shape_probe_table,
			'last_shape_fail': last_shape_fail,
			'first_shape_pass': None,
			'alpha_shape': None,
			'alpha_shape_pos': None,
			'alpha_shape_diag': None,
			'shape_boundary': None
		}

	if last_shape_fail is None:
		alpha_shape_diag = first_shape_pass['diag']
		alpha_shape_pos = first_shape_pass['pos']
		shape_boundary = {
			'status': 'left_endpoint_shape_passing',
			'lo': float(alpha_left),
			'hi': float(first_shape_pass['alpha']),
			'width': 0.0
		}
	else:
		lo_alpha = float(last_shape_fail['alpha'])
		hi_alpha = float(first_shape_pass['alpha'])
		alpha_shape_diag = first_shape_pass['diag']
		alpha_shape_pos = first_shape_pass['pos']
		status = 'alpha_tol_reached'
		if refine:
			iter_idx = 0
			while (hi_alpha - lo_alpha) > alpha_tol and total_checks() < max_checks:
				iter_idx += 1
				mid = 0.5 * (lo_alpha + hi_alpha)
				vprint(f"[shape-bisect #{iter_idx:02d}] lo={lo_alpha:.6f}, hi={hi_alpha:.6f}, mid={mid:.6f}")
				ok, pos, diag, meta = evaluate_shape_path(mid, label="shape-bisect")
				if ok:
					hi_alpha = mid
					alpha_shape_diag = diag
					alpha_shape_pos = pos
				else:
					lo_alpha = mid
			if (hi_alpha - lo_alpha) > alpha_tol:
				status = 'max_checks_reached'
		else:
			status = 'coarse_first_pass'
		shape_boundary = {
			'status': status,
			'lo': float(lo_alpha),
			'hi': float(hi_alpha),
			'width': float(hi_alpha - lo_alpha)
		}

	return {
		'success': True,
		'reason': 'shape_boundary_found',
		'probe_alphas': probe_alphas,
		'shape_probe_table': shape_probe_table,
		'last_shape_fail': last_shape_fail,
		'first_shape_pass': first_shape_pass,
		'alpha_shape': float(alpha_shape_diag['alpha']),
		'alpha_shape_pos': alpha_shape_pos,
		'alpha_shape_diag': alpha_shape_diag,
		'shape_boundary': shape_boundary
	}

def _merge_alpha_probe_tables(shape_rows, cdf_rows):
	rows = {}
	for row in shape_rows:
		a = float(row['alpha'])
		rows.setdefault(a, {'alpha': a}).update(row)
	for row in cdf_rows:
		a = float(row['alpha'])
		rows.setdefault(a, {'alpha': a}).update(row)
	out = []
	for a in sorted(rows):
		row = rows[a]
		if 'shape_ok' in row and 'cdf_ok' in row:
			row['full_ok'] = bool(row['shape_ok'] and row['cdf_ok'])
		else:
			row['full_ok'] = None
		out.append(row)
	return out

def _run_cdf_alpha_boundary(
	probe_alphas,
	alpha_right,
	alpha_tol,
	max_checks,
	total_checks,
	evaluate_cdf,
	compact_cdf,
	vprint,
	stop_after_pass_at=None
):
	"""
	Shared CDF upper-boundary scan/refinement for separated alpha searches.

	When stop_after_pass_at is supplied, that alpha has already been selected
	as the shape boundary. The helper treats it as the first CDF point to test;
	if it fails, the coarse CDF scan proceeds downward until it finds a pass.
	"""
	stop_after_pass_at = None if stop_after_pass_at is None else float(stop_after_pass_at)

	def _maybe_return_overlap_witness(cdf_diag, cdf_probe_table, source):
		if stop_after_pass_at is None:
			return None
		if not cdf_diag.get('cdf_ok', False):
			return None
		if float(cdf_diag['alpha']) + 1e-15 < stop_after_pass_at:
			return None
		return {
			'success': True,
			'reason': 'cdf_overlap_witness',
			'cdf_probe_table': cdf_probe_table,
			'last_cdf_pass': {'alpha': float(cdf_diag['alpha']), 'diag': cdf_diag},
			'first_cdf_fail': first_cdf_fail,
			'alpha_cdf': float(cdf_diag['alpha']),
			'alpha_cdf_diag': cdf_diag,
			'cdf_boundary': {
				'status': 'overlap_witness_not_refined',
				'source': source,
				'lo': float(cdf_diag['alpha']),
				'hi': None,
				'width': 0.0,
				'stop_after_pass_at': float(stop_after_pass_at)
			}
		}

	cdf_probe_table = []
	last_cdf_pass = None
	first_cdf_fail = None
	cdf_scan_exhausted = True
	if stop_after_pass_at is not None:
		vprint(f"[cdf-scan] evaluating CDF boundary probes downward from alpha_shape={stop_after_pass_at:.6f}")
		if total_checks() < max_checks:
			shape_cdf_diag = evaluate_cdf(stop_after_pass_at, label="cdf-probe-shape")
			cdf_probe_table.append(compact_cdf(shape_cdf_diag))
			if shape_cdf_diag.get('cdf_ok', False):
				overlap_witness = _maybe_return_overlap_witness(shape_cdf_diag, cdf_probe_table, 'shape')
				if overlap_witness is not None:
					vprint(
						f"[cdf-scan] CDF pass at alpha_shape={stop_after_pass_at:.6f}; stopping"
					)
					return overlap_witness
			else:
				first_cdf_fail = {'alpha': float(stop_after_pass_at), 'diag': shape_cdf_diag}
		else:
			cdf_scan_exhausted = False

		descending_probes = sorted(
			{float(a) for a in probe_alphas if float(a) < stop_after_pass_at - 1e-15},
			reverse=True
		)
		for alpha_probe in descending_probes:
			if total_checks() >= max_checks:
				cdf_scan_exhausted = False
				break
			cdf_diag = evaluate_cdf(float(alpha_probe), label="cdf-probe")
			cdf_probe_table.append(compact_cdf(cdf_diag))
			if cdf_diag.get('cdf_ok', False):
				last_cdf_pass = {'alpha': float(alpha_probe), 'diag': cdf_diag}
				vprint(
					f"[cdf-scan] first CDF pass below failing alpha at alpha={float(alpha_probe):.6f}; "
					f"bracket=({float(alpha_probe):.6f}, {first_cdf_fail['alpha']:.6f})"
				)
				break
			first_cdf_fail = {'alpha': float(alpha_probe), 'diag': cdf_diag}
		if first_cdf_fail is None and last_cdf_pass is None:
			cdf_scan_exhausted = False
	else:
		vprint("[cdf-scan] evaluating CDF boundary probes")
		for alpha_probe in probe_alphas:
			if total_checks() >= max_checks:
				cdf_scan_exhausted = False
				break
			cdf_diag = evaluate_cdf(float(alpha_probe), label="cdf-probe")
			cdf_probe_table.append(compact_cdf(cdf_diag))
			if cdf_diag.get('cdf_ok', False):
				last_cdf_pass = {'alpha': float(alpha_probe), 'diag': cdf_diag}
				continue
			first_cdf_fail = {'alpha': float(alpha_probe), 'diag': cdf_diag}
			if last_cdf_pass is None:
				vprint(f"[cdf-scan] first CDF probe failed at alpha={float(alpha_probe):.6f}; stopping")
			else:
				vprint(
					f"[cdf-scan] first CDF fail after pass at alpha={float(alpha_probe):.6f}; "
					f"bracket=({last_cdf_pass['alpha']:.6f}, {float(alpha_probe):.6f})"
				)
			break

	if last_cdf_pass is None:
		return {
			'success': False,
			'reason': 'no_cdf_passing_probe',
			'cdf_probe_table': cdf_probe_table,
			'last_cdf_pass': None,
			'first_cdf_fail': first_cdf_fail,
			'alpha_cdf': None,
			'alpha_cdf_diag': None,
			'cdf_boundary': None
		}

	if first_cdf_fail is None:
		alpha_cdf_diag = last_cdf_pass['diag']
		cdf_status = 'right_endpoint_cdf_passing' if cdf_scan_exhausted else 'max_checks_reached_without_cdf_fail'
		cdf_boundary = {
			'status': cdf_status,
			'lo': float(alpha_cdf_diag['alpha']),
			'hi': float(alpha_right),
			'width': 0.0 if cdf_scan_exhausted else float(alpha_right - alpha_cdf_diag['alpha'])
		}
	else:
		lo_diag = last_cdf_pass['diag']
		hi_diag = first_cdf_fail['diag']
		lo_alpha = float(last_cdf_pass['alpha'])
		hi_alpha = float(first_cdf_fail['alpha'])
		alpha_cdf_diag = lo_diag
		status = 'alpha_tol_reached'
		iter_idx = 0
		while (hi_alpha - lo_alpha) > alpha_tol and total_checks() < max_checks:
			iter_idx += 1
			mid = 0.5 * (lo_alpha + hi_alpha)
			vprint(f"[cdf-bisect #{iter_idx:02d}] lo={lo_alpha:.6f}, hi={hi_alpha:.6f}, mid={mid:.6f}")
			mid_diag = evaluate_cdf(mid, label="cdf-bisect")
			if mid_diag.get('cdf_ok', False):
				lo_alpha = mid
				alpha_cdf_diag = mid_diag
				overlap_witness = _maybe_return_overlap_witness(mid_diag, cdf_probe_table, 'bisect')
				if overlap_witness is not None:
					vprint(
						f"[cdf-bisect] CDF pass at alpha={mid:.6f} "
						f"already overlaps alpha_shape={stop_after_pass_at:.6f}; stopping"
					)
					return overlap_witness
			else:
				hi_alpha = mid
				hi_diag = mid_diag
		if (hi_alpha - lo_alpha) > alpha_tol:
			status = 'max_checks_reached'
		cdf_boundary = {
			'status': status,
			'lo': float(lo_alpha),
			'hi': float(hi_alpha),
			'width': float(hi_alpha - lo_alpha)
		}

	return {
		'success': True,
		'reason': 'cdf_boundary_found',
		'cdf_probe_table': cdf_probe_table,
		'last_cdf_pass': last_cdf_pass,
		'first_cdf_fail': first_cdf_fail,
		'alpha_cdf': float(alpha_cdf_diag['alpha']),
		'alpha_cdf_diag': alpha_cdf_diag,
		'cdf_boundary': cdf_boundary
	}

def feasible_at_alpha(alpha, Fn, Gm, x_grid, x0, x1, tol_value, valid_eps=None):
	"""
	Check feasibility at a given alpha:
	- R = (Fn - alpha*Gm)/(1-alpha) is a valid CDF on the grid, allowing for sampling noise
	  via bootstrap validity tolerances.
	- T_alpha = ∫ (LCM(R) - R)^2 dx on [x0,x1] <= tol_value
	Returns (is_feasible, T_alpha, R)

	Parameters
	----------
	alpha : float
	Fn, Gm : np.ndarray
	x_grid : np.ndarray
	x0, x1 : floats
	tol_value : float
	valid_eps : tuple or None
		A pair (eps_range, eps_mon) used to judge grid CDF validity:
		- range: min(y) >= -eps_range and max(y) <= 1 + eps_range
		- monotonicity: np.diff(y) >= -eps_mon
	"""
	if alpha >= 1.0 - 1e-12:
		return (False, np.inf, None)

	R = (Fn - alpha*Gm) / max(1e-12, (1.0 - alpha))

	if valid_eps is None:
		raise ValueError(
			"feasible_at_alpha requires bootstrap validity tolerances via valid_eps; "
			"the legacy fixed-eps CDF check was removed."
		)

	eps_range, eps_mon = valid_eps
	# Range checks (allow ±eps_range)
	if np.min(R) < -eps_range or np.max(R) > 1.0 + eps_range:
		return (False, np.inf, R)
	# Monotonicity (allow small negative dips up to eps_mon)
	if np.any(np.diff(R) < -float(eps_mon)):
		return (False, np.inf, R)

	# LCM gap on the test interval
	Ta = _gap_T(x_grid, R, x0, x1)
	return (Ta <= float(tol_value), Ta, R)


def known_x0_alpha_minimize(
	x_mix,
	x_alt,
	x0,
	alpha_tol=1e-4,
	B=100,                     # bootstraps per feasibility check (bootstrap mode)
	delta_total=0.05,          # overall error budget (only used for union-bound spending)
	per_check_delta=None,      # fixed per-check tail prob (bypasses union-bound if set)
	max_checks=32,             # hard cap to prevent runaway loops
	random_state=None,
	grid=None,
	initial_bracket=None,      # optional tuple (alpha_lo, alpha_hi) to start from (option b)
	verbose=False              # retained for API compatibility; currently unused
):
	"""
	Main entry: compute the smallest alpha such that the residual CDF is concave on [x0,0]
	(up to a bootstrap tolerance and possibly accounting for sampling noise).

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
	B : int
		number of bootstrap replicates per alpha check
	delta_total : float
		overall error budget if you want finite-sample family-wise control via a union bound
	per_check_delta : float or None
		if not None, use a fixed per-check tail level at every feasibility check:
		- bootstrap quantile q = 1 - per_check_delta
	max_checks : int
		safety cap on the total number of feasibility checks (loop guard only; ignored by fixed-delta)
	random_state : int or numpy.random.Generator or None
	RNG seed/control for reproducible bootstrap
	grid : array or None
		if provided, use this grid; else auto-build from samples and {x0,0}
	initial_bracket : tuple or None
		(lo, hi) to initialize the search region. Required unless alpha=0 is feasible.
	verbose : bool
		retained for API compatibility; currently unused

	Returns
	-------
	alpha_hat : float
	H_hat     : np.ndarray
		concave-projected background CDF on the grid (full support), values in [0,1]
	grid_x    : np.ndarray
		grid used for all calculations
	info      : dict
		diagnostics including 'T_at_alpha', 'tol_at_alpha', 'mode', and 'checks_budget'
	"""
	rng = np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))

	x1 = 0.0

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

	# -----------------------------
	# Concavity tolerance per alpha
	# -----------------------------
	def _get_tol(a, checks_budget):
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

		# Union-bound spending.
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

	# --- First feasibility check at alpha = 0 ---
	# Use a SAFE temporary budget to avoid a spurious clamp warning on the first call.
	# Choose S_temp so that delta_total / S_temp >= 1/(B+1) (i.e., q_raw <= q_max), bounded by max_checks.
	if per_check_delta is None:
		S_temp = max(1, min(max_checks, int(math.floor(delta_total * (B + 1)))))
	else:
		S_temp = 1
	tol0 = _get_tol(0.0, checks_budget=S_temp)
	valid0 = _valid_eps_bootstrap(0.0, S_temp)
	ok0, T0, R0 = feasible_at_alpha(0.0, Fn, Gm, grid_x, x0, x1, tol0, valid_eps=valid0)
	if ok0:
		alpha_hat = 0.0
		H_hat = _concave_projection_on_interval(grid_x, R0, x0, x1)
		H_hat = np.clip(np.maximum.accumulate(H_hat), 0.0, 1.0)
		return alpha_hat, H_hat, grid_x, {
			'T_at_alpha': float(T0),
			'tol_at_alpha': float(tol0),
			'mode': 'bootstrap',
			'checks_budget': 1
		}

	# --- Bracketing ---
	bracket_checks = 0

	if initial_bracket is None:
		raise ValueError(
			"initial_bracket is required unless alpha=0 is feasible; "
			"automatic coarse bracketing was removed."
		)

	# Use provided bracket [lo, hi] and scan inside it to find first feasible point.
	lo, hi = initial_bracket
	lo = max(0.0, float(lo))
	hi = min(0.999999, float(hi))
	if not (0.0 <= lo < hi < 1.0):
		raise ValueError("initial_bracket must satisfy 0 <= lo < hi < 1.")
	probes = np.linspace(lo, hi, num=9)
	last_infeasible, found = lo, False
	for a in probes:
		tol_a = _get_tol(a, checks_budget=S_temp)
		valid_a = _valid_eps_bootstrap(a, S_temp)
		ok_a, T_a, R_a = feasible_at_alpha(a, Fn, Gm, grid_x, x0, x1, tol_a, valid_eps=valid_a)
		bracket_checks += 1
		if ok_a:
			alpha_lo, alpha_hi = last_infeasible, a
			found = True
			break
		last_infeasible = a
	if not found:
		# If no feasible point found inside the bracket, try the right endpoint.
		tol_hi = _get_tol(hi, checks_budget=S_temp)
		valid_hi = _valid_eps_bootstrap(hi, S_temp)
		ok_hi, T_hi, R_hi = feasible_at_alpha(hi, Fn, Gm, grid_x, x0, x1, tol_hi, valid_eps=valid_hi)
		bracket_checks += 1
		if ok_hi:
			alpha_lo, alpha_hi = last_infeasible, hi
		else:
			return np.nan, None, grid_x, {
				'T_at_alpha': float(T_hi),
				'tol_at_alpha': float(tol_hi),
				'mode': 'bootstrap',
				'checks_budget': bracket_checks
			}

	# --- Single (constant) budget for remaining checks (unused if per_check_delta is set) ---
	exp_bisect = int(math.ceil(math.log2(max((alpha_hi - alpha_lo)/max(alpha_tol, 1e-12), 1.0))))
	checks_budget = min(max_checks, bracket_checks + exp_bisect + 2)

	# --- Bisection ---
	checks = bracket_checks
	last_T, last_tol = np.nan, np.nan
	while (alpha_hi - alpha_lo) > alpha_tol and checks < max_checks:
		mid = 0.5 * (alpha_lo + alpha_hi)
		tol_mid = _get_tol(mid, checks_budget=checks_budget)
		valid_mid = _valid_eps_bootstrap(mid, checks_budget)
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
		'mode': 'bootstrap',
		'checks_budget': checks_budget
	}


def alpha_minimize_given_x0_revisited(
	x_mix,
	x_alt,
	x0,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	gap_tol=None,
	near_miss_probes=5,
	max_checks=64,
	random_state=None,
	grid=None,
	max_grid_points=5000,
	verbose=False
):
	"""
	Fixed-x0 alpha search with separated shape and CDF-validity boundaries.

	This implements the "given x0, revisited" blueprint:
	- scan candidate alpha values and record ShapeOK and CDFOK separately;
	- refine the lower shape boundary using ShapeOK only;
	- refine the upper CDF-validity boundary using CDFOK only;
	- return the earliest overlap, or a separated-boundaries diagnostic.

	Returns
	-------
	alpha_hat : float
	H_hat     : np.ndarray or None
	grid_x    : np.ndarray
	info      : dict
		Diagnostics including alpha_shape, alpha_cdf, gap status, and the
		final shape/CDF statistics.
	"""
	rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))
	if len(x_mix) == 0 or len(x_alt) == 0:
		raise ValueError("x_mix and x_alt must both be nonempty.")
	x0 = float(x0)
	if not np.isfinite(x0) or x0 >= 0.0:
		raise ValueError("x0 must be finite and strictly less than 0.")
	if alpha_tol <= 0:
		raise ValueError("alpha_tol must be positive.")
	B = int(B)
	if B < 1:
		raise ValueError("B must be at least 1.")
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	n_alpha_probes = int(n_alpha_probes)
	max_checks = int(max_checks)
	if max_checks < 2*n_alpha_probes:
		raise ValueError("max_checks must be at least 2*n_alpha_probes for separated shape/CDF probes.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	cdf_delta = delta if cdf_delta is None else float(cdf_delta)

	alpha_left, alpha_right = alpha_bounds
	alpha_left = max(0.0, float(alpha_left))
	alpha_right = min(0.999999, float(alpha_right))
	if not (0.0 <= alpha_left < alpha_right < 1.0):
		raise ValueError("alpha_bounds must satisfy 0 <= lo < hi < 1.")

	x1 = 0.0
	if grid is None:
		grid_x = _build_grid(x_mix, x_alt, x0, x1, max_points=max_grid_points)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		grid_x = grid_x[np.isfinite(grid_x)]
		if len(grid_x) == 0:
			raise ValueError("grid must contain at least one finite value.")
	grid_x = np.unique(np.concatenate([grid_x, np.array([x0, x1], dtype=float)]))
	grid_x = np.unique(grid_x[grid_x <= x1])
	if len(grid_x) == 0:
		raise ValueError("grid has no points at or below 0.")
	if grid_x[0] > x0:
		grid_x = np.concatenate([[x0], grid_x])
	if grid_x[-1] < x1:
		grid_x = np.concatenate([grid_x, [x1]])

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	n, m = len(x_mix), len(x_alt)
	q = _bootstrap_quantile_level(delta, B, "alpha_minimize_given_x0_revisited")
	q_cdf = _bootstrap_quantile_level(cdf_delta, B, "alpha_minimize_given_x0_revisited.cdf")
	G_hat = _as_valid_cdf(_convex_projection_on_interval(grid_x, Gm, x0, x1))
	G_diff = Gm - G_hat
	G_convex_gap = _convex_gap_T(grid_x, Gm, x0, x1)
	G_distance_ks = float(np.max(np.abs(G_diff))) if len(G_diff) else 0.0
	G_distance_ise = _ise_on_grid(grid_x, G_diff)

	def _vprint(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	def _ok_text(ok):
		return "OK" if ok else "NO"

	_vprint(
		f"[setup-given-x0] n={n}, m={m}, grid={len(grid_x)}, x0={x0:.6g}, "
		f"boot_seed={_random_state_label(random_state)}, "
		f"alpha_bounds=({alpha_left:.6f}, {alpha_right:.6f}), "
		f"alpha_tol={float(alpha_tol):.3g}, gap_tol={float(gap_tol):.3g}, "
		f"B={B}, delta={float(delta):.3g}, q={q:.6f}, "
		f"cdf_delta={float(cdf_delta):.3g}, q_cdf={q_cdf:.6f}"
	)
	_vprint(
		f"[G-fit-given-x0] convex_gap={G_convex_gap:.3g}, "
		f"distance_ks={G_distance_ks:.3g}, distance_ise={G_distance_ise:.3g}"
	)

	shape_checks = 0
	cdf_checks = 0
	base_cache = {}
	shape_cache = {}
	cdf_cache = {}
	full_cache = {}

	def _total_alpha_checks():
		return int(shape_checks + cdf_checks)

	def _fit_base(alpha):
		alpha = float(alpha)
		if alpha in base_cache:
			return base_cache[alpha]
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		base = {
			'alpha': alpha,
			'den': den,
			'R_obs': R_obs,
			'H_hat': H_hat,
			'cdf_obs': cdf_obs,
			'shape_gap': _gap_T(grid_x, R_obs, x0, x1)
		}
		base_cache[alpha] = base
		return base

	def _bootstrap_sample(alpha, H_hat):
		F_hat = _as_valid_cdf((1.0 - float(alpha))*H_hat + float(alpha)*G_hat)
		x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
		x_mix_b.sort(kind='mergesort')
		x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
		x_alt_b.sort(kind='mergesort')
		Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
		Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
		return Fn_b, Gm_b

	def _shape_at(alpha, label="shape"):
		nonlocal shape_checks
		alpha = float(alpha)
		if alpha in shape_cache:
			return shape_cache[alpha]
		shape_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha,
				'shape_ok': False,
				'shape_gap': np.inf,
				'shape_tol': np.nan,
				'cdf_range_violation': np.inf,
				'cdf_low_violation': np.inf,
				'cdf_high_violation': np.inf,
				'cdf_mon_violation': np.inf,
				'R_obs': None,
				'H_hat': None,
				'reason': 'alpha_too_close_to_one'
			}
			shape_cache[alpha] = diag
			_vprint(f"[{label} #{shape_checks}] alpha={alpha:.6f}: alpha too close to 1, ShapeOK=NO")
			return diag

		base = _fit_base(alpha)
		shape_star = np.empty(B, dtype=float)
		for b in range(B):
			Fn_b, Gm_b = _bootstrap_sample(alpha, base['H_hat'])
			R_b = (Fn_b - alpha*Gm_b) / base['den']
			shape_star[b] = _gap_T(grid_x, R_b, x0, x1)
		shape_tol = float(np.quantile(shape_star, q, method='higher'))
		shape_ok = bool(base['shape_gap'] <= shape_tol)
		cdf_obs = base['cdf_obs']
		diag = {
			'alpha': alpha,
			'shape_ok': shape_ok,
			'shape_gap': float(base['shape_gap']),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_mon_violation': float(cdf_obs['mon']),
			'R_obs': base['R_obs'],
			'H_hat': base['H_hat']
		}
		shape_cache[alpha] = diag
		_vprint(
			f"[{label} #{shape_checks}] alpha={alpha:.6f}, "
			f"shape={base['shape_gap']:.3g}/{shape_tol:.3g}, "
			f"ShapeOK={_ok_text(shape_ok)}"
		)
		return diag

	def _cdf_at(alpha, label="cdf"):
		nonlocal cdf_checks
		alpha = float(alpha)
		if alpha in cdf_cache:
			return cdf_cache[alpha]
		cdf_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha,
				'cdf_ok': False,
				'cdf_range_violation': np.inf,
				'cdf_low_violation': np.inf,
				'cdf_high_violation': np.inf,
				'cdf_range_tol': np.nan,
				'cdf_mon_violation': np.inf,
				'cdf_mon_tol': np.nan,
				'R_obs': None,
				'H_hat': None,
				'reason': 'alpha_too_close_to_one'
			}
			cdf_cache[alpha] = diag
			_vprint(f"[{label} #{cdf_checks}] alpha={alpha:.6f}: alpha too close to 1, CDFOK=NO")
			return diag

		base = _fit_base(alpha)
		range_star = np.empty(int(B), dtype=float)
		mon_star = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			Fn_b, Gm_b = _bootstrap_sample(alpha, base['H_hat'])
			R_b = (Fn_b - alpha*Gm_b) / base['den']
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
		range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		cdf_obs = base['cdf_obs']
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		diag = {
			'alpha': alpha,
			'cdf_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'R_obs': base['R_obs'],
			'H_hat': base['H_hat']
		}
		cdf_cache[alpha] = diag
		_vprint(
			f"[{label} #{cdf_checks}] alpha={alpha:.6f}, "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, "
			f"CDFOK={_ok_text(cdf_ok)}"
		)
		return diag

	def _full_diag(alpha, label="full"):
		alpha = float(alpha)
		if alpha in full_cache:
			return full_cache[alpha]
		shape_diag = _shape_at(alpha, label=f"{label}-shape")
		cdf_diag = _cdf_at(alpha, label=f"{label}-cdf")
		full_ok = bool(shape_diag['shape_ok'] and cdf_diag['cdf_ok'])
		diag = {
			'alpha': alpha,
			'shape_ok': bool(shape_diag['shape_ok']),
			'cdf_ok': bool(cdf_diag['cdf_ok']),
			'full_ok': full_ok,
			'shape_gap': float(shape_diag['shape_gap']),
			'shape_tol': float(shape_diag['shape_tol']),
			'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
			'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
			'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
			'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
			'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
			'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
			'R_obs': shape_diag.get('R_obs'),
			'H_hat': shape_diag.get('H_hat')
		}
		full_cache[alpha] = diag
		_vprint(f"[{label}] alpha={alpha:.6f}, full={_ok_text(full_ok)}")
		return diag

	def _compact_shape(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'shape_ok': bool(diag['shape_ok']),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation'])
		}

	def _compact_cdf(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'cdf_ok': bool(diag['cdf_ok']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_delta': float(diag.get('cdf_delta', cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', q_cdf))
		}

	def _compact_diag(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'shape_ok': bool(diag['shape_ok']),
			'cdf_ok': bool(diag['cdf_ok']),
			'full_ok': bool(diag['full_ok']),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_delta': float(diag.get('cdf_delta', cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', q_cdf))
		}

	def _refine_shape(fail_diag, pass_diag):
		lo = fail_diag
		hi = pass_diag
		status = 'alpha_tol_reached'
		while (hi['alpha'] - lo['alpha']) > alpha_tol and _total_alpha_checks() < max_checks:
			mid = 0.5 * (lo['alpha'] + hi['alpha'])
			mid_diag = _shape_at(mid, label="shape-bisect")
			if mid_diag['shape_ok']:
				hi = mid_diag
			else:
				lo = mid_diag
		if (hi['alpha'] - lo['alpha']) > alpha_tol:
			status = 'max_checks_reached'
		return hi, {
			'status': status,
			'lo': float(lo['alpha']),
			'hi': float(hi['alpha']),
			'width': float(hi['alpha'] - lo['alpha'])
		}

	def _make_info(success, reason, final_diag=None, extra=None):
		info = {
			'success': bool(success),
			'reason': reason,
			'mode': 'fixed_x0_separated_bootstrap',
			'alpha': float(final_diag['alpha']) if final_diag is not None else np.nan,
			'x0': float(x0),
			'alpha_checks': int(_total_alpha_checks()),
			'total_alpha_checks': int(_total_alpha_checks()),
			'shape_checks': int(shape_checks),
			'cdf_checks': int(cdf_checks),
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'alpha_tol': float(alpha_tol),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
			'max_checks': int(max_checks),
			'n_alpha_probes': int(n_alpha_probes),
			'alpha_bounds': (float(alpha_left), float(alpha_right)),
			'n_grid': int(len(grid_x)),
			'n_mix': int(n),
			'n_alt': int(m),
			'G_convex_gap': float(G_convex_gap),
			'G_distance_ks': float(G_distance_ks),
			'G_distance_ise': float(G_distance_ise)
		}
		if final_diag is not None:
			info.update(_compact_diag(final_diag))
			info['alpha'] = float(final_diag['alpha'])
			info['residual_lcm_gap'] = float(final_diag['shape_gap'])
			info['residual_lcm_tol'] = float(final_diag['shape_tol'])
		if extra:
			info.update(extra)
		return info

	probe_alphas = np.linspace(alpha_left, alpha_right, num=n_alpha_probes)
	shape_probe_table = []
	last_shape_fail = None
	first_shape_pass = None
	_vprint("[shape-scan] evaluating shape boundary probes")
	for alpha_probe in probe_alphas:
		if _total_alpha_checks() >= max_checks:
			break
		shape_diag = _shape_at(alpha_probe, label="shape-probe")
		shape_probe_table.append(_compact_shape(shape_diag))
		if shape_diag['shape_ok']:
			first_shape_pass = {'alpha': float(alpha_probe), 'diag': shape_diag}
			_vprint(f"[shape-scan] first shape pass at alpha={float(alpha_probe):.6f}")
			break
		last_shape_fail = {'alpha': float(alpha_probe), 'diag': shape_diag}

	if first_shape_pass is None:
		final_alpha = float(last_shape_fail['alpha']) if last_shape_fail is not None else float(alpha_left)
		final_diag = _full_diag(final_alpha, label="no-shape-final")
		probe_table = _merge_alpha_probe_tables(shape_probe_table, [])
		info = _make_info(
			False, 'no_shape_passing_probe', final_diag=final_diag,
			extra={
				'shape_probe_table': shape_probe_table,
				'cdf_probe_table': [],
				'coarse_probe_table': probe_table
			}
		)
		return np.nan, None, grid_x, info

	if last_shape_fail is None:
		alpha_shape_diag = first_shape_pass['diag']
		shape_boundary = {
			'status': 'left_endpoint_shape_passing',
			'lo': float(alpha_left),
			'hi': float(alpha_shape_diag['alpha']),
			'width': 0.0
		}
	else:
		alpha_shape_diag, shape_boundary = _refine_shape(
			last_shape_fail['diag'],
			first_shape_pass['diag']
		)

	alpha_shape = float(alpha_shape_diag['alpha'])
	full_at_shape = _full_diag(alpha_shape, label="direct-shape")
	if full_at_shape['full_ok']:
		cdf_probe_table = [_compact_cdf(full_at_shape)]
		probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
		shared_extra = {
			'alpha_shape': alpha_shape,
			'alpha_cdf': alpha_shape,
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {
				'status': 'direct_alpha_shape_pass',
				'lo': float(alpha_shape),
				'hi': float(alpha_shape),
				'width': 0.0
			},
			'shape_profile_has_holes': False,
			'cdf_profile_has_holes': False,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': _compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': _compact_cdf(full_at_shape)
		}
		_vprint(f"[direct-alpha-shape] alpha={alpha_shape:.6f}, ShapeOK=OK, CDFOK=OK; stopping")
		info = _make_info(
			True, 'overlap_shape_boundary', final_diag=full_at_shape,
			extra=shared_extra
		)
		return alpha_shape, full_at_shape['H_hat'], grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=alpha_right,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=_total_alpha_checks,
		evaluate_cdf=_cdf_at,
		compact_cdf=_compact_cdf,
		vprint=_vprint,
		stop_after_pass_at=alpha_shape
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
	if not cdf_boundary_result['success']:
		first_cdf_fail = cdf_boundary_result.get('first_cdf_fail')
		final_alpha = float(first_cdf_fail['alpha']) if first_cdf_fail is not None else float(alpha_shape)
		final_diag = _full_diag(final_alpha, label="no-cdf-final")
		extra = {
			'alpha_shape': alpha_shape,
			'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': _compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': None
		}
		info = _make_info(
			False, 'no_cdf_passing_probe', final_diag=final_diag, extra=extra
		)
		return np.nan, None, grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	cdf_boundary = cdf_boundary_result['cdf_boundary']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = {
		'alpha_shape': alpha_shape,
		'alpha_cdf': alpha_cdf,
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary,
		'shape_profile_has_holes': False,
		'cdf_profile_has_holes': False,
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'coarse_probe_table': probe_table,
		'alpha_shape_diag': _compact_shape(alpha_shape_diag),
		'alpha_cdf_diag': _compact_cdf(alpha_cdf_diag)
	}

	_vprint(
		f"[boundaries] alpha_shape={alpha_shape:.6f}, alpha_cdf={alpha_cdf:.6f}, "
		f"gap={gap:.3g}, status={shared_extra['gap_status']}"
	)

	def _scan_for_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or _total_alpha_checks() >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan_diags = []
		for a in alphas:
			if _total_alpha_checks() >= max_checks:
				break
			diag = _full_diag(a, label=label)
			scan_diags.append(diag)
			if diag['full_ok']:
				return diag, scan_diags
		return None, scan_diags

	if gap <= 0.0:
		full_at_shape = _full_diag(alpha_shape, label="overlap-shape")
		if full_at_shape['full_ok']:
			info = _make_info(
				True, 'overlap_shape_boundary', final_diag=full_at_shape,
				extra=shared_extra
			)
			return alpha_shape, full_at_shape['H_hat'], grid_x, info

		pass_diag, scan_diags = _scan_for_full_pass(alpha_shape, alpha_cdf, "overlap-scan")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = [_compact_diag(d) for d in scan_diags]
		if pass_diag is not None:
			info = _make_info(
				True, 'overlap_local_full_pass', final_diag=pass_diag,
				extra=extra
			)
			return float(pass_diag['alpha']), pass_diag['H_hat'], grid_x, info
		info = _make_info(
			False, 'overlap_without_full_passing_probe', final_diag=full_at_shape,
			extra=extra
		)
		return np.nan, None, grid_x, info

	if gap <= gap_tol:
		pass_diag, scan_diags = _scan_for_full_pass(alpha_cdf, alpha_shape, "near-miss")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = [_compact_diag(d) for d in scan_diags]
		if pass_diag is not None:
			info = _make_info(
				True, 'near_miss_full_passing_probe', final_diag=pass_diag,
				extra=extra
			)
			return float(pass_diag['alpha']), pass_diag['H_hat'], grid_x, info
		full_at_shape = _full_diag(alpha_shape, label="near-miss-shape")
		info = _make_info(
			False, 'near_miss_no_overlap', final_diag=full_at_shape,
			extra=extra
		)
		return np.nan, None, grid_x, info

	full_at_shape = _full_diag(alpha_shape, label="gap-shape")
	info = _make_info(
		False, 'separated_boundaries_gap', final_diag=full_at_shape,
		extra=shared_extra
	)
	return np.nan, None, grid_x, info


def alpha_minimize_unknown_x0(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0
):
	"""
	Adaptive search for the smallest feasible alpha when the cutoff x0 is unknown.

	This implements the nested alpha/cutoff search described in the adaptive
	section of minimal_match_subtraction_pava_revised.tex:
	- find the leftmost count-admissible cutoff with convex-compatible G;
	- for each alpha, test residual CDF validity globally and concavity on a
	  suffix [x_j, 0];
	- update the running residual cutoff x_R only after feasible alpha checks.
	A positive x0_tol stops the internal cutoff bisection once the candidate
	cutoff bracket has width at most x0_tol on the score scale. The returned
	cutoff is still the right endpoint of that bracket, i.e. a passing cutoff.

	Returns
	-------
	alpha_hat : float
	x0_hat    : float or None
	H_hat     : np.ndarray or None
		Final fitted residual/background CDF on the full grid.
	G_hat     : np.ndarray or None
		Anchored fitted matched CDF on the full grid, computed at x_G.
	grid_x    : np.ndarray
	info      : dict
		Diagnostics for the adaptive bootstrap search.
	"""
	rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))
	if len(x_mix) == 0 or len(x_alt) == 0:
		raise ValueError("x_mix and x_alt must both be nonempty.")
	if alpha_tol <= 0:
		raise ValueError("alpha_tol must be positive.")
	x0_tol = 0.0 if x0_tol is None else float(x0_tol)
	if not np.isfinite(x0_tol) or x0_tol < 0.0:
		raise ValueError("x0_tol must be finite and nonnegative.")
	if max_checks < 1:
		raise ValueError("max_checks must be at least 1.")
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	B = int(B)
	if B < 1:
		raise ValueError("B must be at least 1.")
	cdf_delta = delta if cdf_delta is None else float(cdf_delta)
	min_mix_tail = int(min_mix_tail)
	min_alt_tail = int(min_alt_tail)
	if min_mix_tail < 1 or min_alt_tail < 1:
		raise ValueError("min_mix_tail and min_alt_tail must be positive.")

	alpha_left, alpha_right = alpha_bounds
	alpha_left = max(0.0, float(alpha_left))
	alpha_right = min(0.999999, float(alpha_right))
	if not (0.0 <= alpha_left < alpha_right < 1.0):
		raise ValueError("alpha_bounds must satisfy 0 <= lo < hi < 1.")

	def _vprint(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	def _ok_text(ok):
		return "OK" if ok else "NO"

	def _report_invalid_r(alpha, x0, R, cdf_obs, range_tol, mon_tol):
		if not verbose or R is None:
			return
		if cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol:
			return
		diffs = np.diff(R)
		min_diff = float(np.min(diffs)) if len(diffs) else float('nan')
		msg = (
			f"[invalid ValidR] alpha={alpha:.6f}, x0={x0:.6g}, "
			f"minDelta={min_diff:.3g}, minR={float(np.min(R)):.3g}, "
			f"maxR={float(np.max(R)):.3g}"
		)
		if cdf_obs['low'] > range_tol:
			i = int(np.argmin(R))
			msg += f", R<0 at x~{grid_x[i]:.6g}"
		if cdf_obs['high'] > range_tol:
			i = int(np.argmax(R))
			msg += f", R>1 at x~{grid_x[i]:.6g}"
		if cdf_obs['mon'] > mon_tol and len(diffs):
			j = int(np.argmin(diffs))
			msg += f", nonmonotone between x~{grid_x[j]:.6g} and {grid_x[j+1]:.6g}"
		_vprint(msg)

	x1 = 0.0
	if grid is None:
		x_left = float(np.min([x_mix.min(), x_alt.min()]))
		grid_x = _build_grid(x_mix, x_alt, x_left, x1, max_points=max_grid_points)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		grid_x = grid_x[np.isfinite(grid_x)]
		if len(grid_x) == 0:
			raise ValueError("grid must contain at least one finite value.")
		if grid_x[-1] < x1:
			grid_x = np.concatenate([grid_x, [x1]])

	if candidate_cutoffs is not None:
		cutoff_values = np.sort(np.asarray(candidate_cutoffs, dtype=float))
		cutoff_values = cutoff_values[np.isfinite(cutoff_values)]
		cutoff_values = cutoff_values[cutoff_values < x1]
		grid_x = np.unique(np.concatenate([grid_x, cutoff_values, np.array([x1], dtype=float)]))
	else:
		grid_x = np.unique(np.concatenate([grid_x, np.array([x1], dtype=float)]))

	grid_x = grid_x[grid_x <= x1]
	if len(grid_x) == 0:
		raise ValueError("grid has no points at or below 0.")
	if grid_x[-1] < x1:
		grid_x = np.concatenate([grid_x, [x1]])

	_vprint(
		f"[setup] n={len(x_mix)}, m={len(x_alt)}, grid={len(grid_x)}, "
		f"boot_seed={_random_state_label(random_state)}, "
		f"alpha_bounds=({alpha_left:.6f}, {alpha_right:.6f}), "
		f"alpha_tol={float(alpha_tol):.3g}, x0_tol={float(x0_tol):.3g}, "
		f"B={int(B)}, delta={float(delta):.3g}, target_q={1.0 - float(delta):.3g}, "
		f"cdf_delta={float(cdf_delta):.3g}, cdf_target_q={1.0 - float(cdf_delta):.3g}"
	)

	if candidate_cutoffs is None:
		candidate_values = grid_x[grid_x < x1]
	else:
		candidate_values = cutoff_values
	candidate_values = np.unique(candidate_values[candidate_values < x1])
	if len(candidate_values) == 0:
		_vprint("[setup] no candidate cutoffs below 0")
		Fn_empty = _ecdf_on_grid(x_mix, grid_x)
		Gm_empty = _ecdf_on_grid(x_alt, grid_x)
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_candidate_cutoffs',
			'mode': 'adaptive_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'alpha_checks': 0,
			'cutoff_checks': 0,
			'valid_g_checks': 0,
			'valid_r_checks': 0,
			'n_grid': int(len(grid_x)),
			'n_mix': int(len(x_mix)),
			'n_alt': int(len(x_alt)),
			'Fn_at_grid_end': float(Fn_empty[-1]),
			'Gm_at_grid_end': float(Gm_empty[-1])
		}

	candidate_indices = np.array([int(np.searchsorted(grid_x, x, side='left')) for x in candidate_values], dtype=int)
	candidate_indices = np.unique(candidate_indices)
	candidate_indices = candidate_indices[grid_x[candidate_indices] < x1]
	candidate_x = grid_x[candidate_indices]

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	n, m = len(x_mix), len(x_alt)
	q = _bootstrap_quantile_level(delta, int(B), "alpha_minimize_unknown_x0")
	q_cdf = _bootstrap_quantile_level(cdf_delta, int(B), "alpha_minimize_unknown_x0.cdf")
	_vprint(f"[setup] bootstrap_quantile={q:.6f}")

	mix_tail_counts = np.array([np.sum(x_mix >= x0) for x0 in candidate_x], dtype=int)
	alt_tail_counts = np.array([np.sum(x_alt >= x0) for x0 in candidate_x], dtype=int)
	count_ok = (mix_tail_counts >= min_mix_tail) & (alt_tail_counts >= min_alt_tail)
	if not np.any(count_ok):
		_vprint(
			f"[count] no count-admissible cutoff: min_mix_tail={min_mix_tail}, "
			f"min_alt_tail={min_alt_tail}, max_mix_tail={int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0}, "
			f"max_alt_tail={int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0}"
		)
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_count_admissible_cutoff',
			'mode': 'adaptive_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'alpha_checks': 0,
			'cutoff_checks': 0,
			'valid_g_checks': 0,
			'valid_r_checks': 0,
			'n_grid': int(len(grid_x)),
			'n_mix': int(n),
			'n_alt': int(m),
			'min_mix_tail': int(min_mix_tail),
			'min_alt_tail': int(min_alt_tail),
			'max_mix_tail': int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0,
			'max_alt_tail': int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0
		}

	candidate_indices = candidate_indices[count_ok]
	candidate_x = grid_x[candidate_indices]
	x_count_pos = len(candidate_indices) - 1
	_vprint(
		f"[count] admissible_cutoffs={len(candidate_indices)}, "
		f"x_min={float(candidate_x[0]):.6g}, x_count={float(candidate_x[-1]):.6g}, "
		f"tail_at_x_count=({int(np.sum(x_mix >= candidate_x[-1]))}, {int(np.sum(x_alt >= candidate_x[-1]))})"
	)

	alpha_checks = 0
	cutoff_checks = 0
	valid_g_checks = 0
	valid_r_checks = 0
	g_cache = {}
	last_failure_diag = None

	def _tail_counts_at_pos(pos):
		x0 = float(grid_x[candidate_indices[pos]])
		return int(np.sum(x_mix >= x0)), int(np.sum(x_alt >= x0))

	def _valid_g(pos):
		nonlocal cutoff_checks, valid_g_checks
		pos = int(pos)
		if pos in g_cache:
			return g_cache[pos]

		cutoff_checks += 1
		valid_g_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		G_hat = _as_valid_cdf(_convex_projection_on_interval(grid_x, Gm, x0, x1))
		T_obs = _convex_gap_T(grid_x, Gm, x0, x1)
		Tstar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			Tstar[b] = _convex_gap_T(grid_x, Gm_b, x0, x1)
		tol = float(np.quantile(Tstar, q, method='higher'))
		diff = Gm - G_hat
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		diag = {
			'valid': bool(T_obs <= tol),
			'x0': x0,
			'pos': pos,
			'G_hat': G_hat,
			'convex_gap': float(T_obs),
			'convex_tol': tol,
			'bootstrap_quantile': float(q),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail,
			'distance_ks': float(np.max(np.abs(diff))),
			'distance_ise': _ise_on_grid(grid_x, diff)
		}
		g_cache[pos] = diag
		_vprint(
			f"[ValidG] pos={pos}, x0={x0:.6g}, tails=({mix_tail},{alt_tail}), "
			f"T={T_obs:.3g}, tol={tol:.3g}: {_ok_text(diag['valid'])}"
		)
		return diag

	def _valid_r(alpha, pos):
		nonlocal cutoff_checks, valid_r_checks
		alpha = float(alpha)
		pos = int(pos)
		cutoff_checks += 1
		valid_r_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		T_obs = _gap_T(grid_x, R_obs, x0, x1)
		g_diag = g_anchor_diag
		G_hat = g_anchor_diag['G_hat']
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat)

		range_star = np.empty(int(B), dtype=float)
		mon_star = np.empty(int(B), dtype=float)
		Tstar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
			Tstar[b] = _gap_T(grid_x, R_b, x0, x1)

		range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		shape_tol = float(np.quantile(Tstar, q, method='higher'))
		valid = (
			cdf_obs['range'] <= range_tol and
			cdf_obs['mon'] <= mon_tol and
			T_obs <= shape_tol
		)
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		_report_invalid_r(alpha, x0, R_obs, cdf_obs, range_tol, mon_tol)
		_vprint(
			f"[ValidR] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"tails=({mix_tail},{alt_tail}), "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, "
			f"T={T_obs:.3g}/{shape_tol:.3g}, "
			f"ValidR={_ok_text(valid)}"
		)
		return {
			'valid': bool(valid),
			'alpha': alpha,
			'x0': x0,
			'pos': pos,
			'H_hat': H_hat,
			'G_hat': G_hat,
			'R_obs': R_obs,
			'shape_gap': float(T_obs),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'g_valid': bool(g_diag['valid']),
			'g_x0': float(g_diag['x0']),
			'g_convex_gap': float(g_diag['convex_gap']),
			'g_convex_tol': float(g_diag['convex_tol']),
			'g_distance_ks': float(g_diag['distance_ks']),
			'g_distance_ise': float(g_diag['distance_ise']),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail
		}

	def _first_valid_g_pos():
		_vprint("[ValidG-search] locating leftmost convex-compatible matched cutoff")
		right = x_count_pos
		if _valid_g(0)['valid']:
			_vprint(f"[ValidG-search] leftmost cutoff passes: pos=0, x_G={float(grid_x[candidate_indices[0]]):.6g}")
			return 0
		if not _valid_g(right)['valid']:
			_vprint(
				f"[ValidG-search] rightmost count-admissible cutoff fails: "
				f"pos={right}, x_count={float(grid_x[candidate_indices[right]]):.6g}"
			)
			return None
		left = 0
		while right - left > 1:
			mid = (left + right) // 2
			if _valid_g(mid)['valid']:
				_vprint(f"[ValidG-search] mid pos={mid} passes; move right endpoint")
				right = mid
			else:
				_vprint(f"[ValidG-search] mid pos={mid} fails; move left endpoint")
				left = mid
		_vprint(f"[ValidG-search] x_G pos={right}, x_G={float(grid_x[candidate_indices[right]]):.6g}")
		return right

	x_g_pos = _first_valid_g_pos()
	if x_g_pos is None:
		_vprint("[failure] no_convex_compatible_matched_cutoff")
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_convex_compatible_matched_cutoff',
			'mode': 'adaptive_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'x_G': None,
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_checks': int(alpha_checks),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'valid_r_checks': int(valid_r_checks),
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices))
		}
	g_anchor_diag = _valid_g(x_g_pos)
	_vprint(
		f"[ValidG-anchor] using x_G={g_anchor_diag['x0']:.6g} "
		"for the matched bootstrap component"
	)

	def _make_info(success, reason, alpha=None, pos=None, r_diag=None, alpha_lo=None, alpha_hi=None):
		x0 = float(grid_x[candidate_indices[pos]]) if pos is not None else None
		info = {
			'success': bool(success),
			'reason': reason,
			'mode': 'adaptive_bootstrap',
			'alpha': float(alpha) if alpha is not None else np.nan,
			'x0': x0,
			'x_G': float(grid_x[candidate_indices[x_g_pos]]),
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_lo': float(alpha_lo) if alpha_lo is not None else None,
			'alpha_hi': float(alpha_hi) if alpha_hi is not None else None,
			'alpha_checks': int(alpha_checks),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'valid_r_checks': int(valid_r_checks),
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'alpha_tol': float(alpha_tol),
			'x0_tol': float(x0_tol),
			'max_checks': int(max_checks),
			'n_alpha_probes': int(n_alpha_probes),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices)),
			'n_mix': int(n),
			'n_alt': int(m),
			'min_mix_tail': int(min_mix_tail),
			'min_alt_tail': int(min_alt_tail)
		}
		if r_diag is not None:
			info.update({
				'residual_lcm_gap': float(r_diag['shape_gap']),
				'residual_lcm_tol': float(r_diag['shape_tol']),
				'cdf_range_violation': float(r_diag['cdf_range_violation']),
				'cdf_low_violation': float(r_diag['cdf_low_violation']),
				'cdf_high_violation': float(r_diag['cdf_high_violation']),
				'cdf_range_tol': float(r_diag['cdf_range_tol']),
				'cdf_mon_violation': float(r_diag['cdf_mon_violation']),
				'cdf_mon_tol': float(r_diag['cdf_mon_tol']),
				'G_anchor_x0': float(r_diag['g_x0']),
				'G_convex_gap': float(r_diag['g_convex_gap']),
				'G_convex_tol': float(r_diag['g_convex_tol']),
				'G_anchor_valid': bool(r_diag['g_valid']),
				'G_distance_ks': float(r_diag['g_distance_ks']),
				'G_distance_ise': float(r_diag['g_distance_ise']),
				'mix_tail_count': int(r_diag['mix_tail_count']),
				'alt_tail_count': int(r_diag['alt_tail_count'])
			})
		elif pos is not None:
			mix_tail, alt_tail = _tail_counts_at_pos(pos)
			info.update({
				'mix_tail_count': int(mix_tail),
				'alt_tail_count': int(alt_tail)
			})
		return info

	def _is_feasible_alpha(alpha, x_r_pos):
		nonlocal alpha_checks, last_failure_diag
		alpha_checks += 1
		_vprint(f"[alpha-check #{alpha_checks}] alpha={float(alpha):.6f}, current_x_R_pos={int(x_r_pos)}")
		if alpha >= 1.0 - 1e-12:
			last_failure_diag = {'alpha': float(alpha), 'reason': 'alpha_too_close_to_one'}
			_vprint(f"[alpha-check #{alpha_checks}] alpha too close to 1: NO")
			return False, x_r_pos, last_failure_diag

		start_pos = max(int(x_g_pos), int(x_r_pos))
		if start_pos > x_count_pos:
			last_failure_diag = {'alpha': float(alpha), 'reason': 'start_cutoff_exceeds_count_cutoff'}
			_vprint(
				f"[alpha-check #{alpha_checks}] start cutoff exceeds x_count: "
				f"start_pos={start_pos}, x_count_pos={x_count_pos}: NO"
			)
			return False, x_r_pos, last_failure_diag

		_vprint(
			f"[alpha-check #{alpha_checks}] test start cutoff pos={start_pos}, "
			f"x_start={float(grid_x[candidate_indices[start_pos]]):.6g}"
		)
		start_diag = _valid_r(alpha, start_pos)
		if start_diag['valid']:
			_vprint(
				f"[alpha-check #{alpha_checks}] start cutoff passes; "
				f"new_x_R={float(grid_x[candidate_indices[start_pos]]):.6g}: OK"
			)
			return True, start_pos, start_diag

		if start_pos == x_count_pos:
			last_failure_diag = start_diag
			_vprint(f"[alpha-check #{alpha_checks}] start is x_count and fails: NO")
			return False, x_r_pos, start_diag

		_vprint(
			f"[alpha-check #{alpha_checks}] test rightmost count cutoff pos={x_count_pos}, "
			f"x_count={float(grid_x[candidate_indices[x_count_pos]]):.6g}"
		)
		right_diag = _valid_r(alpha, x_count_pos)
		if not right_diag['valid']:
			last_failure_diag = right_diag
			_vprint(f"[alpha-check #{alpha_checks}] x_count fails; alpha infeasible")
			return False, x_r_pos, right_diag

		left = start_pos
		right = x_count_pos
		best_diag = right_diag
		def _cutoff_width(lo_pos, hi_pos):
			return float(grid_x[candidate_indices[hi_pos]] - grid_x[candidate_indices[lo_pos]])
		_vprint(
			f"[cutoff-bisect] alpha={float(alpha):.6f}, lo_pos={left}, hi_pos={right}, "
			f"width={_cutoff_width(left, right):.3g}, x0_tol={float(x0_tol):.3g}"
		)
		while right - left > 1 and _cutoff_width(left, right) > x0_tol:
			mid = (left + right) // 2
			mid_diag = _valid_r(alpha, mid)
			if mid_diag['valid']:
				_vprint(f"[cutoff-bisect] pos={mid} passes; move hi")
				right = mid
				best_diag = mid_diag
			else:
				_vprint(f"[cutoff-bisect] pos={mid} fails; move lo")
				left = mid
		stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		if best_diag['pos'] != right:
			best_diag = _valid_r(alpha, right)
		_vprint(
			f"[alpha-check #{alpha_checks}] cutoff bisection found "
			f"x_R={float(grid_x[candidate_indices[right]]):.6g}, "
			f"width={_cutoff_width(left, right):.3g}, stop={stop_reason}: OK"
		)
		return True, right, best_diag

	x_r_pos = 0
	_vprint("[alpha=0] initial feasibility check")
	ok0, new_x_r_pos, diag0 = _is_feasible_alpha(0.0, x_r_pos)
	if ok0:
		info = _make_info(True, 'alpha_zero_feasible', alpha=0.0, pos=new_x_r_pos, r_diag=diag0, alpha_lo=0.0, alpha_hi=0.0)
		_vprint(f"[done] alpha_hat=0, x0={info['x0']:.6g}, reason=alpha_zero_feasible")
		return 0.0, info['x0'], diag0['H_hat'], diag0['G_hat'], grid_x, info

	last_infeasible = 0.0
	last_diag = diag0
	found_bracket = False
	alpha_lo = 0.0
	alpha_hi = np.nan
	best_pos = None
	best_diag = None
	probes = np.linspace(alpha_left, alpha_right, num=int(n_alpha_probes))
	_vprint(
		f"[bracket] probing {int(n_alpha_probes)} alphas in "
		f"[{alpha_left:.6f}, {alpha_right:.6f}]"
	)
	for alpha_probe in probes:
		alpha_probe = float(alpha_probe)
		if alpha_probe <= 0.0 + 1e-15:
			continue
		if alpha_checks >= max_checks:
			_vprint(f"[bracket] max_checks reached before probe alpha={alpha_probe:.6f}")
			break
		_vprint(f"[bracket] probe alpha={alpha_probe:.6f}")
		ok, new_x_r_pos, diag = _is_feasible_alpha(alpha_probe, x_r_pos)
		if ok:
			alpha_lo = last_infeasible
			alpha_hi = alpha_probe
			x_r_pos = new_x_r_pos
			best_pos = new_x_r_pos
			best_diag = diag
			found_bracket = True
			_vprint(
				f"[bracket] found interval: lo={alpha_lo:.6f}, hi={alpha_hi:.6f}, "
				f"x_R={float(grid_x[candidate_indices[x_r_pos]]):.6g}"
			)
			break
		last_infeasible = alpha_probe
		last_diag = diag
		_vprint(f"[bracket] alpha={alpha_probe:.6f} infeasible")

	if not found_bracket:
		reason = 'no_feasible_alpha_in_search_interval'
		if alpha_checks >= max_checks:
			reason = 'max_checks_exhausted_before_bracket'
		_vprint(f"[failure] {reason}")
		info = _make_info(False, reason, alpha=np.nan, pos=None, r_diag=last_diag if isinstance(last_diag, dict) and 'shape_gap' in last_diag else None)
		return np.nan, None, None, None, grid_x, info

	iter_idx = 0
	while (alpha_hi - alpha_lo) > alpha_tol and alpha_checks < max_checks:
		iter_idx += 1
		mid = 0.5 * (alpha_lo + alpha_hi)
		_vprint(f"[bisect #{iter_idx:02d}] lo={alpha_lo:.6f}, hi={alpha_hi:.6f}, mid={mid:.6f}")
		ok, new_x_r_pos, diag = _is_feasible_alpha(mid, x_r_pos)
		if ok:
			alpha_hi = mid
			x_r_pos = new_x_r_pos
			best_pos = new_x_r_pos
			best_diag = diag
			_vprint(
				f"[bisect #{iter_idx:02d}] mid OK; new hi={alpha_hi:.6f}, "
				f"x_R={float(grid_x[candidate_indices[x_r_pos]]):.6g}"
			)
		else:
			alpha_lo = mid
			last_diag = diag
			_vprint(f"[bisect #{iter_idx:02d}] mid NO; new lo={alpha_lo:.6f}")

	reason = 'alpha_tol_reached' if (alpha_hi - alpha_lo) <= alpha_tol else 'max_checks_reached_with_feasible_bracket'
	info = _make_info(True, reason, alpha=alpha_hi, pos=best_pos, r_diag=best_diag, alpha_lo=alpha_lo, alpha_hi=alpha_hi)
	_vprint(
		f"[done] alpha_hat={float(alpha_hi):.6f}, x0={info['x0']:.6g}, "
		f"reason={reason}, alpha_checks={alpha_checks}, cutoff_checks={cutoff_checks}"
	)
	return float(alpha_hi), info['x0'], best_diag['H_hat'], best_diag['G_hat'], grid_x, info


def alpha_minimize_fixed_derived_x0(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	x_count_weight=0.25,
	require_flat_tail_reject=False,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5,
	_cutoff_strategy="weighted",
	_fixed_shape_requires_flat=False
):
	"""
	Derive a single working cutoff during bracketing, then run a fixed-cutoff alpha search.

	The cutoff derivation is:
	- find x_G using the matched-CDF convex compatibility check;
	- optionally restrict candidate cutoffs by the minimum tail-count screen;
	- find the first feasible bracketing alpha using the adaptive residual cutoff search;
	- let x_R_br be the residual cutoff returned at that bracketing alpha;
	- set x_target = (1 - x_count_weight) * x_R_br + x_count_weight * x_count;
	- map x_target to an allowed candidate cutoff, falling back toward x_R_br
	  if the target cutoff does not pass at the bracketing alpha;
	- if require_flat_tail_reject is True, require the bracketing cutoff and the
	  derived fixed cutoff to reject the fitted flat-tail null;
	- hold the resulting x0_star fixed and run the final alpha search with
	  separated shape and global-CDF boundaries.

	Returns
	-------
	alpha_hat : float
	x0_hat    : float or None
	H_hat     : np.ndarray or None
	G_hat     : np.ndarray or None
		Anchored fitted matched CDF on the full grid, computed at x_G.
	grid_x    : np.ndarray
	info      : dict
		Diagnostics for the bracketing-derived fixed-cutoff search.
	"""
	rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))
	if len(x_mix) == 0 or len(x_alt) == 0:
		raise ValueError("x_mix and x_alt must both be nonempty.")
	if alpha_tol <= 0:
		raise ValueError("alpha_tol must be positive.")
	x0_tol = 0.0 if x0_tol is None else float(x0_tol)
	if not np.isfinite(x0_tol) or x0_tol < 0.0:
		raise ValueError("x0_tol must be finite and nonnegative.")
	x_count_weight = float(x_count_weight)
	if not np.isfinite(x_count_weight) or not (0.0 <= x_count_weight <= 1.0):
		raise ValueError("x_count_weight must be finite and in [0, 1].")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")
	_cutoff_strategy = str(_cutoff_strategy)
	if _cutoff_strategy not in {"weighted", "rightmost"}:
		raise ValueError("_cutoff_strategy must be 'weighted' or 'rightmost'.")
	_fixed_shape_requires_flat = bool(_fixed_shape_requires_flat)
	if max_checks < 1:
		raise ValueError("max_checks must be at least 1.")
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	B = int(B)
	if B < 1:
		raise ValueError("B must be at least 1.")
	cdf_delta = delta if cdf_delta is None else float(cdf_delta)
	flat_delta = delta if flat_delta is None else float(flat_delta)
	require_flat_tail_reject = bool(require_flat_tail_reject)
	use_tail_count_cutoff = bool(use_tail_count_cutoff)
	x_count_weight_effective = x_count_weight
	if _cutoff_strategy == "rightmost":
		x_count_weight_effective = 0.0
	if not use_tail_count_cutoff and x_count_weight != 0.0:
		warnings.warn(
			"alpha_minimize_fixed_derived_x0: x_count_weight is ignored when "
			"use_tail_count_cutoff=False because x_count is not count-derived.",
			RuntimeWarning
		)
		x_count_weight_effective = 0.0
	min_mix_tail = int(min_mix_tail)
	min_alt_tail = int(min_alt_tail)
	if use_tail_count_cutoff:
		if min_mix_tail < 1 or min_alt_tail < 1:
			raise ValueError("min_mix_tail and min_alt_tail must be positive when use_tail_count_cutoff=True.")
	else:
		if min_mix_tail < 0 or min_alt_tail < 0:
			raise ValueError("min_mix_tail and min_alt_tail must be nonnegative.")

	alpha_left, alpha_right = alpha_bounds
	alpha_left = max(0.0, float(alpha_left))
	alpha_right = min(0.999999, float(alpha_right))
	if not (0.0 <= alpha_left < alpha_right < 1.0):
		raise ValueError("alpha_bounds must satisfy 0 <= lo < hi < 1.")

	def _vprint(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	def _ok_text(ok):
		return "OK" if ok else "NO"

	def _report_invalid_r(alpha, x0, R, cdf_obs, range_tol, mon_tol):
		if not verbose or R is None:
			return
		if cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol:
			return
		diffs = np.diff(R)
		min_diff = float(np.min(diffs)) if len(diffs) else float('nan')
		msg = (
			f"[invalid ValidR] alpha={alpha:.6f}, x0={x0:.6g}, "
			f"minDelta={min_diff:.3g}, minR={float(np.min(R)):.3g}, "
			f"maxR={float(np.max(R)):.3g}"
		)
		if cdf_obs['low'] > range_tol:
			i = int(np.argmin(R))
			msg += f", R<0 at x~{grid_x[i]:.6g}"
		if cdf_obs['high'] > range_tol:
			i = int(np.argmax(R))
			msg += f", R>1 at x~{grid_x[i]:.6g}"
		if cdf_obs['mon'] > mon_tol and len(diffs):
			j = int(np.argmin(diffs))
			msg += f", nonmonotone between x~{grid_x[j]:.6g} and {grid_x[j+1]:.6g}"
		_vprint(msg)

	x1 = 0.0
	if grid is None:
		x_left = float(np.min([x_mix.min(), x_alt.min()]))
		grid_x = _build_grid(x_mix, x_alt, x_left, x1, max_points=max_grid_points)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		grid_x = grid_x[np.isfinite(grid_x)]
		if len(grid_x) == 0:
			raise ValueError("grid must contain at least one finite value.")
		if grid_x[-1] < x1:
			grid_x = np.concatenate([grid_x, [x1]])

	if candidate_cutoffs is not None:
		cutoff_values = np.sort(np.asarray(candidate_cutoffs, dtype=float))
		cutoff_values = cutoff_values[np.isfinite(cutoff_values)]
		cutoff_values = cutoff_values[cutoff_values < x1]
		grid_x = np.unique(np.concatenate([grid_x, cutoff_values, np.array([x1], dtype=float)]))
	else:
		grid_x = np.unique(np.concatenate([grid_x, np.array([x1], dtype=float)]))

	grid_x = grid_x[grid_x <= x1]
	if len(grid_x) == 0:
		raise ValueError("grid has no points at or below 0.")
	if grid_x[-1] < x1:
		grid_x = np.concatenate([grid_x, [x1]])

	n, m = len(x_mix), len(x_alt)
	q = _bootstrap_quantile_level(delta, B, "alpha_minimize_fixed_derived_x0")
	q_cdf = _bootstrap_quantile_level(cdf_delta, B, "alpha_minimize_fixed_derived_x0.cdf")
	q_flat = (
		_bootstrap_quantile_level(flat_delta, B, "alpha_minimize_fixed_derived_x0.flat_tail")
		if require_flat_tail_reject else q
	)

	_vprint(
		f"[setup-fixed-derived] n={len(x_mix)}, m={len(x_alt)}, grid={len(grid_x)}, "
		f"boot_seed={_random_state_label(random_state)}, "
		f"alpha_bounds=({alpha_left:.6f}, {alpha_right:.6f}), "
		f"alpha_tol={float(alpha_tol):.3g}, x0_tol={float(x0_tol):.3g}, "
		f"gap_tol={float(gap_tol):.3g}, x_count_weight={float(x_count_weight):.3g}, "
		f"x_count_weight_eff={float(x_count_weight_effective):.3g}, B={int(B)}, "
		f"delta={float(delta):.3g}, q={q:.6f}, "
		f"cdf_delta={float(cdf_delta):.3g}, q_cdf={q_cdf:.6f}, "
		f"flat_reject={require_flat_tail_reject}, flat_delta={float(flat_delta):.3g}, "
		f"q_flat={q_flat:.6f}, use_tail_count_cutoff={use_tail_count_cutoff}, "
		f"cutoff_strategy={_cutoff_strategy}, fixed_shape_requires_flat={_fixed_shape_requires_flat}"
	)

	if candidate_cutoffs is None:
		candidate_values = grid_x[grid_x < x1]
	else:
		candidate_values = cutoff_values
	candidate_values = np.unique(candidate_values[candidate_values < x1])
	if len(candidate_values) == 0:
		Fn_empty = _ecdf_on_grid(x_mix, grid_x)
		Gm_empty = _ecdf_on_grid(x_alt, grid_x)
		_vprint("[setup-fixed-derived] no candidate cutoffs below 0")
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_candidate_cutoffs',
			'mode': 'fixed_derived_x0_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'alpha_checks': 0,
			'cutoff_checks': 0,
			'valid_g_checks': 0,
			'valid_r_checks': 0,
			'flat_tail_checks': 0,
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'require_flat_tail_reject': bool(require_flat_tail_reject),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'use_tail_count_cutoff': bool(use_tail_count_cutoff),
			'n_grid': int(len(grid_x)),
			'n_mix': int(len(x_mix)),
			'n_alt': int(len(x_alt)),
			'Fn_at_grid_end': float(Fn_empty[-1]),
			'Gm_at_grid_end': float(Gm_empty[-1])
		}

	candidate_indices = np.array([int(np.searchsorted(grid_x, x, side='left')) for x in candidate_values], dtype=int)
	candidate_indices = np.unique(candidate_indices)
	candidate_indices = candidate_indices[grid_x[candidate_indices] < x1]
	candidate_x = grid_x[candidate_indices]

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	_vprint(f"[setup-fixed-derived] bootstrap_quantile={q:.6f}")

	mix_tail_counts = np.array([np.sum(x_mix >= x0) for x0 in candidate_x], dtype=int)
	alt_tail_counts = np.array([np.sum(x_alt >= x0) for x0 in candidate_x], dtype=int)
	if use_tail_count_cutoff:
		count_ok = (mix_tail_counts >= min_mix_tail) & (alt_tail_counts >= min_alt_tail)
		if not np.any(count_ok):
			_vprint(
				f"[count] no count-admissible cutoff: min_mix_tail={min_mix_tail}, "
				f"min_alt_tail={min_alt_tail}, max_mix_tail={int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0}, "
				f"max_alt_tail={int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0}"
			)
			return np.nan, None, None, None, grid_x, {
				'success': False,
				'reason': 'no_count_admissible_cutoff',
				'mode': 'fixed_derived_x0_bootstrap',
				'alpha': np.nan,
				'x0': None,
				'alpha_checks': 0,
				'cutoff_checks': 0,
				'valid_g_checks': 0,
				'valid_r_checks': 0,
				'flat_tail_checks': 0,
				'B': int(B),
				'delta': float(delta),
				'bootstrap_quantile': float(q),
				'cdf_delta': float(cdf_delta),
				'cdf_bootstrap_quantile': float(q_cdf),
				'require_flat_tail_reject': bool(require_flat_tail_reject),
				'flat_delta': float(flat_delta),
				'flat_bootstrap_quantile': float(q_flat),
				'use_tail_count_cutoff': bool(use_tail_count_cutoff),
				'n_grid': int(len(grid_x)),
				'n_mix': int(n),
				'n_alt': int(m),
				'min_mix_tail': int(min_mix_tail),
				'min_alt_tail': int(min_alt_tail),
				'max_mix_tail': int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0,
				'max_alt_tail': int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0
			}
	else:
		count_ok = np.ones(len(candidate_indices), dtype=bool)

	candidate_indices = candidate_indices[count_ok]
	candidate_x = grid_x[candidate_indices]
	x_count_pos = len(candidate_indices) - 1
	if use_tail_count_cutoff:
		_vprint(
			f"[count] admissible_cutoffs={len(candidate_indices)}, "
			f"x_min={float(candidate_x[0]):.6g}, x_count={float(candidate_x[-1]):.6g}, "
			f"tail_at_x_count=({int(np.sum(x_mix >= candidate_x[-1]))}, {int(np.sum(x_alt >= candidate_x[-1]))})"
		)
	else:
		_vprint(
			f"[count] tail-count cutoff disabled; candidate_cutoffs={len(candidate_indices)}, "
			f"x_min={float(candidate_x[0]):.6g}, x_right={float(candidate_x[-1]):.6g}, "
			f"tail_at_right=({int(np.sum(x_mix >= candidate_x[-1]))}, {int(np.sum(x_alt >= candidate_x[-1]))})"
		)

	alpha_checks = 0
	cutoff_checks = 0
	valid_g_checks = 0
	shape_r_checks = 0
	valid_r_checks = 0
	flat_tail_checks = 0
	g_cache = {}
	shape_cache = {}
	r_cache = {}
	flat_cache = {}
	shape_refs = []
	cutoff_update_history = []
	shape_x0_pos = None
	flat_bisect_mode = "three_way"

	def _total_alpha_checks():
		return int(alpha_checks)

	def _tail_counts_at_pos(pos):
		x0 = float(grid_x[candidate_indices[pos]])
		return int(np.sum(x_mix >= x0)), int(np.sum(x_alt >= x0))

	def _valid_g(pos):
		nonlocal cutoff_checks, valid_g_checks
		pos = int(pos)
		if pos in g_cache:
			return g_cache[pos]

		cutoff_checks += 1
		valid_g_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		G_hat = _as_valid_cdf(_convex_projection_on_interval(grid_x, Gm, x0, x1))
		T_obs = _convex_gap_T(grid_x, Gm, x0, x1)
		Tstar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			Tstar[b] = _convex_gap_T(grid_x, Gm_b, x0, x1)
		tol = float(np.quantile(Tstar, q, method='higher'))
		diff = Gm - G_hat
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		diag = {
			'valid': bool(T_obs <= tol),
			'x0': x0,
			'pos': pos,
			'G_hat': G_hat,
			'convex_gap': float(T_obs),
			'convex_tol': tol,
			'bootstrap_quantile': float(q),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail,
			'distance_ks': float(np.max(np.abs(diff))),
			'distance_ise': _ise_on_grid(grid_x, diff)
		}
		g_cache[pos] = diag
		_vprint(
			f"[ValidG] pos={pos}, x0={x0:.6g}, tails=({mix_tail},{alt_tail}), "
			f"T={T_obs:.3g}, tol={tol:.3g}: {_ok_text(diag['valid'])}"
		)
		return diag

	def _first_valid_g_pos():
		_vprint("[ValidG-search] locating leftmost convex-compatible matched cutoff")
		right = x_count_pos
		if _valid_g(0)['valid']:
			_vprint(f"[ValidG-search] leftmost cutoff passes: pos=0, x_G={float(grid_x[candidate_indices[0]]):.6g}")
			return 0
		if not _valid_g(right)['valid']:
			_vprint(
				f"[ValidG-search] rightmost count-admissible cutoff fails: "
				f"pos={right}, x_count={float(grid_x[candidate_indices[right]]):.6g}"
			)
			return None
		left = 0
		while right - left > 1:
			mid = (left + right) // 2
			if _valid_g(mid)['valid']:
				_vprint(f"[ValidG-search] mid pos={mid} passes; move right endpoint")
				right = mid
			else:
				_vprint(f"[ValidG-search] mid pos={mid} fails; move left endpoint")
				left = mid
		_vprint(f"[ValidG-search] x_G pos={right}, x_G={float(grid_x[candidate_indices[right]]):.6g}")
		return right

	x_g_pos = _first_valid_g_pos()
	if x_g_pos is None:
		_vprint("[failure] no_convex_compatible_matched_cutoff")
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_convex_compatible_matched_cutoff',
			'mode': 'fixed_derived_x0_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'x_G': None,
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_checks': int(alpha_checks),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'valid_r_checks': int(valid_r_checks),
					'B': int(B),
					'delta': float(delta),
					'bootstrap_quantile': float(q),
					'cdf_delta': float(cdf_delta),
					'cdf_bootstrap_quantile': float(q_cdf),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices))
		}
	g_anchor_diag = _valid_g(x_g_pos)
	_vprint(
		f"[ValidG-anchor] using x_G={g_anchor_diag['x0']:.6g} "
		"for the matched bootstrap component"
	)

	def _valid_r(alpha, pos):
		nonlocal cutoff_checks, valid_r_checks
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in r_cache:
			diag = r_cache[key]
			_vprint(
				f"[ValidR cached] alpha={alpha:.6f}, pos={pos}, x0={float(diag['x0']):.6g}, "
				f"range={float(diag['cdf_range_violation']):.3g}/{float(diag['cdf_range_tol']):.3g}, "
				f"mon={float(diag['cdf_mon_violation']):.3g}/{float(diag['cdf_mon_tol']):.3g}, "
				f"T={float(diag['shape_gap']):.3g}/{float(diag['shape_tol']):.3g}, "
				f"ShapeR={_ok_text(diag['shape_ok'])}, CDFR={_ok_text(diag['cdf_ok'])}, "
				f"ValidR={_ok_text(diag['valid'])}"
			)
			return diag
		cutoff_checks += 1
		valid_r_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		T_obs = _gap_T(grid_x, R_obs, x0, x1)
		G_hat = g_anchor_diag['G_hat']
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat)

		range_star = np.empty(int(B), dtype=float)
		mon_star = np.empty(int(B), dtype=float)
		Tstar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
			Tstar[b] = _gap_T(grid_x, R_b, x0, x1)

		range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		shape_tol = float(np.quantile(Tstar, q, method='higher'))
		shape_ok = bool(T_obs <= shape_tol)
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		valid = bool(shape_ok and cdf_ok)
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		_report_invalid_r(alpha, x0, R_obs, cdf_obs, range_tol, mon_tol)
		_vprint(
			f"[ValidR] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"tails=({mix_tail},{alt_tail}), "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, "
			f"T={T_obs:.3g}/{shape_tol:.3g}, "
			f"ShapeR={_ok_text(shape_ok)}, CDFR={_ok_text(cdf_ok)}, ValidR={_ok_text(valid)}"
		)
		diag = {
			'valid': bool(valid),
			'shape_ok': bool(shape_ok),
			'cdf_ok': bool(cdf_ok),
			'full_ok': bool(valid),
			'alpha': alpha,
			'x0': x0,
			'pos': pos,
			'H_hat': H_hat,
			'G_hat': G_hat,
			'R_obs': R_obs,
			'shape_gap': float(T_obs),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'g_valid': bool(g_anchor_diag['valid']),
			'g_x0': float(g_anchor_diag['x0']),
			'g_convex_gap': float(g_anchor_diag['convex_gap']),
			'g_convex_tol': float(g_anchor_diag['convex_tol']),
			'g_distance_ks': float(g_anchor_diag['distance_ks']),
			'g_distance_ise': float(g_anchor_diag['distance_ise']),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail
		}
		r_cache[key] = diag
		return diag

	def _shape_r(alpha, pos, label="ShapeR"):
		nonlocal cutoff_checks, shape_r_checks
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in shape_cache:
			diag = shape_cache[key]
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, x0={float(diag['x0']):.6g}, "
				f"tails=({int(diag['mix_tail_count'])},{int(diag['alt_tail_count'])}), "
				f"T={float(diag['shape_gap']):.3g}/{float(diag['shape_tol']):.3g}, "
				f"ShapeR={_ok_text(diag['shape_ok'])}"
			)
			return diag
		cutoff_checks += 1
		shape_r_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		T_obs = _gap_T(grid_x, R_obs, x0, x1)
		G_hat = g_anchor_diag['G_hat']
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat)
		Tstar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			Tstar[b] = _gap_T(grid_x, R_b, x0, x1)
		shape_tol = float(np.quantile(Tstar, q, method='higher'))
		shape_ok = bool(T_obs <= shape_tol)
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		diag = {
			'alpha': alpha,
			'pos': pos,
			'x0': x0,
			'shape_ok': shape_ok,
			'shape_gap': float(T_obs),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_mon_violation': float(cdf_obs['mon']),
			'H_hat': H_hat,
			'G_hat': G_hat,
			'R_obs': R_obs,
			'g_valid': bool(g_anchor_diag['valid']),
			'g_x0': float(g_anchor_diag['x0']),
			'g_convex_gap': float(g_anchor_diag['convex_gap']),
			'g_convex_tol': float(g_anchor_diag['convex_tol']),
			'g_distance_ks': float(g_anchor_diag['distance_ks']),
			'g_distance_ise': float(g_anchor_diag['distance_ise']),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail
		}
		shape_cache[key] = diag
		_vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"tails=({mix_tail},{alt_tail}), T={T_obs:.3g}/{shape_tol:.3g}, "
			f"ShapeR={_ok_text(shape_ok)}"
		)
		return diag

	def _flat_reject(alpha, pos, shape_diag, label="FlatReject", allow_shape_fail=False):
		nonlocal cutoff_checks, flat_tail_checks
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in flat_cache:
			flat_diag = flat_cache[key]
			shape_diag.update(flat_diag)
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, x0={float(shape_diag['x0']):.6g}, "
				f"U={float(flat_diag['flat_stat']):.3g}/{float(flat_diag['flat_tol']):.3g}, "
				f"FlatReject={_ok_text(flat_diag['flat_reject'])}"
			)
			return flat_diag
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not allow_shape_fail:
			flat_diag = {
				'flat_available': False,
				'flat_reject': False,
				'flat_stat': np.nan,
				'flat_tol': np.nan,
				'flat_delta': float(flat_delta),
				'flat_bootstrap_quantile': float(q_flat),
				'flat_reason': 'shape_failed'
			}
			shape_diag.update(flat_diag)
			return flat_diag
		cutoff_checks += 1
		flat_tail_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		U_obs, H_flat = _flat_tail_stat(grid_x, shape_diag['H_hat'], x0, x1)
		H_flat = _as_valid_cdf(H_flat)
		G_hat = g_anchor_diag['G_hat']
		F_flat = _as_valid_cdf((1.0 - alpha)*H_flat + alpha*G_hat)
		Ustar = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_mix_b = _sample_from_cdf(grid_x, F_flat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			H_b = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_b, x0, x1))
			Ustar[b], _ = _flat_tail_stat(grid_x, H_b, x0, x1)
		flat_tol = float(np.quantile(Ustar, q_flat, method='higher'))
		flat_reject = bool(U_obs > flat_tol)
		flat_diag = {
			'flat_available': True,
			'flat_reject': flat_reject,
			'flat_stat': float(U_obs),
			'flat_tol': flat_tol,
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'flat_reason': 'computed',
			'flat_computed_with_shape_fail': bool(not shape_ok),
			'H_flat': H_flat
		}
		shape_diag.update(flat_diag)
		flat_cache[key] = flat_diag
		_vprint(
			f"[{label} #{flat_tail_checks}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"U={U_obs:.3g}/{flat_tol:.3g}, FlatReject={_ok_text(flat_reject)}"
			f"{', ShapeR=NO directional' if not shape_ok else ''}"
		)
		return flat_diag

	def _cutoff_selection_ok(alpha, pos, shape_diag, label="cutoff"):
		if not shape_diag.get('shape_ok', False):
			return False
		if not require_flat_tail_reject:
			return True
		flat_diag = _flat_reject(alpha, pos, shape_diag, label=label)
		return bool(flat_diag['flat_reject'])

	def _cutoff_width(lo_pos, hi_pos):
		return float(grid_x[candidate_indices[hi_pos]] - grid_x[candidate_indices[lo_pos]])

	def _cutoff_state(alpha, pos, shape_diag, label="cutoff"):
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not require_flat_tail_reject:
			return 'shape_fail'
		if not shape_ok and flat_bisect_mode != "guarded":
			return 'shape_fail'
		if not require_flat_tail_reject:
			return 'pass'
		flat_diag = _flat_reject(
			alpha, pos, shape_diag, label=label,
			allow_shape_fail=(flat_bisect_mode == "guarded")
		)
		flat_ok = bool(flat_diag['flat_reject'])
		if shape_ok and flat_ok:
			return 'pass'
		if not shape_ok and flat_ok:
			return 'shape_fail_flat_ok'
		if not shape_ok:
			return 'shape_fail_flat_fail'
		return 'flat_fail'

	def _derive_shape_cutoff(alpha, label="derive"):
		start_pos = int(x_g_pos)
		if start_pos > x_count_pos:
			return False, None, None, {
				'status': 'start_cutoff_exceeds_count_cutoff',
				'stop': None,
				'left_pos': None,
				'right_pos': None,
				'width': None
			}
		start_diag = _shape_r(alpha, start_pos, label=f"{label}-xG")
		start_state = _cutoff_state(alpha, start_pos, start_diag, label=f"{label}-flat-xG")
		if start_state == 'pass':
			return True, start_pos, start_diag, {
				'status': 'x_G_passes',
				'stop': 'leftmost',
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		if start_state == 'flat_fail' and flat_bisect_mode != "guarded":
			return False, None, start_diag, {
				'status': 'x_G_flat_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		if start_pos == x_count_pos:
			return False, None, start_diag, {
				'status': 'single_cutoff_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		right_diag = _shape_r(alpha, x_count_pos, label=f"{label}-xcount")
		right_state = _cutoff_state(alpha, x_count_pos, right_diag, label=f"{label}-flat-xcount")
		if right_state == 'shape_fail' or right_state == 'shape_fail_flat_ok':
			return False, None, right_diag, {
				'status': 'x_count_shape_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': x_count_pos,
				'width': _cutoff_width(start_pos, x_count_pos)
			}
		left = start_pos
		right = x_count_pos
		best_pos = x_count_pos if right_state == 'pass' else None
		best_diag = right_diag if right_state == 'pass' else None
		while right - left > 1 and _cutoff_width(left, right) > x0_tol:
			mid = (left + right) // 2
			mid_diag = _shape_r(alpha, mid, label=f"{label}-bisect")
			mid_state = _cutoff_state(alpha, mid, mid_diag, label=f"{label}-flat-bisect")
			if mid_state == 'shape_fail':
				if require_flat_tail_reject and flat_bisect_mode == "guarded":
					right = mid
					right_diag = mid_diag
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-fail; move hi")
				else:
					left = mid
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail; move lo")
			elif mid_state == 'shape_fail_flat_ok':
				left = mid
				_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-ok; move lo")
			elif mid_state in {'flat_fail', 'shape_fail_flat_fail'}:
				right = mid
				right_diag = mid_diag
				if mid_state == 'shape_fail_flat_fail':
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-fail; move hi")
				else:
					_vprint(f"[{label}-cutoff-bisect] pos={mid} flat-fail; move hi")
			else:
				right = mid
				right_diag = mid_diag
				best_pos = mid
				best_diag = mid_diag
				_vprint(f"[{label}-cutoff-bisect] pos={mid} passes; move hi")
		stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		if best_diag is None:
			return False, None, right_diag, {
				'status': 'no_shape_flat_overlap' if require_flat_tail_reject else 'no_shape_passing_cutoff',
				'stop': stop_reason,
				'left_pos': int(left),
				'right_pos': int(right),
				'width': _cutoff_width(left, right)
			}
		if best_diag['pos'] != best_pos:
			best_diag = _shape_r(alpha, best_pos, label=f"{label}-best")
			_cutoff_selection_ok(alpha, best_pos, best_diag, label=f"{label}-flat-best")
		return True, int(best_pos), best_diag, {
			'status': 'found',
			'stop': stop_reason,
			'left_pos': int(left),
			'right_pos': int(best_pos),
			'width': _cutoff_width(left, int(best_pos))
		}

	def _derive_rightmost_cutoff(alpha, label="derive-rightmost"):
		def _rightmost_state(pos, state_label):
			diag = _shape_r(alpha, pos, label=state_label)
			shape_ok = bool(diag.get('shape_ok', False))
			if not require_flat_tail_reject:
				return ('pass' if shape_ok else 'shape_fail'), diag
			flat_diag = _flat_reject(
				alpha, pos, diag,
				label=state_label.replace("-bisect", "-flat-bisect").replace("-xG", "-flat-xG").replace("-xcount", "-flat-xcount"),
				allow_shape_fail=True
			)
			flat_ok = bool(flat_diag.get('flat_reject', False))
			if shape_ok and flat_ok:
				return 'pass', diag
			if (not shape_ok) and flat_ok:
				return 'shape_fail_flat_ok', diag
			if not shape_ok:
				return 'shape_fail_flat_fail', diag
			return 'flat_fail', diag

		start_pos = int(x_g_pos)
		if start_pos > x_count_pos:
			return False, None, None, {
				'status': 'start_cutoff_exceeds_count_cutoff',
				'stop': None,
				'left_pos': None,
				'right_pos': None,
				'width': None
			}

		start_diag = _shape_r(alpha, start_pos, label=f"{label}-xG")
		if start_diag.get('shape_ok', False):
			start_state, start_diag = _rightmost_state(start_pos, f"{label}-xG-selection")
		else:
			start_state = 'shape_fail'
		right_diag = _shape_r(alpha, x_count_pos, label=f"{label}-xcount")
		right_shape_ok = bool(right_diag.get('shape_ok', False))

		if (not start_diag.get('shape_ok', False)) and (not right_shape_ok):
			return False, None, right_diag, {
				'status': 'endpoint_shape_fails',
				'start_state': start_state,
				'right_shape_ok': bool(right_shape_ok),
				'stop': None,
				'left_pos': int(start_pos),
				'right_pos': int(x_count_pos),
				'width': _cutoff_width(int(start_pos), int(x_count_pos))
			}

		if right_shape_ok:
			right_state, right_diag = _rightmost_state(x_count_pos, f"{label}-xcount-selection")
			if right_state == 'pass':
				return True, int(x_count_pos), right_diag, {
					'status': 'x_count_passes',
					'stop': 'rightmost',
					'left_pos': int(start_pos),
					'right_pos': int(x_count_pos),
					'width': _cutoff_width(int(start_pos), int(x_count_pos))
				}

		left = int(start_pos)
		right = int(x_count_pos)
		best_pos = int(start_pos) if start_state == 'pass' else None
		best_diag = start_diag if start_state == 'pass' else None
		stop_reason = None
		ambiguous_pos = None
		while right - left > 1 and _cutoff_width(left, right) > x0_tol:
			mid = (left + right) // 2
			mid_state, mid_diag = _rightmost_state(mid, f"{label}-bisect")
			if mid_state == 'pass':
				left = mid
				best_pos = mid
				best_diag = mid_diag
				_vprint(f"[{label}-cutoff-bisect] pos={mid} passes; move lo")
			elif mid_state in {'shape_fail', 'shape_fail_flat_ok'}:
				left = mid
				_vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; move lo")
			elif mid_state == 'shape_fail_flat_fail':
				right_diag = mid_diag
				ambiguous_pos = mid
				stop_reason = 'shape_flat_double_fail'
				_vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; stop ambiguous")
				break
			else:
				right = mid
				_vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; move hi")

		if stop_reason is None:
			stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		right_bound_pos = int(ambiguous_pos) if ambiguous_pos is not None else int(right)
		if best_diag is None:
			return False, None, right_diag, {
				'status': 'shape_flat_double_fail_no_cutoff' if ambiguous_pos is not None else 'no_rightmost_passing_cutoff',
				'stop': stop_reason,
				'left_pos': int(left),
				'right_pos': right_bound_pos,
				'ambiguous_pos': int(ambiguous_pos) if ambiguous_pos is not None else None,
				'width': _cutoff_width(int(left), right_bound_pos)
			}
		return True, int(best_pos), best_diag, {
			'status': 'rightmost_found_before_double_fail' if ambiguous_pos is not None else 'rightmost_found',
			'stop': stop_reason,
			'left_pos': int(start_pos),
			'right_pos': int(best_pos),
			'right_bound_pos': right_bound_pos,
			'ambiguous_pos': int(ambiguous_pos) if ambiguous_pos is not None else None,
			'width': _cutoff_width(int(best_pos), right_bound_pos) if right_bound_pos >= best_pos else 0.0
		}

	def _compact_shape(diag):
		if diag is None:
			return None
		out = {
			'alpha': float(diag['alpha']),
			'pos': int(diag['pos']),
			'x0': float(diag['x0']),
			'shape_ok': bool(diag['shape_ok']),
			'cutoff_selection_ok': bool(
				diag.get('shape_ok', False) and
				(not require_flat_tail_reject or bool(diag.get('flat_reject', False)))
			),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'mix_tail_count': int(diag['mix_tail_count']),
			'alt_tail_count': int(diag['alt_tail_count'])
		}
		if require_flat_tail_reject or 'flat_reject' in diag:
			out.update({
				'flat_available': bool(diag.get('flat_available', False)),
				'flat_reject': None if diag.get('flat_reject') is None else bool(diag.get('flat_reject')),
				'flat_stat': float(diag.get('flat_stat', np.nan)),
				'flat_tol': float(diag.get('flat_tol', np.nan)),
				'flat_delta': float(diag.get('flat_delta', flat_delta)),
				'flat_bootstrap_quantile': float(diag.get('flat_bootstrap_quantile', q_flat)),
				'flat_reason': diag.get('flat_reason'),
				'flat_computed_with_shape_fail': bool(diag.get('flat_computed_with_shape_fail', False))
			})
		return out

	def _record_shape_ref(alpha, pos, diag, source):
		rec = {
			'alpha': float(alpha),
			'pos': int(pos),
			'x0': float(grid_x[candidate_indices[int(pos)]]),
			'source': source,
			'shape_diag': diag
		}
		shape_refs.append(rec)
		return rec

	def _evaluate_shape_path(alpha, label="shape-alpha"):
		nonlocal alpha_checks, shape_x0_pos
		alpha = float(alpha)
		alpha_checks += 1
		old_pos = shape_x0_pos
		_vprint(
			f"[{label} #{alpha_checks}] alpha={alpha:.6f}, "
			f"current_x0={'None' if shape_x0_pos is None else f'{float(grid_x[candidate_indices[shape_x0_pos]]):.6g}'}"
		)
		if alpha >= 1.0 - 1e-12:
			return False, old_pos, None, {
				'updated': False,
				'old_pos': old_pos,
				'new_pos': old_pos,
				'search': {'status': 'alpha_too_close_to_one'}
			}
		if shape_x0_pos is not None:
			current_diag = _shape_r(alpha, shape_x0_pos, label=f"{label}-current")
			if _cutoff_selection_ok(alpha, shape_x0_pos, current_diag, label=f"{label}-flat-current"):
				_record_shape_ref(alpha, shape_x0_pos, current_diag, source='current')
				return True, shape_x0_pos, current_diag, {
					'updated': False,
					'old_pos': old_pos,
					'new_pos': shape_x0_pos,
					'search': {'status': 'current_cutoff_passes'}
				}
		if _cutoff_strategy == "rightmost":
			ok, new_pos, new_diag, search = _derive_rightmost_cutoff(alpha, label=f"{label}-derive-rightmost")
		else:
			ok, new_pos, new_diag, search = _derive_shape_cutoff(alpha, label=f"{label}-derive")
		if not ok:
			return False, old_pos, new_diag, {
				'updated': False,
				'old_pos': old_pos,
				'new_pos': old_pos,
				'search': search
			}
		shape_x0_pos = int(new_pos)
		_record_shape_ref(alpha, shape_x0_pos, new_diag, source='derived')
		update = {
			'alpha': float(alpha),
			'old_pos': None if old_pos is None else int(old_pos),
			'old_x0': None if old_pos is None else float(grid_x[candidate_indices[int(old_pos)]]),
			'new_pos': int(shape_x0_pos),
			'new_x0': float(grid_x[candidate_indices[int(shape_x0_pos)]]),
			'search': search,
			'shape_diag': _compact_shape(new_diag)
		}
		if old_pos != shape_x0_pos:
			cutoff_update_history.append(update)
		return True, shape_x0_pos, new_diag, {
			'updated': old_pos != shape_x0_pos,
			'old_pos': old_pos,
			'new_pos': shape_x0_pos,
			'search': search
		}

	def _shape_probe_record(diag, meta):
		out = _compact_shape(diag)
		if out is None:
			out = {}
		out.update({
			'search': meta.get('search') if meta else None,
			'updated': bool(meta.get('updated', False)) if meta else False
		})
		return out

	def _make_info(success, reason, alpha=None, pos=None, r_diag=None,
		alpha_lo=None, alpha_hi=None, extra=None):
		x0 = float(grid_x[candidate_indices[pos]]) if pos is not None else None
		info = {
			'success': bool(success),
			'reason': reason,
			'mode': 'fixed_derived_x0_bootstrap',
			'alpha': float(alpha) if alpha is not None else np.nan,
			'x0': x0,
			'x_G': float(grid_x[candidate_indices[x_g_pos]]),
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_lo': float(alpha_lo) if alpha_lo is not None else None,
			'alpha_hi': float(alpha_hi) if alpha_hi is not None else None,
			'alpha_checks': int(alpha_checks),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'shape_r_checks': int(shape_r_checks),
			'valid_r_checks': int(valid_r_checks),
			'flat_tail_checks': int(flat_tail_checks),
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'require_flat_tail_reject': bool(require_flat_tail_reject),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'use_tail_count_cutoff': bool(use_tail_count_cutoff),
			'alpha_tol': float(alpha_tol),
			'x0_tol': float(x0_tol),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
			'cutoff_strategy': _cutoff_strategy,
			'fixed_shape_requires_flat': bool(_fixed_shape_requires_flat),
			'x_count_weight': float(x_count_weight),
			'x_count_weight_effective': float(x_count_weight_effective),
			'max_checks': int(max_checks),
			'n_alpha_probes': int(n_alpha_probes),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices)),
			'n_mix': int(n),
			'n_alt': int(m),
			'min_mix_tail': int(min_mix_tail),
			'min_alt_tail': int(min_alt_tail),
			'cutoff_update_history': cutoff_update_history
		}
		if r_diag is not None:
			info.update({
				'shape_ok': bool(r_diag.get('shape_ok', False)),
				'fixed_shape_ok': bool(r_diag.get('fixed_shape_ok', r_diag.get('shape_ok', False))),
				'residual_lcm_gap': float(r_diag['shape_gap']),
				'residual_lcm_tol': float(r_diag['shape_tol']),
				'cdf_range_violation': float(r_diag['cdf_range_violation']),
				'cdf_low_violation': float(r_diag['cdf_low_violation']),
				'cdf_high_violation': float(r_diag['cdf_high_violation']),
				'cdf_mon_violation': float(r_diag['cdf_mon_violation']),
				'G_anchor_x0': float(r_diag['g_x0']),
				'G_convex_gap': float(r_diag['g_convex_gap']),
				'G_convex_tol': float(r_diag['g_convex_tol']),
				'G_anchor_valid': bool(r_diag['g_valid']),
				'G_distance_ks': float(r_diag['g_distance_ks']),
				'G_distance_ise': float(r_diag['g_distance_ise']),
				'mix_tail_count': int(r_diag['mix_tail_count']),
				'alt_tail_count': int(r_diag['alt_tail_count'])
			})
			if 'cdf_range_tol' in r_diag:
				info.update({
					'cdf_ok': bool(r_diag.get('cdf_ok', False)),
					'full_ok': bool(r_diag.get('valid', False)),
					'cdf_range_tol': float(r_diag['cdf_range_tol']),
					'cdf_mon_tol': float(r_diag['cdf_mon_tol'])
				})
			else:
				info.update({
					'cdf_ok': None,
					'full_ok': None,
					'cdf_range_tol': None,
					'cdf_mon_tol': None
				})
			if 'flat_reject' in r_diag:
				info.update({
					'flat_available': bool(r_diag.get('flat_available', False)),
					'flat_reject': None if r_diag.get('flat_reject') is None else bool(r_diag.get('flat_reject')),
					'flat_stat': float(r_diag.get('flat_stat', np.nan)),
					'flat_tol': float(r_diag.get('flat_tol', np.nan)),
					'flat_delta': float(r_diag.get('flat_delta', flat_delta)),
					'flat_bootstrap_quantile': float(r_diag.get('flat_bootstrap_quantile', q_flat))
				})
		elif pos is not None:
			mix_tail, alt_tail = _tail_counts_at_pos(pos)
			info.update({
				'mix_tail_count': int(mix_tail),
				'alt_tail_count': int(alt_tail)
			})
		if extra:
			info.update(extra)
		return info

	# Bracketing phase: derive x_R from the first passing probe only.
	shape_bracket = _run_shape_alpha_bracket(
		alpha_left=alpha_left,
		alpha_right=alpha_right,
		n_alpha_probes=n_alpha_probes,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=_total_alpha_checks,
		evaluate_shape_path=_evaluate_shape_path,
		shape_probe_record=_shape_probe_record,
		vprint=_vprint,
		refine=False
	)
	if not shape_bracket['success']:
		last_fail = shape_bracket.get('last_shape_fail')
		last_diag = last_fail.get('diag') if isinstance(last_fail, dict) else None
		reason = shape_bracket['reason']
		if alpha_checks >= max_checks:
			reason = 'max_checks_exhausted_before_shape_bracket'
		_vprint(f"[failure] {reason}")
		info = _make_info(
			False, reason, alpha=np.nan, pos=None,
			r_diag=last_diag if isinstance(last_diag, dict) and 'shape_gap' in last_diag else None,
			extra={'shape_probe_table': shape_bracket['shape_probe_table']}
		)
		return np.nan, None, None, None, grid_x, info

	alpha_br = float(shape_bracket['alpha_shape'])
	x_r_br_pos = int(shape_bracket['alpha_shape_pos'])
	br_diag = shape_bracket['alpha_shape_diag']
	_vprint(
		f"[derive] shape boundary alpha_br={alpha_br:.6f}, "
		f"x_br={float(grid_x[candidate_indices[x_r_br_pos]]):.6g}, "
		f"cutoff_strategy={_cutoff_strategy}"
	)

	# Derive and validate a fixed cutoff.
	x_r_br = float(grid_x[candidate_indices[x_r_br_pos]])
	x_count = float(grid_x[candidate_indices[x_count_pos]])
	moved_back = False
	moved_back_steps = 0
	target_pos = None
	target_valid_initial = None
	target_selection_initial = None
	x_target = None
	if _cutoff_strategy == "rightmost":
		x0_star_pos = x_r_br_pos
		x0_star_diag = br_diag
		_vprint(
			f"[derive-x0-rightmost] x_right_br={x_r_br:.6g}, x_count={x_count:.6g}, "
			f"target_pos={x0_star_pos}"
		)
	else:
		x_target = (1.0 - x_count_weight_effective)*x_r_br + x_count_weight_effective*x_count
		target_span = np.arange(x_r_br_pos, x_count_pos + 1, dtype=int)
		target_values = grid_x[candidate_indices[target_span]]
		target_pos = int(target_span[int(np.argmin(np.abs(target_values - x_target)))])
		_vprint(
			f"[derive-x0] x_R_br={x_r_br:.6g}, x_count={x_count:.6g}, "
			f"x_target={x_target:.6g}, target_pos={target_pos}"
		)

		target_diag = br_diag if target_pos == x_r_br_pos else _shape_r(alpha_br, target_pos, label="derive-x0-target")
		target_valid_initial = bool(target_diag['shape_ok'])
		target_selection_initial = bool(
			_cutoff_selection_ok(alpha_br, target_pos, target_diag, label="derive-flat-target")
		)
		x0_star_pos = target_pos
		x0_star_diag = target_diag
		if not target_selection_initial:
			_vprint("[derive-x0] target cutoff fails selection; moving back toward x_R_br")
			for pos in range(target_pos - 1, x_r_br_pos - 1, -1):
				cand_diag = br_diag if pos == x_r_br_pos else _shape_r(alpha_br, pos, label="derive-x0-fallback")
				moved_back_steps += 1
				if _cutoff_selection_ok(alpha_br, pos, cand_diag, label="derive-flat-fallback"):
					x0_star_pos = pos
					x0_star_diag = cand_diag
					moved_back = True
					break
	if not _cutoff_selection_ok(alpha_br, x0_star_pos, x0_star_diag, label="derive-flat-final"):
		# This should not occur because x_R_br was returned as a passing cutoff.
		_vprint("[failure] bracketing cutoff failed during fixed cutoff derivation")
		info = _make_info(
			False, 'derived_x0_validation_failed',
			alpha=np.nan, pos=x_r_br_pos, r_diag=br_diag,
			extra={
				'alpha_br': float(alpha_br),
				'x_R_br': float(x_r_br),
				'x_target': None if x_target is None else float(x_target),
				'x0_target_initial': None if target_pos is None else float(grid_x[candidate_indices[target_pos]]),
				'target_valid_initial': None if target_valid_initial is None else bool(target_valid_initial),
				'target_selection_initial': None if target_selection_initial is None else bool(target_selection_initial),
				'x0_moved_back': bool(moved_back),
				'x0_moved_back_steps': int(moved_back_steps)
			}
		)
		return np.nan, None, None, None, grid_x, info

	x0_star = float(grid_x[candidate_indices[x0_star_pos]])
	_vprint(
		f"[derive-x0] fixed x0_star={x0_star:.6g}, "
		f"target_initial={'None' if target_pos is None else f'{float(grid_x[candidate_indices[target_pos]]):.6g}'}, "
		f"moved_back={moved_back}"
	)

	derive_extra = {
		'alpha_br': float(alpha_br),
		'cutoff_strategy': _cutoff_strategy,
		'fixed_shape_requires_flat': bool(_fixed_shape_requires_flat),
		'derive_shape_boundary': shape_bracket['shape_boundary'],
		'derive_shape_probe_table': shape_bracket['shape_probe_table'],
		'alpha_br_diag': _compact_shape(br_diag),
		'n_shape_refs': int(len(shape_refs)),
		'x_R_br': float(x_r_br),
		'x_right_br': float(x_r_br) if _cutoff_strategy == "rightmost" else None,
		'x_target': None if x_target is None else float(x_target),
		'x_count_weight_effective': float(x_count_weight_effective),
		'x0_target_initial': None if target_pos is None else float(grid_x[candidate_indices[target_pos]]),
		'target_pos': None if target_pos is None else int(target_pos),
		'target_valid_initial': None if target_valid_initial is None else bool(target_valid_initial),
		'target_selection_initial': None if target_selection_initial is None else bool(target_selection_initial),
		'x0_moved_back': bool(moved_back),
		'x0_moved_back_steps': int(moved_back_steps),
		'x0_star_selection_valid': bool(_cutoff_selection_ok(alpha_br, x0_star_pos, x0_star_diag, label="derive-flat-summary"))
	}
	if require_flat_tail_reject:
		derive_extra.update({
			'x0_star_flat_reject': bool(x0_star_diag.get('flat_reject', False)),
			'x0_star_flat_stat': float(x0_star_diag.get('flat_stat', np.nan)),
			'x0_star_flat_tol': float(x0_star_diag.get('flat_tol', np.nan)),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat)
		})

	fixed_shape_cache = {}
	fixed_cdf_cache = {}
	fixed_full_cache = {}

	def _fixed_shape_ok(diag):
		return bool(diag.get('fixed_shape_ok', diag.get('shape_ok', False)))

	def _fixed_shape_msg(diag):
		if _fixed_shape_requires_flat and require_flat_tail_reject:
			return (
				f"ShapeR={_ok_text(diag['shape_ok'])}, "
				f"FlatReject={_ok_text(diag.get('flat_reject', False))}, "
				f"FixedShape={_ok_text(_fixed_shape_ok(diag))}"
			)
		return f"ShapeR={_ok_text(diag['shape_ok'])}"

	def _fixed_shape_at(alpha, label="fixed-shape"):
		nonlocal alpha_checks
		alpha = float(alpha)
		if alpha in fixed_shape_cache:
			diag = fixed_shape_cache[alpha]
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
				f"{_fixed_shape_msg(diag)}"
			)
			return diag
		alpha_checks += 1
		diag = dict(_shape_r(alpha, x0_star_pos, label=f"{label}-eval"))
		if _fixed_shape_requires_flat and require_flat_tail_reject:
			flat_diag = _flat_reject(alpha, x0_star_pos, diag, label=f"{label}-flat")
			diag['fixed_shape_ok'] = bool(diag['shape_ok'] and flat_diag['flat_reject'])
		else:
			diag['fixed_shape_ok'] = bool(diag['shape_ok'])
		fixed_shape_cache[alpha] = diag
		_vprint(
			f"[{label} #{alpha_checks}] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
			f"{_fixed_shape_msg(diag)}"
		)
		return diag

	def _fixed_cdf_at(alpha, label="fixed-cdf"):
		nonlocal alpha_checks, valid_r_checks
		alpha = float(alpha)
		if alpha in fixed_cdf_cache:
			diag = fixed_cdf_cache[alpha]
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
				f"CDFR={_ok_text(diag['cdf_ok'])}"
			)
			return diag
		alpha_checks += 1
		valid_r_checks += 1
		x0 = float(grid_x[candidate_indices[x0_star_pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		shape_diag = fixed_shape_cache.get(alpha)
		if shape_diag is None:
			shape_diag = shape_cache.get((alpha, x0_star_pos))
		H_hat = (
			shape_diag['H_hat'] if shape_diag is not None
			else _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		)
		G_hat = g_anchor_diag['G_hat']
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat)
		range_star = np.empty(int(B), dtype=float)
		mon_star = np.empty(int(B), dtype=float)
		for b in range(int(B)):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
		range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		_report_invalid_r(alpha, x0, R_obs, cdf_obs, range_tol, mon_tol)
		diag = {
			'alpha': alpha,
			'x0': x0,
			'pos': x0_star_pos,
			'cdf_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'H_hat': H_hat,
			'G_hat': G_hat,
			'R_obs': R_obs,
			'mix_tail_count': int(np.sum(x_mix >= x0)),
			'alt_tail_count': int(np.sum(x_alt >= x0))
		}
		fixed_cdf_cache[alpha] = diag
		_vprint(
			f"[{label} #{alpha_checks}] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, CDFR={_ok_text(cdf_ok)}"
		)
		return diag

	def _merge_fixed_diag(shape_diag, cdf_diag):
		diag = dict(shape_diag)
		diag.update({
			'cdf_ok': bool(cdf_diag['cdf_ok']),
			'full_ok': bool(_fixed_shape_ok(shape_diag) and cdf_diag['cdf_ok']),
			'valid': bool(_fixed_shape_ok(shape_diag) and cdf_diag['cdf_ok']),
			'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
			'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
			'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
			'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
			'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
			'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
			'cdf_delta': float(cdf_diag.get('cdf_delta', cdf_delta)),
			'cdf_bootstrap_quantile': float(cdf_diag.get('cdf_bootstrap_quantile', q_cdf))
		})
		return diag

	def _fixed_full_at(alpha, label="fixed-full"):
		alpha = float(alpha)
		if alpha in fixed_full_cache:
			diag = fixed_full_cache[alpha]
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
				f"{_fixed_shape_msg(diag)}, CDFR={_ok_text(diag['cdf_ok'])}, "
				f"Full={_ok_text(diag['valid'])}"
			)
			return diag
		shape_diag = _fixed_shape_at(alpha, label=f"{label}-shape")
		cdf_diag = _fixed_cdf_at(alpha, label=f"{label}-cdf")
		diag = _merge_fixed_diag(shape_diag, cdf_diag)
		fixed_full_cache[alpha] = diag
		_vprint(
			f"[{label}] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
			f"{_fixed_shape_msg(diag)}, CDFR={_ok_text(diag['cdf_ok'])}, "
			f"Full={_ok_text(diag['valid'])}"
		)
		return diag

	def _compact_fixed_shape(diag):
		if diag is None:
			return None
		out = {
			'alpha': float(diag['alpha']),
			'shape_ok': bool(diag['shape_ok']),
			'fixed_shape_ok': bool(_fixed_shape_ok(diag)),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation'])
		}
		if 'flat_reject' in diag:
			out.update({
				'flat_reject': bool(diag.get('flat_reject', False)),
				'flat_stat': float(diag.get('flat_stat', np.nan)),
				'flat_tol': float(diag.get('flat_tol', np.nan))
			})
		return out

	def _compact_fixed_cdf(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'cdf_ok': bool(diag['cdf_ok']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_delta': float(diag.get('cdf_delta', cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', q_cdf))
		}

	def _compact_fixed_diag(diag):
		if diag is None:
			return None
		out = _compact_fixed_shape(diag)
		out.update(_compact_fixed_cdf(diag))
		out['full_ok'] = bool(diag['valid'])
		return out

	def _refine_fixed_shape(fail_diag, pass_diag):
		lo = fail_diag
		hi = pass_diag
		status = 'alpha_tol_reached'
		iter_idx = 0
		while (hi['alpha'] - lo['alpha']) > alpha_tol and alpha_checks < max_checks:
			iter_idx += 1
			mid = 0.5 * (lo['alpha'] + hi['alpha'])
			_vprint(f"[fixed-shape-bisect #{iter_idx:02d}] lo={lo['alpha']:.6f}, hi={hi['alpha']:.6f}, mid={mid:.6f}")
			mid_diag = _fixed_shape_at(mid, label="fixed-shape-bisect")
			if _fixed_shape_ok(mid_diag):
				hi = mid_diag
			else:
				lo = mid_diag
		if (hi['alpha'] - lo['alpha']) > alpha_tol:
			status = 'max_checks_reached'
		return hi, {
			'status': status,
			'lo': float(lo['alpha']),
			'hi': float(hi['alpha']),
			'width': float(hi['alpha'] - lo['alpha'])
		}

	# Final fixed-cutoff alpha minimization with separated boundaries.
	probe_alphas = np.linspace(alpha_left, alpha_right, num=int(n_alpha_probes))
	shape_probe_table = []
	last_shape_fail = None
	first_shape_pass = None
	_vprint("[fixed-shape-scan] evaluating shape boundary probes at derived fixed cutoff")
	for alpha_probe in probe_alphas:
		if alpha_checks >= max_checks:
			break
		diag = _fixed_shape_at(alpha_probe, label="fixed-shape-probe")
		shape_probe_table.append(_compact_fixed_shape(diag))
		if _fixed_shape_ok(diag):
			first_shape_pass = {'alpha': float(alpha_probe), 'diag': diag}
			_vprint(f"[fixed-shape-scan] first shape pass alpha={float(alpha_probe):.6f}")
			break
		last_shape_fail = {'alpha': float(alpha_probe), 'diag': diag}

	if first_shape_pass is None:
		final_diag = last_shape_fail['diag'] if last_shape_fail is not None else x0_star_diag
		extra = dict(derive_extra)
		extra.update({
			'gap_status': 'no_shape_passing_probe',
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': [],
			'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, []),
			'alpha_shape_diag': None,
			'alpha_cdf_diag': None
		})
		info = _make_info(False, 'no_fixed_shape_passing_probe', alpha=np.nan, pos=x0_star_pos, r_diag=final_diag, extra=extra)
		return np.nan, None, None, None, grid_x, info

	if last_shape_fail is None:
		alpha_shape_diag = first_shape_pass['diag']
		shape_boundary = {
			'status': 'left_endpoint_shape_passing',
			'lo': float(alpha_left),
			'hi': float(alpha_shape_diag['alpha']),
			'width': 0.0
		}
	else:
		alpha_shape_diag, shape_boundary = _refine_fixed_shape(last_shape_fail['diag'], first_shape_pass['diag'])

	alpha_shape = float(alpha_shape_diag['alpha'])
	full_at_shape = _fixed_full_at(alpha_shape, label="direct-shape")
	if full_at_shape['valid']:
		cdf_probe_table = [_compact_fixed_cdf(full_at_shape)]
		probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
		shared_extra = dict(derive_extra)
		shared_extra.update({
			'alpha_shape': float(alpha_shape),
			'alpha_cdf': float(alpha_shape),
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {
				'status': 'direct_alpha_shape_pass',
				'lo': float(alpha_shape),
				'hi': float(alpha_shape),
				'width': 0.0
			},
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': _compact_fixed_shape(alpha_shape_diag),
			'alpha_cdf_diag': _compact_fixed_cdf(full_at_shape)
		})
		info = _make_info(
			True, 'overlap_shape_boundary_fixed_derived_x0',
			alpha=alpha_shape, pos=x0_star_pos, r_diag=full_at_shape,
			alpha_lo=shape_boundary.get('lo'), alpha_hi=alpha_shape,
			extra=shared_extra
		)
		_vprint(f"[direct-alpha-shape] alpha={alpha_shape:.6f}, ShapeR=OK, CDFR=OK; stopping")
		_vprint(f"[done-fixed-derived] alpha_hat={alpha_shape:.6f}, x0={info['x0']:.6g}, reason={info['reason']}")
		return alpha_shape, info['x0'], full_at_shape['H_hat'], full_at_shape['G_hat'], grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=alpha_right,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=lambda: alpha_checks,
		evaluate_cdf=lambda a, label: _fixed_cdf_at(a, label=label),
		compact_cdf=_compact_fixed_cdf,
		vprint=_vprint,
		stop_after_pass_at=alpha_shape
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
	if not cdf_boundary_result['success']:
		first_cdf_fail = cdf_boundary_result.get('first_cdf_fail')
		if first_cdf_fail is not None:
			fail_alpha = float(first_cdf_fail['alpha'])
			fail_shape_diag = _fixed_shape_at(fail_alpha, label="cdf-fail-shape")
			final_diag = _merge_fixed_diag(fail_shape_diag, first_cdf_fail['diag'])
		else:
			final_diag = alpha_shape_diag
		extra = dict(derive_extra)
		extra.update({
			'alpha_shape': alpha_shape,
			'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': _compact_fixed_shape(alpha_shape_diag),
			'alpha_cdf_diag': None
		})
		info = _make_info(False, 'no_fixed_cdf_passing_probe', alpha=np.nan, pos=x0_star_pos, r_diag=final_diag, extra=extra)
		return np.nan, None, None, None, grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	cdf_boundary = cdf_boundary_result['cdf_boundary']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = dict(derive_extra)
	shared_extra.update({
		'alpha_shape': float(alpha_shape),
		'alpha_cdf': float(alpha_cdf),
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary,
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'coarse_probe_table': probe_table,
		'alpha_shape_diag': _compact_fixed_shape(alpha_shape_diag),
		'alpha_cdf_diag': _compact_fixed_cdf(alpha_cdf_diag)
	})
	_vprint(
		f"[fixed-boundaries] alpha_shape={alpha_shape:.6f}, alpha_cdf={alpha_cdf:.6f}, "
		f"gap={gap:.3g}, status={shared_extra['gap_status']}"
	)

	def _scan_for_fixed_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or alpha_checks >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan = []
		for a in alphas:
			if alpha_checks >= max_checks:
				break
			diag = _fixed_full_at(a, label=label)
			scan.append(_compact_fixed_diag(diag))
			if diag['valid']:
				return diag, scan
		return None, scan

	full_at_shape = _fixed_full_at(alpha_shape, label="fixed-at-shape")
	if gap <= 0.0:
		if full_at_shape['valid']:
			info = _make_info(
				True, 'overlap_shape_boundary_fixed_derived_x0',
				alpha=alpha_shape, pos=x0_star_pos, r_diag=full_at_shape,
				alpha_lo=shape_boundary.get('lo'), alpha_hi=alpha_shape,
				extra=shared_extra
			)
			_vprint(f"[done-fixed-derived] alpha_hat={alpha_shape:.6f}, x0={info['x0']:.6g}, reason={info['reason']}")
			return alpha_shape, info['x0'], full_at_shape['H_hat'], full_at_shape['G_hat'], grid_x, info
		pass_diag, scan = _scan_for_fixed_full_pass(alpha_shape, alpha_cdf, "fixed-overlap-scan")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = scan
		if pass_diag is not None:
			info = _make_info(
				True, 'overlap_local_full_pass_fixed_derived_x0',
				alpha=pass_diag['alpha'], pos=x0_star_pos, r_diag=pass_diag,
				alpha_lo=shape_boundary.get('lo'), alpha_hi=pass_diag['alpha'],
				extra=extra
			)
			_vprint(f"[done-fixed-derived] alpha_hat={float(pass_diag['alpha']):.6f}, x0={info['x0']:.6g}, reason={info['reason']}")
			return float(pass_diag['alpha']), info['x0'], pass_diag['H_hat'], pass_diag['G_hat'], grid_x, info
		info = _make_info(
			False, 'overlap_without_full_passing_probe_fixed_derived_x0',
			alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape,
			extra=extra
		)
		return np.nan, None, None, None, grid_x, info

	if gap <= gap_tol:
		pass_diag, scan = _scan_for_fixed_full_pass(alpha_cdf, alpha_shape, "fixed-near-miss")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = scan
		if pass_diag is not None:
			info = _make_info(
				True, 'near_miss_full_passing_probe_fixed_derived_x0',
				alpha=pass_diag['alpha'], pos=x0_star_pos, r_diag=pass_diag,
				alpha_lo=alpha_cdf, alpha_hi=alpha_shape,
				extra=extra
			)
			_vprint(f"[done-fixed-derived] alpha_hat={float(pass_diag['alpha']):.6f}, x0={info['x0']:.6g}, reason={info['reason']}")
			return float(pass_diag['alpha']), info['x0'], pass_diag['H_hat'], pass_diag['G_hat'], grid_x, info
		info = _make_info(
			False, 'near_miss_no_overlap_fixed_derived_x0',
			alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape,
			extra=extra
		)
		return np.nan, None, None, None, grid_x, info

	info = _make_info(
		False, 'separated_boundaries_gap_fixed_derived_x0',
		alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape,
		extra=shared_extra
	)
	return np.nan, None, None, None, grid_x, info


def alpha_minimize_fixed_derived_x0_rightmost(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	require_flat_tail_reject=True,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5
):
	"""
	Rightmost-cutoff variant of alpha_minimize_fixed_derived_x0.

	The bracketing phase derives the rightmost cutoff at the first passing
	alpha probe, subject to x0 <= x_count and, when requested, FlatReject.  The
	final fixed-cutoff alpha scan/bisection is then run from scratch at that
	fixed cutoff.  If require_flat_tail_reject is True, FlatReject is part of
	that final fixed-shape predicate as well.
	"""
	return alpha_minimize_fixed_derived_x0(
		x_mix,
		x_alt,
		alpha_tol=alpha_tol,
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		B=B,
		delta=delta,
		cdf_delta=cdf_delta,
		flat_delta=flat_delta,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		max_checks=max_checks,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		x_count_weight=0.0,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		gap_tol=gap_tol,
		near_miss_probes=near_miss_probes,
		_cutoff_strategy="rightmost",
		_fixed_shape_requires_flat=True
	)


def alpha_minimize_derived_x0_revisited(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=64,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	gap_tol=None,
	near_miss_probes=5,
	cdf_ref_max_dist=None,
	require_flat_tail_reject=False,
	use_tail_count_cutoff=True,
	flat_bisect_mode="three_way"
):
	"""
	Semi-adjustable derived-cutoff search with separated shape and CDF boundaries.

	The residual cutoff is derived using suffix shape checks. If
	require_flat_tail_reject is True, each shape-passing candidate cutoff must
	also reject a fitted flat-tail null before it can be selected. The
	flat-tail tolerance uses flat_delta when supplied, otherwise delta, because
	this predicate is used through rejection rather than acceptance. The
	flat-tail cutoff bisection can use either the default three-way rule or
	flat_bisect_mode="guarded", where ShapeR=NO and FlatReject=OK is the only
	move-right outcome. During the shape-boundary search, a failed cutoff check
	at the current cutoff triggers a fresh cutoff derivation at that alpha, and
	the working cutoff may move either left or right. CDF validity is searched
	as a separate upper alpha boundary; CDF-only bootstrap calibration uses the
	nearest previously shape-derived cutoff as its reference interval.

	Returns
	-------
	alpha_hat : float
	x0_hat    : float or None
	H_hat     : np.ndarray or None
	G_hat     : np.ndarray or None
		Anchored fitted matched CDF on the full grid, computed at x_G.
	grid_x    : np.ndarray
	info      : dict
		Diagnostics for the semi-adjustable separated-boundary search.
	"""
	rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))
	if len(x_mix) == 0 or len(x_alt) == 0:
		raise ValueError("x_mix and x_alt must both be nonempty.")
	if alpha_tol <= 0:
		raise ValueError("alpha_tol must be positive.")
	B = int(B)
	if B < 1:
		raise ValueError("B must be at least 1.")
	n_alpha_probes = int(n_alpha_probes)
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	max_checks = int(max_checks)
	if max_checks < 2*n_alpha_probes:
		raise ValueError("max_checks must be at least 2*n_alpha_probes for separated shape/CDF probes.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")
	x0_tol = 0.0 if x0_tol is None else float(x0_tol)
	if not np.isfinite(x0_tol) or x0_tol < 0.0:
		raise ValueError("x0_tol must be finite and nonnegative.")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	cdf_delta = delta if cdf_delta is None else float(cdf_delta)
	flat_delta = delta if flat_delta is None else float(flat_delta)
	require_flat_tail_reject = bool(require_flat_tail_reject)
	use_tail_count_cutoff = bool(use_tail_count_cutoff)
	flat_bisect_mode = str(flat_bisect_mode)
	if flat_bisect_mode not in {"three_way", "guarded"}:
		raise ValueError("flat_bisect_mode must be 'three_way' or 'guarded'.")
	min_mix_tail = int(min_mix_tail)
	min_alt_tail = int(min_alt_tail)
	if use_tail_count_cutoff:
		if min_mix_tail < 1 or min_alt_tail < 1:
			raise ValueError("min_mix_tail and min_alt_tail must be positive when use_tail_count_cutoff=True.")
	else:
		if min_mix_tail < 0 or min_alt_tail < 0:
			raise ValueError("min_mix_tail and min_alt_tail must be nonnegative.")

	alpha_left, alpha_right = alpha_bounds
	alpha_left = max(0.0, float(alpha_left))
	alpha_right = min(0.999999, float(alpha_right))
	if not (0.0 <= alpha_left < alpha_right < 1.0):
		raise ValueError("alpha_bounds must satisfy 0 <= lo < hi < 1.")
	probe_spacing = (alpha_right - alpha_left) / max(1, n_alpha_probes - 1)
	if cdf_ref_max_dist is None:
		cdf_ref_max_dist = probe_spacing
	else:
		cdf_ref_max_dist = float(cdf_ref_max_dist)
	if not np.isfinite(cdf_ref_max_dist) or cdf_ref_max_dist < 0.0:
		raise ValueError("cdf_ref_max_dist must be finite and nonnegative.")
	q = _bootstrap_quantile_level(delta, B, "alpha_minimize_derived_x0_revisited")
	q_cdf = _bootstrap_quantile_level(cdf_delta, B, "alpha_minimize_derived_x0_revisited.cdf")
	q_flat = (
		_bootstrap_quantile_level(flat_delta, B, "alpha_minimize_derived_x0_revisited.flat_tail")
		if require_flat_tail_reject else q
	)

	def _vprint(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	def _ok_text(ok):
		return "OK" if ok else "NO"

	x1 = 0.0
	if grid is None:
		x_left = float(np.min([x_mix.min(), x_alt.min()]))
		grid_x = _build_grid(x_mix, x_alt, x_left, x1, max_points=max_grid_points)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		grid_x = grid_x[np.isfinite(grid_x)]
		if len(grid_x) == 0:
			raise ValueError("grid must contain at least one finite value.")

	if candidate_cutoffs is not None:
		cutoff_values = np.sort(np.asarray(candidate_cutoffs, dtype=float))
		cutoff_values = cutoff_values[np.isfinite(cutoff_values)]
		cutoff_values = cutoff_values[cutoff_values < x1]
		grid_x = np.unique(np.concatenate([grid_x, cutoff_values, np.array([x1], dtype=float)]))
	else:
		grid_x = np.unique(np.concatenate([grid_x, np.array([x1], dtype=float)]))

	grid_x = grid_x[grid_x <= x1]
	if len(grid_x) == 0:
		raise ValueError("grid has no points at or below 0.")
	if grid_x[-1] < x1:
		grid_x = np.concatenate([grid_x, [x1]])

	if candidate_cutoffs is None:
		candidate_values = grid_x[grid_x < x1]
	else:
		candidate_values = cutoff_values
	candidate_values = np.unique(candidate_values[candidate_values < x1])
	if len(candidate_values) == 0:
		Fn_empty = _ecdf_on_grid(x_mix, grid_x)
		Gm_empty = _ecdf_on_grid(x_alt, grid_x)
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_candidate_cutoffs',
			'mode': 'semiadjustable_x0_separated_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'alpha_checks': 0,
			'cdf_checks': 0,
			'cutoff_checks': 0,
			'valid_g_checks': 0,
			'shape_r_checks': 0,
			'flat_tail_checks': 0,
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'require_flat_tail_reject': bool(require_flat_tail_reject),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'use_tail_count_cutoff': bool(use_tail_count_cutoff),
			'flat_bisect_mode': flat_bisect_mode,
			'n_grid': int(len(grid_x)),
			'n_mix': int(len(x_mix)),
			'n_alt': int(len(x_alt)),
			'Fn_at_grid_end': float(Fn_empty[-1]),
			'Gm_at_grid_end': float(Gm_empty[-1])
		}

	candidate_indices = np.array([int(np.searchsorted(grid_x, x, side='left')) for x in candidate_values], dtype=int)
	candidate_indices = np.unique(candidate_indices)
	candidate_indices = candidate_indices[grid_x[candidate_indices] < x1]
	candidate_x = grid_x[candidate_indices]

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	n, m = len(x_mix), len(x_alt)

	mix_tail_counts = np.array([np.sum(x_mix >= x0) for x0 in candidate_x], dtype=int)
	alt_tail_counts = np.array([np.sum(x_alt >= x0) for x0 in candidate_x], dtype=int)
	if use_tail_count_cutoff:
		count_ok = (mix_tail_counts >= min_mix_tail) & (alt_tail_counts >= min_alt_tail)
		if not np.any(count_ok):
			return np.nan, None, None, None, grid_x, {
				'success': False,
				'reason': 'no_count_admissible_cutoff',
				'mode': 'semiadjustable_x0_separated_bootstrap',
				'alpha': np.nan,
				'x0': None,
				'alpha_checks': 0,
				'cdf_checks': 0,
				'cutoff_checks': 0,
				'valid_g_checks': 0,
				'shape_r_checks': 0,
				'flat_tail_checks': 0,
				'delta': float(delta),
				'bootstrap_quantile': float(q),
				'cdf_delta': float(cdf_delta),
				'cdf_bootstrap_quantile': float(q_cdf),
				'require_flat_tail_reject': bool(require_flat_tail_reject),
				'flat_delta': float(flat_delta),
				'flat_bootstrap_quantile': float(q_flat),
				'flat_bisect_mode': flat_bisect_mode,
				'n_grid': int(len(grid_x)),
				'n_mix': int(n),
				'n_alt': int(m),
				'use_tail_count_cutoff': bool(use_tail_count_cutoff),
				'min_mix_tail': int(min_mix_tail),
				'min_alt_tail': int(min_alt_tail),
				'max_mix_tail': int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0,
				'max_alt_tail': int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0
			}
	else:
		count_ok = np.ones(len(candidate_indices), dtype=bool)

	candidate_indices = candidate_indices[count_ok]
	candidate_x = grid_x[candidate_indices]
	x_count_pos = len(candidate_indices) - 1

	_vprint(
		f"[setup-semi-x0] n={n}, m={m}, grid={len(grid_x)}, "
		f"boot_seed={_random_state_label(random_state)}, "
		f"alpha_bounds=({alpha_left:.6f}, {alpha_right:.6f}), "
		f"alpha_tol={float(alpha_tol):.3g}, x0_tol={float(x0_tol):.3g}, "
		f"gap_tol={float(gap_tol):.3g}, flat_reject={require_flat_tail_reject}, "
		f"flat_bisect_mode={flat_bisect_mode}, use_tail_count_cutoff={use_tail_count_cutoff}, "
		f"B={B}, delta={float(delta):.3g}, q={q:.6f}, "
		f"cdf_delta={float(cdf_delta):.3g}, q_cdf={q_cdf:.6f}, "
		f"flat_delta={float(flat_delta):.3g}, q_flat={q_flat:.6f}"
	)
	if use_tail_count_cutoff:
		_vprint(
			f"[count] admissible_cutoffs={len(candidate_indices)}, "
			f"x_min={float(candidate_x[0]):.6g}, x_count={float(candidate_x[-1]):.6g}, "
			f"tail_at_x_count=({int(np.sum(x_mix >= candidate_x[-1]))}, {int(np.sum(x_alt >= candidate_x[-1]))})"
		)
	else:
		_vprint(
			f"[count] tail-count cutoff disabled; candidate_cutoffs={len(candidate_indices)}, "
			f"x_min={float(candidate_x[0]):.6g}, x_right={float(candidate_x[-1]):.6g}, "
			f"tail_at_right=({int(np.sum(x_mix >= candidate_x[-1]))}, {int(np.sum(x_alt >= candidate_x[-1]))})"
		)

	alpha_checks = 0
	cdf_checks = 0
	cutoff_checks = 0
	valid_g_checks = 0
	shape_r_checks = 0
	flat_tail_checks = 0
	g_cache = {}
	shape_cache = {}
	flat_cache = {}
	cdf_cache = {}
	shape_refs = []
	cutoff_update_history = []
	x0_star_pos = None

	def _total_alpha_checks():
		return int(alpha_checks + cdf_checks)

	def _tail_counts_at_pos(pos):
		x0 = float(grid_x[candidate_indices[pos]])
		return int(np.sum(x_mix >= x0)), int(np.sum(x_alt >= x0))

	def _cutoff_width(lo_pos, hi_pos):
		return float(grid_x[candidate_indices[hi_pos]] - grid_x[candidate_indices[lo_pos]])

	def _valid_g(pos):
		nonlocal cutoff_checks, valid_g_checks
		pos = int(pos)
		if pos in g_cache:
			return g_cache[pos]
		cutoff_checks += 1
		valid_g_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		G_hat = _as_valid_cdf(_convex_projection_on_interval(grid_x, Gm, x0, x1))
		T_obs = _convex_gap_T(grid_x, Gm, x0, x1)
		Tstar = np.empty(B, dtype=float)
		for b in range(B):
			x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
			x_alt_b.sort(kind='mergesort')
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			Tstar[b] = _convex_gap_T(grid_x, Gm_b, x0, x1)
		tol = float(np.quantile(Tstar, q, method='higher'))
		diff = Gm - G_hat
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		diag = {
			'valid': bool(T_obs <= tol),
			'x0': x0,
			'pos': pos,
			'G_hat': G_hat,
			'convex_gap': float(T_obs),
			'convex_tol': tol,
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail,
			'distance_ks': float(np.max(np.abs(diff))),
			'distance_ise': _ise_on_grid(grid_x, diff)
		}
		g_cache[pos] = diag
		_vprint(
			f"[ValidG] pos={pos}, x0={x0:.6g}, tails=({mix_tail},{alt_tail}), "
			f"T={T_obs:.3g}, tol={tol:.3g}: {_ok_text(diag['valid'])}"
		)
		return diag

	def _first_valid_g_pos():
		_vprint("[ValidG-search] locating leftmost convex-compatible matched cutoff")
		right = x_count_pos
		if _valid_g(0)['valid']:
			_vprint(f"[ValidG-search] leftmost cutoff passes: pos=0, x_G={float(grid_x[candidate_indices[0]]):.6g}")
			return 0
		if not _valid_g(right)['valid']:
			_vprint(
				f"[ValidG-search] rightmost count-admissible cutoff fails: "
				f"pos={right}, x_count={float(grid_x[candidate_indices[right]]):.6g}"
			)
			return None
		left = 0
		while right - left > 1:
			mid = (left + right) // 2
			if _valid_g(mid)['valid']:
				right = mid
			else:
				left = mid
		_vprint(f"[ValidG-search] x_G pos={right}, x_G={float(grid_x[candidate_indices[right]]):.6g}")
		return right

	x_g_pos = _first_valid_g_pos()
	if x_g_pos is None:
		return np.nan, None, None, None, grid_x, {
			'success': False,
			'reason': 'no_convex_compatible_matched_cutoff',
			'mode': 'semiadjustable_x0_separated_bootstrap',
			'alpha': np.nan,
			'x0': None,
			'x_G': None,
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_checks': int(alpha_checks),
			'cdf_checks': int(cdf_checks),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'shape_r_checks': int(shape_r_checks),
			'flat_tail_checks': int(flat_tail_checks),
			'require_flat_tail_reject': bool(require_flat_tail_reject),
			'use_tail_count_cutoff': bool(use_tail_count_cutoff),
			'flat_bisect_mode': flat_bisect_mode,
			'B': int(B),
			'delta': float(delta),
			'bootstrap_quantile': float(q),
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices))
		}
	g_anchor_diag = _valid_g(x_g_pos)
	G_hat_anchor = g_anchor_diag['G_hat']
	_vprint(
		f"[ValidG-anchor] using x_G={g_anchor_diag['x0']:.6g} "
		"for the matched bootstrap component"
	)

	def _shape_r(alpha, pos, label="ShapeR"):
		nonlocal cutoff_checks, shape_r_checks
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in shape_cache:
			diag = shape_cache[key]
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, x0={float(diag['x0']):.6g}, "
				f"tails=({int(diag['mix_tail_count'])},{int(diag['alt_tail_count'])}), "
				f"T={float(diag['shape_gap']):.3g}/{float(diag['shape_tol']):.3g}, "
				f"ShapeR={_ok_text(diag['shape_ok'])}"
			)
			return diag
		cutoff_checks += 1
		shape_r_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		T_obs = _gap_T(grid_x, R_obs, x0, x1)
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat_anchor)
		Tstar = np.empty(B, dtype=float)
		for b in range(B):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat_anchor, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			Tstar[b] = _gap_T(grid_x, R_b, x0, x1)
		shape_tol = float(np.quantile(Tstar, q, method='higher'))
		shape_ok = bool(T_obs <= shape_tol)
		mix_tail, alt_tail = _tail_counts_at_pos(pos)
		diag = {
			'alpha': alpha,
			'pos': pos,
			'x0': x0,
			'shape_ok': shape_ok,
			'shape_gap': float(T_obs),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_mon_violation': float(cdf_obs['mon']),
			'H_hat': H_hat,
			'G_hat': G_hat_anchor,
			'R_obs': R_obs,
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail
		}
		shape_cache[key] = diag
		_vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"tails=({mix_tail},{alt_tail}), T={T_obs:.3g}/{shape_tol:.3g}, "
			f"ShapeR={_ok_text(shape_ok)}"
		)
		return diag

	def _flat_reject(alpha, pos, shape_diag, label="FlatReject", allow_shape_fail=False):
		nonlocal cutoff_checks, flat_tail_checks
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in flat_cache:
			flat_diag = flat_cache[key]
			shape_diag.update(flat_diag)
			_vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, "
				f"x0={float(shape_diag['x0']):.6g}, "
				f"U={float(flat_diag['flat_stat']):.3g}/{float(flat_diag['flat_tol']):.3g}, "
				f"FlatReject={_ok_text(flat_diag['flat_reject'])}"
				f"{', ShapeR=NO directional' if flat_diag.get('flat_computed_with_shape_fail', False) else ''}"
			)
			return flat_diag
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not allow_shape_fail:
			flat_diag = {
				'flat_available': False,
				'flat_reject': False,
				'flat_stat': np.nan,
				'flat_tol': np.nan,
				'flat_delta': float(flat_delta),
				'flat_bootstrap_quantile': float(q_flat),
				'flat_reason': 'shape_failed'
			}
			shape_diag.update(flat_diag)
			return flat_diag
		cutoff_checks += 1
		flat_tail_checks += 1
		x0 = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		U_obs, H_flat = _flat_tail_stat(grid_x, shape_diag['H_hat'], x0, x1)
		H_flat = _as_valid_cdf(H_flat)
		F_flat = _as_valid_cdf((1.0 - alpha)*H_flat + alpha*G_hat_anchor)
		Ustar = np.empty(B, dtype=float)
		for b in range(B):
			x_mix_b = _sample_from_cdf(grid_x, F_flat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat_anchor, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			H_b = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_b, x0, x1))
			Ustar[b], _ = _flat_tail_stat(grid_x, H_b, x0, x1)
		flat_tol = float(np.quantile(Ustar, q_flat, method='higher'))
		flat_reject = bool(U_obs > flat_tol)
		flat_diag = {
			'flat_available': True,
			'flat_reject': flat_reject,
			'flat_stat': float(U_obs),
			'flat_tol': flat_tol,
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'flat_reason': 'computed',
			'flat_computed_with_shape_fail': bool(not shape_ok),
			'H_flat': H_flat
		}
		shape_diag.update(flat_diag)
		flat_cache[key] = flat_diag
		_vprint(
			f"[{label} #{flat_tail_checks}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"U={U_obs:.3g}/{flat_tol:.3g}, FlatReject={_ok_text(flat_reject)}"
			f"{', ShapeR=NO directional' if not shape_ok else ''}"
		)
		return flat_diag

	def _cutoff_selection_ok(alpha, pos, shape_diag, label="cutoff"):
		if not shape_diag.get('shape_ok', False):
			return False
		if not require_flat_tail_reject:
			return True
		flat_diag = _flat_reject(alpha, pos, shape_diag, label=label)
		return bool(flat_diag['flat_reject'])

	def _cutoff_state(alpha, pos, shape_diag, label="cutoff"):
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not require_flat_tail_reject:
			return 'shape_fail'
		if not shape_ok and flat_bisect_mode != "guarded":
			return 'shape_fail'
		if not require_flat_tail_reject:
			return 'pass'
		flat_diag = _flat_reject(
			alpha, pos, shape_diag, label=label,
			allow_shape_fail=(flat_bisect_mode == "guarded")
		)
		flat_ok = bool(flat_diag['flat_reject'])
		if shape_ok and flat_ok:
			return 'pass'
		if not shape_ok and flat_ok:
			return 'shape_fail_flat_ok'
		if not shape_ok:
			return 'shape_fail_flat_fail'
		return 'flat_fail'

	def _derive_shape_cutoff(alpha, label="derive"):
		start_pos = int(x_g_pos)
		if start_pos > x_count_pos:
			return False, None, None, {
				'status': 'start_cutoff_exceeds_count_cutoff',
				'stop': None,
				'left_pos': None,
				'right_pos': None,
				'width': None
			}
		start_diag = _shape_r(alpha, start_pos, label=f"{label}-xG")
		start_state = _cutoff_state(alpha, start_pos, start_diag, label=f"{label}-flat-xG")
		if start_state == 'pass':
			return True, start_pos, start_diag, {
				'status': 'x_G_passes',
				'stop': 'leftmost',
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		if start_state == 'flat_fail' and flat_bisect_mode != "guarded":
			return False, None, start_diag, {
				'status': 'x_G_flat_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		if start_pos == x_count_pos:
			return False, None, start_diag, {
				'status': 'single_cutoff_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': start_pos,
				'width': 0.0
			}
		right_diag = _shape_r(alpha, x_count_pos, label=f"{label}-xcount")
		right_state = _cutoff_state(alpha, x_count_pos, right_diag, label=f"{label}-flat-xcount")
		if right_state == 'shape_fail' or right_state == 'shape_fail_flat_ok':
			return False, None, right_diag, {
				'status': 'x_count_shape_fails',
				'stop': None,
				'left_pos': start_pos,
				'right_pos': x_count_pos,
				'width': _cutoff_width(start_pos, x_count_pos)
			}
		left = start_pos
		right = x_count_pos
		best_pos = x_count_pos if right_state == 'pass' else None
		best_diag = right_diag if right_state == 'pass' else None
		while right - left > 1 and _cutoff_width(left, right) > x0_tol:
			mid = (left + right) // 2
			mid_diag = _shape_r(alpha, mid, label=f"{label}-bisect")
			mid_state = _cutoff_state(alpha, mid, mid_diag, label=f"{label}-flat-bisect")
			if mid_state == 'shape_fail':
				if require_flat_tail_reject and flat_bisect_mode == "guarded":
					right = mid
					right_diag = mid_diag
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-fail; move hi")
				else:
					left = mid
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail; move lo")
			elif mid_state == 'shape_fail_flat_ok':
				left = mid
				_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-ok; move lo")
			elif mid_state in {'flat_fail', 'shape_fail_flat_fail'}:
				right = mid
				right_diag = mid_diag
				if mid_state == 'shape_fail_flat_fail':
					_vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-fail; move hi")
				else:
					_vprint(f"[{label}-cutoff-bisect] pos={mid} flat-fail; move hi")
			else:
				right = mid
				right_diag = mid_diag
				best_pos = mid
				best_diag = mid_diag
				_vprint(f"[{label}-cutoff-bisect] pos={mid} passes; move hi")
		stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		if best_diag is None:
			return False, None, right_diag, {
				'status': 'no_shape_flat_overlap' if require_flat_tail_reject else 'no_shape_passing_cutoff',
				'stop': stop_reason,
				'left_pos': int(left),
				'right_pos': int(right),
				'width': _cutoff_width(left, right)
			}
		if best_diag['pos'] != best_pos:
			best_diag = _shape_r(alpha, best_pos, label=f"{label}-best")
			_cutoff_selection_ok(alpha, best_pos, best_diag, label=f"{label}-flat-best")
		return True, int(best_pos), best_diag, {
			'status': 'found',
			'stop': stop_reason,
			'left_pos': int(left),
			'right_pos': int(best_pos),
			'width': _cutoff_width(left, int(best_pos))
		}

	def _compact_shape(diag):
		if diag is None:
			return None
		out = {
			'alpha': float(diag['alpha']),
			'pos': int(diag['pos']),
			'x0': float(diag['x0']),
			'shape_ok': bool(diag['shape_ok']),
			'cutoff_selection_ok': bool(
				diag.get('shape_ok', False) and
				(not require_flat_tail_reject or bool(diag.get('flat_reject', False)))
			),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'mix_tail_count': int(diag['mix_tail_count']),
			'alt_tail_count': int(diag['alt_tail_count'])
		}
		if require_flat_tail_reject or 'flat_reject' in diag:
			out.update({
				'flat_available': bool(diag.get('flat_available', False)),
				'flat_reject': None if diag.get('flat_reject') is None else bool(diag.get('flat_reject')),
				'flat_stat': float(diag.get('flat_stat', np.nan)),
				'flat_tol': float(diag.get('flat_tol', np.nan)),
				'flat_delta': float(diag.get('flat_delta', flat_delta)),
				'flat_bootstrap_quantile': float(diag.get('flat_bootstrap_quantile', q_flat)),
				'flat_reason': diag.get('flat_reason'),
				'flat_computed_with_shape_fail': bool(diag.get('flat_computed_with_shape_fail', False))
			})
		return out

	def _compact_cdf(diag):
		if diag is None:
			return None
		out = {
			'alpha': float(diag['alpha']),
			'cdf_ok': bool(diag['cdf_ok']),
			'cdf_available': bool(diag.get('cdf_available', True)),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_delta': float(diag.get('cdf_delta', cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', q_cdf)),
			'alpha_ref': None if diag.get('alpha_ref') is None else float(diag['alpha_ref']),
			'x_ref': None if diag.get('x_ref') is None else float(diag['x_ref']),
			'ref_distance': None if diag.get('ref_distance') is None else float(diag['ref_distance']),
			'weak_ref_calibration': bool(diag.get('weak_ref_calibration', False))
		}
		return out

	def _record_shape_ref(alpha, pos, diag, source):
		rec = {
			'alpha': float(alpha),
			'pos': int(pos),
			'x0': float(grid_x[candidate_indices[int(pos)]]),
			'source': source,
			'shape_diag': diag
		}
		shape_refs.append(rec)
		return rec

	def _nearest_shape_ref(alpha):
		if not shape_refs:
			return None
		alpha = float(alpha)
		return min(shape_refs, key=lambda rec: abs(float(rec['alpha']) - alpha))

	def _evaluate_shape_path(alpha, label="shape-alpha"):
		nonlocal alpha_checks, x0_star_pos
		alpha = float(alpha)
		alpha_checks += 1
		old_pos = x0_star_pos
		_vprint(
			f"[{label} #{alpha_checks}] alpha={alpha:.6f}, "
			f"current_x0={'None' if x0_star_pos is None else f'{float(grid_x[candidate_indices[x0_star_pos]]):.6g}'}"
		)
		if alpha >= 1.0 - 1e-12:
			return False, old_pos, None, {
				'updated': False,
				'old_pos': old_pos,
				'new_pos': old_pos,
				'search': {'status': 'alpha_too_close_to_one'}
			}
		if x0_star_pos is not None:
			current_diag = _shape_r(alpha, x0_star_pos, label=f"{label}-current")
			if _cutoff_selection_ok(alpha, x0_star_pos, current_diag, label=f"{label}-flat-current"):
				_record_shape_ref(alpha, x0_star_pos, current_diag, source='current')
				return True, x0_star_pos, current_diag, {
					'updated': False,
					'old_pos': old_pos,
					'new_pos': x0_star_pos,
					'search': {'status': 'current_cutoff_passes'}
				}
		ok, new_pos, new_diag, search = _derive_shape_cutoff(alpha, label=f"{label}-derive")
		if not ok:
			return False, old_pos, new_diag, {
				'updated': False,
				'old_pos': old_pos,
				'new_pos': old_pos,
				'search': search
			}
		x0_star_pos = int(new_pos)
		_record_shape_ref(alpha, x0_star_pos, new_diag, source='derived')
		update = {
			'alpha': float(alpha),
			'old_pos': None if old_pos is None else int(old_pos),
			'old_x0': None if old_pos is None else float(grid_x[candidate_indices[int(old_pos)]]),
			'new_pos': int(x0_star_pos),
			'new_x0': float(grid_x[candidate_indices[int(x0_star_pos)]]),
			'search': search,
			'shape_diag': _compact_shape(new_diag)
		}
		if old_pos != x0_star_pos:
			cutoff_update_history.append(update)
		return True, x0_star_pos, new_diag, {
			'updated': old_pos != x0_star_pos,
			'old_pos': old_pos,
			'new_pos': x0_star_pos,
			'search': search
		}

	def _cdf_at(alpha, pos, alpha_ref, label="CDF"):
		nonlocal cdf_checks
		alpha = float(alpha)
		pos = int(pos)
		alpha_ref = float(alpha_ref)
		key = (alpha, pos, alpha_ref)
		if key in cdf_cache:
			return cdf_cache[key]
		cdf_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha,
				'pos': pos,
				'x_ref': float(grid_x[candidate_indices[pos]]),
				'alpha_ref': alpha_ref,
				'ref_distance': abs(alpha - alpha_ref),
				'weak_ref_calibration': abs(alpha - alpha_ref) > cdf_ref_max_dist,
				'cdf_available': True,
				'cdf_ok': False,
				'cdf_range_violation': np.inf,
				'cdf_low_violation': np.inf,
				'cdf_high_violation': np.inf,
				'cdf_range_tol': np.nan,
				'cdf_mon_violation': np.inf,
				'cdf_mon_tol': np.nan,
				'H_hat': None,
				'R_obs': None
			}
			cdf_cache[key] = diag
			return diag
		x_ref = float(grid_x[candidate_indices[pos]])
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		shape_diag = shape_cache.get((alpha, pos))
		H_hat = shape_diag['H_hat'] if shape_diag is not None else _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x_ref, x1))
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*G_hat_anchor)
		range_star = np.empty(B, dtype=float)
		mon_star = np.empty(B, dtype=float)
		for b in range(B):
			x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(grid_x, G_hat_anchor, m, rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
		range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		ref_distance = abs(alpha - alpha_ref)
		diag = {
			'alpha': alpha,
			'pos': pos,
			'x_ref': x_ref,
			'alpha_ref': alpha_ref,
			'ref_distance': float(ref_distance),
			'weak_ref_calibration': bool(ref_distance > cdf_ref_max_dist),
			'cdf_available': True,
			'cdf_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_delta': float(cdf_delta),
			'cdf_bootstrap_quantile': float(q_cdf),
			'H_hat': H_hat,
			'R_obs': R_obs
		}
		cdf_cache[key] = diag
		_vprint(
			f"[{label} #{cdf_checks}] alpha={alpha:.6f}, x_ref={x_ref:.6g} "
			f"(alpha_ref={alpha_ref:.6f}, dist={ref_distance:.3g}), "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, CDFOK={_ok_text(cdf_ok)}"
		)
		return diag

	def _evaluate_cdf(alpha, label="cdf-alpha", pos=None, alpha_ref=None):
		if pos is None:
			ref = _nearest_shape_ref(alpha)
			if ref is None:
				alpha = float(alpha)
				return {
					'alpha': alpha,
					'pos': None,
					'x_ref': None,
					'alpha_ref': None,
					'ref_distance': None,
					'weak_ref_calibration': True,
					'cdf_available': False,
					'cdf_ok': False,
					'cdf_range_violation': np.nan,
					'cdf_low_violation': np.nan,
					'cdf_high_violation': np.nan,
					'cdf_range_tol': np.nan,
					'cdf_mon_violation': np.nan,
					'cdf_mon_tol': np.nan,
					'H_hat': None,
					'R_obs': None
				}
			pos = ref['pos']
			alpha_ref = ref['alpha']
		elif alpha_ref is None:
			alpha_ref = alpha
		return _cdf_at(alpha, pos, alpha_ref, label=label)

	def _full_check(alpha, label="full"):
		ok_shape, pos, shape_diag, shape_meta = _evaluate_shape_path(alpha, label=f"{label}-shape")
		if ok_shape:
			cdf_diag = _evaluate_cdf(alpha, label=f"{label}-cdf", pos=pos, alpha_ref=alpha)
		else:
			cdf_diag = _evaluate_cdf(alpha, label=f"{label}-cdf")
		return {
			'alpha': float(alpha),
			'shape_ok': bool(ok_shape),
			'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
			'full_ok': bool(ok_shape and cdf_diag.get('cdf_ok', False)),
			'pos': None if pos is None else int(pos),
			'x0': None if pos is None else float(grid_x[candidate_indices[int(pos)]]),
			'shape_diag': shape_diag,
			'cdf_diag': cdf_diag,
			'shape_meta': shape_meta
		}

	def _shape_probe_record(diag, meta):
		out = _compact_shape(diag)
		if out is None:
			out = {}
		out.update({
			'search': meta.get('search') if meta else None,
			'updated': bool(meta.get('updated', False)) if meta else False
		})
		return out

	def _make_info(success, reason, alpha=None, pos=None, shape_diag=None, cdf_diag=None, extra=None):
		x0 = float(grid_x[candidate_indices[pos]]) if pos is not None else None
		info = {
			'success': bool(success),
			'reason': reason,
			'mode': 'semiadjustable_x0_separated_bootstrap',
			'alpha': float(alpha) if alpha is not None else np.nan,
			'x0': x0,
			'x_G': float(grid_x[candidate_indices[x_g_pos]]),
			'x_count': float(grid_x[candidate_indices[x_count_pos]]),
			'alpha_checks': int(alpha_checks),
			'cdf_checks': int(cdf_checks),
			'total_alpha_checks': int(_total_alpha_checks()),
			'cutoff_checks': int(cutoff_checks),
			'valid_g_checks': int(valid_g_checks),
			'shape_r_checks': int(shape_r_checks),
			'flat_tail_checks': int(flat_tail_checks),
				'B': int(B),
				'delta': float(delta),
				'bootstrap_quantile': float(q),
				'cdf_delta': float(cdf_delta),
				'cdf_bootstrap_quantile': float(q_cdf),
			'alpha_tol': float(alpha_tol),
			'x0_tol': float(x0_tol),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
			'cdf_ref_max_dist': float(cdf_ref_max_dist),
			'require_flat_tail_reject': bool(require_flat_tail_reject),
			'flat_delta': float(flat_delta),
			'flat_bootstrap_quantile': float(q_flat),
			'use_tail_count_cutoff': bool(use_tail_count_cutoff),
			'flat_bisect_mode': flat_bisect_mode,
			'max_checks': int(max_checks),
			'n_alpha_probes': int(n_alpha_probes),
			'n_grid': int(len(grid_x)),
			'n_candidate_cutoffs': int(len(candidate_indices)),
			'n_mix': int(n),
			'n_alt': int(m),
			'min_mix_tail': int(min_mix_tail),
			'min_alt_tail': int(min_alt_tail),
			'G_anchor_x0': float(g_anchor_diag['x0']),
			'G_convex_gap': float(g_anchor_diag['convex_gap']),
			'G_convex_tol': float(g_anchor_diag['convex_tol']),
			'G_anchor_valid': bool(g_anchor_diag['valid']),
			'G_distance_ks': float(g_anchor_diag['distance_ks']),
			'G_distance_ise': float(g_anchor_diag['distance_ise']),
			'cutoff_update_history': cutoff_update_history
		}
		if shape_diag is not None:
			info.update({
				'shape_ok': bool(shape_diag['shape_ok']),
				'cutoff_selection_ok': bool(
					shape_diag.get('shape_ok', False) and
					(not require_flat_tail_reject or bool(shape_diag.get('flat_reject', False)))
				),
				'residual_lcm_gap': float(shape_diag['shape_gap']),
				'residual_lcm_tol': float(shape_diag['shape_tol']),
				'mix_tail_count': int(shape_diag['mix_tail_count']),
				'alt_tail_count': int(shape_diag['alt_tail_count'])
			})
			if require_flat_tail_reject or 'flat_reject' in shape_diag:
				info.update({
					'flat_available': bool(shape_diag.get('flat_available', False)),
					'flat_reject': None if shape_diag.get('flat_reject') is None else bool(shape_diag.get('flat_reject')),
					'flat_stat': float(shape_diag.get('flat_stat', np.nan)),
					'flat_tol': float(shape_diag.get('flat_tol', np.nan)),
					'flat_delta': float(shape_diag.get('flat_delta', flat_delta)),
					'flat_bootstrap_quantile': float(shape_diag.get('flat_bootstrap_quantile', q_flat)),
					'flat_reason': shape_diag.get('flat_reason'),
					'flat_computed_with_shape_fail': bool(shape_diag.get('flat_computed_with_shape_fail', False))
				})
		if cdf_diag is not None:
			info.update({
				'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
				'cdf_available': bool(cdf_diag.get('cdf_available', True)),
				'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
				'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
				'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
				'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
					'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
					'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
					'cdf_delta': float(cdf_diag.get('cdf_delta', cdf_delta)),
					'cdf_bootstrap_quantile': float(cdf_diag.get('cdf_bootstrap_quantile', q_cdf)),
				'cdf_alpha_ref': None if cdf_diag.get('alpha_ref') is None else float(cdf_diag['alpha_ref']),
				'cdf_x_ref': None if cdf_diag.get('x_ref') is None else float(cdf_diag['x_ref']),
				'cdf_ref_distance': None if cdf_diag.get('ref_distance') is None else float(cdf_diag['ref_distance']),
				'cdf_weak_ref_calibration': bool(cdf_diag.get('weak_ref_calibration', False))
			})
		if extra:
			info.update(extra)
		return info

	shape_bracket = _run_shape_alpha_bracket(
		alpha_left=alpha_left,
		alpha_right=alpha_right,
		n_alpha_probes=n_alpha_probes,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=_total_alpha_checks,
		evaluate_shape_path=_evaluate_shape_path,
		shape_probe_record=_shape_probe_record,
		vprint=_vprint
	)
	probe_alphas = shape_bracket['probe_alphas']
	shape_probe_table = shape_bracket['shape_probe_table']
	if not shape_bracket['success']:
		info = _make_info(
			False, shape_bracket['reason'],
			extra={'shape_probe_table': shape_probe_table}
		)
		return np.nan, None, None, G_hat_anchor, grid_x, info

	alpha_shape_diag = shape_bracket['alpha_shape_diag']
	alpha_shape_pos = shape_bracket['alpha_shape_pos']
	shape_boundary = shape_bracket['shape_boundary']
	alpha_shape = float(shape_bracket['alpha_shape'])
	cdf_at_shape = _evaluate_cdf(alpha_shape, label="direct-cdf-at-shape", pos=alpha_shape_pos, alpha_ref=alpha_shape)
	if cdf_at_shape['cdf_ok']:
		cdf_probe_table = [_compact_cdf(cdf_at_shape)]
		shared_extra = {
			'alpha_shape': float(alpha_shape),
			'alpha_cdf': float(alpha_shape),
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {
				'status': 'direct_alpha_shape_pass',
				'lo': float(alpha_shape),
				'hi': float(alpha_shape),
				'width': 0.0
			},
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'alpha_shape_diag': _compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': _compact_cdf(cdf_at_shape),
			'n_shape_refs': int(len(shape_refs))
		}
		_vprint(f"[direct-alpha-shape] alpha={alpha_shape:.6f}, ShapeR=OK, CDFOK=OK; stopping")
		info = _make_info(
			True, 'overlap_shape_boundary',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=shared_extra
		)
		return alpha_shape, info['x0'], alpha_shape_diag['H_hat'], G_hat_anchor, grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=alpha_right,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=_total_alpha_checks,
		evaluate_cdf=_evaluate_cdf,
		compact_cdf=_compact_cdf,
		vprint=_vprint,
		stop_after_pass_at=alpha_shape
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	if not cdf_boundary_result['success']:
		cdf_diag_at_shape = _evaluate_cdf(alpha_shape, label="cdf-at-shape", pos=alpha_shape_pos, alpha_ref=alpha_shape)
		extra = {
			'alpha_shape': float(alpha_shape),
			'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'alpha_shape_diag': _compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': None
		}
		info = _make_info(
			False, 'no_cdf_passing_probe',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_diag_at_shape,
			extra=extra
		)
		return np.nan, None, None, G_hat_anchor, grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	cdf_boundary = cdf_boundary_result['cdf_boundary']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = {
		'alpha_shape': float(alpha_shape),
		'alpha_cdf': float(alpha_cdf),
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary,
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'alpha_shape_diag': _compact_shape(alpha_shape_diag),
		'alpha_cdf_diag': _compact_cdf(alpha_cdf_diag),
		'n_shape_refs': int(len(shape_refs))
	}
	_vprint(
		f"[boundaries] alpha_shape={alpha_shape:.6f}, alpha_cdf={alpha_cdf:.6f}, "
		f"gap={gap:.3g}, status={shared_extra['gap_status']}"
	)

	def _scan_for_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or _total_alpha_checks() >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan = []
		for a in alphas:
			if _total_alpha_checks() >= max_checks:
				break
			full = _full_check(a, label=label)
			scan.append({
				'alpha': float(full['alpha']),
				'shape_ok': bool(full['shape_ok']),
				'cdf_ok': bool(full['cdf_ok']),
				'full_ok': bool(full['full_ok']),
				'x0': full['x0'],
				'shape_diag': _compact_shape(full['shape_diag']),
				'cdf_diag': _compact_cdf(full['cdf_diag'])
			})
			if full['full_ok']:
				return full, scan
		return None, scan

	cdf_at_shape = _evaluate_cdf(alpha_shape, label="cdf-at-shape", pos=alpha_shape_pos, alpha_ref=alpha_shape)
	if gap <= 0.0:
		if cdf_at_shape['cdf_ok']:
			extra = dict(shared_extra)
			info = _make_info(
				True, 'overlap_shape_boundary',
				alpha=alpha_shape, pos=alpha_shape_pos,
				shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
				extra=extra
			)
			return alpha_shape, info['x0'], alpha_shape_diag['H_hat'], G_hat_anchor, grid_x, info
		full, scan = _scan_for_full_pass(alpha_shape, alpha_cdf, "overlap-scan")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = scan
		if full is not None:
			info = _make_info(
				True, 'overlap_local_full_pass',
				alpha=full['alpha'], pos=full['pos'],
				shape_diag=full['shape_diag'], cdf_diag=full['cdf_diag'],
				extra=extra
			)
			return float(full['alpha']), info['x0'], full['shape_diag']['H_hat'], G_hat_anchor, grid_x, info
		info = _make_info(
			False, 'overlap_without_full_passing_probe',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=extra
		)
		return np.nan, None, None, G_hat_anchor, grid_x, info

	if gap <= gap_tol:
		full, scan = _scan_for_full_pass(alpha_cdf, alpha_shape, "near-miss")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = scan
		if full is not None:
			info = _make_info(
				True, 'near_miss_full_passing_probe',
				alpha=full['alpha'], pos=full['pos'],
				shape_diag=full['shape_diag'], cdf_diag=full['cdf_diag'],
				extra=extra
			)
			return float(full['alpha']), info['x0'], full['shape_diag']['H_hat'], G_hat_anchor, grid_x, info
		info = _make_info(
			False, 'near_miss_no_overlap',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=extra
		)
		return np.nan, None, None, G_hat_anchor, grid_x, info

	info = _make_info(
		False, 'separated_boundaries_gap',
		alpha=alpha_shape, pos=alpha_shape_pos,
		shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
		extra=shared_extra
	)
	return np.nan, None, None, G_hat_anchor, grid_x, info


def mixture_sanity_check(Fn, Gm, H_hat, alpha_hat, x_grid,
	delta=0.05, B=400, rng=None, return_sim=False, sample_size=None):
	"""
	Post-fit sanity check for the mixture:
	    M_hat = (1 - alpha_hat) * H_hat + alpha_hat * Gm
	Compare the empirical mixture ECDF F_n to M_hat on the same grid.

	Returns dict:
	  - 'D_ks'   : sup_x |F_n - M_hat|
	  - 'ISE'    : ∫ (F_n - M_hat)^2 dx  (trapezoid on x_grid)
	  - 'crit'   : bootstrap (1-δ)-quantile
	  - 'p_value': bootstrap Monte Carlo p-value
	  - 'pass_test'   : D_ks <= crit
	  - 'M_hat'  : fitted mixture CDF on x_grid
	"""
	if rng is None:
		rng = np.random.default_rng()
	B = int(B)
	if B < 1:
		raise ValueError("B must be at least 1.")

	# Fitted mixture CDF (guard against tiny numeric drifts)
	Fn = np.asarray(Fn, dtype=float)
	M_hat = (1.0 - float(alpha_hat)) * np.asarray(H_hat, dtype=float) + float(alpha_hat) * np.asarray(Gm, dtype=float)
	M_hat = np.minimum(1.0, np.maximum.accumulate(np.maximum(M_hat, 0.0)))

	diff = Fn - M_hat
	D_ks = float(np.max(np.abs(diff)))

	# L2 over x via trapezoid
	if len(x_grid) >= 2:
		ISE = float(np.sum(0.5 * (diff[:-1]**2 + diff[1:]**2) * np.diff(x_grid)))
	else:
		ISE = 0.0

	if sample_size is None:
		jumps = np.diff(np.concatenate(([0.0], Fn)))
		jumps = jumps[np.isfinite(jumps) & (jumps > 10.0*np.finfo(float).eps)]
		if len(jumps) == 0:
			sample_size = len(x_grid)
		else:
			sample_size = int(round(1.0 / float(np.min(jumps))))
	else:
		sample_size = int(sample_size)
	if sample_size < 1:
		raise ValueError("sample_size must be positive.")

	# Bootstrap calibration (fixed-model; conservative)
	Dstar = np.empty(B, dtype=float)
	for b in range(B):
		x_b = _sample_from_cdf(x_grid, M_hat, size=sample_size, rng=rng)
		x_b.sort(kind='mergesort')
		Fn_b = _ecdf_on_grid(x_b, x_grid)
		Dstar[b] = float(np.max(np.abs(Fn_b - M_hat)))

	q = 1.0 - float(delta)
	crit = float(np.quantile(Dstar, q, method='higher'))
	pv = float((1 + np.sum(Dstar >= D_ks)) / (B + 1))
	out = dict(D_ks=D_ks, ISE=ISE, crit=crit, p_value=pv, pass_test=(D_ks <= crit), M_hat=M_hat,
		bootstrap_sample_size=int(sample_size))
	if return_sim:
		out['Dstar'] = Dstar
	return out


def _ok_text(ok):
	return "OK" if ok else "NO"


def _diag_float(value):
	if value is None:
		return None
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return out if np.isfinite(out) else None


def _check_alpha_bounds(alpha_bounds):
	alpha_left, alpha_right = alpha_bounds
	alpha_left = max(0.0, float(alpha_left))
	alpha_right = min(0.999999, float(alpha_right))
	if not (0.0 <= alpha_left < alpha_right < 1.0):
		raise ValueError("alpha_bounds must satisfy 0 <= lo < hi < 1.")
	return alpha_left, alpha_right


def _normalize_confirm_tail_mode(confirm_tail_mode, require_flat_tail_reject):
	if confirm_tail_mode is None:
		if isinstance(require_flat_tail_reject, str):
			value = require_flat_tail_reject
		else:
			return "flat" if bool(require_flat_tail_reject) else "none"
	else:
		value = confirm_tail_mode
	if isinstance(value, str):
		key = value.strip().lower().replace("-", "_")
		aliases = {
			"none": "none",
			"false": "none",
			"no": "none",
			"0": "none",
			"flat": "flat",
			"true": "flat",
			"yes": "flat",
			"1": "flat",
			"det": "deterministic",
			"deterministic": "deterministic",
			"determinstic": "deterministic",
		}
		if key in aliases:
			return aliases[key]
	if isinstance(value, (bool, np.bool_)):
		return "flat" if bool(value) else "none"
	raise ValueError("confirm_tail_mode must be 'none', 'flat', or 'deterministic'.")


@dataclass(frozen=True)
class _BootstrapPlan:
	B: int
	delta: float
	q: float
	cdf_B: int
	cdf_delta: float
	q_cdf: float
	flat_delta: float
	q_flat: float

	@classmethod
	def make(cls, label, B, delta, cdf_delta=None, cdf_B=None,
			flat_delta=None, require_flat_tail_reject=False):
		B = int(B)
		if B < 1:
			raise ValueError("B must be at least 1.")
		cdf_B = B if cdf_B is None else int(cdf_B)
		if cdf_B < 1:
			raise ValueError("cdf_B must be at least 1.")
		delta = float(delta)
		cdf_delta = delta if cdf_delta is None else float(cdf_delta)
		flat_delta = delta if flat_delta is None else float(flat_delta)
		q = _bootstrap_quantile_level(delta, B, label)
		q_cdf = _bootstrap_quantile_level(cdf_delta, cdf_B, f"{label}.cdf")
		q_flat = (
			_bootstrap_quantile_level(flat_delta, B, f"{label}.flat_tail")
			if require_flat_tail_reject else q
		)
		return cls(
			B=B, delta=delta, q=q,
			cdf_B=cdf_B, cdf_delta=cdf_delta, q_cdf=q_cdf,
			flat_delta=flat_delta, q_flat=q_flat
		)

	def info(self):
		return {
			'B': int(self.B),
			'delta': float(self.delta),
			'bootstrap_quantile': float(self.q),
			'cdf_B': int(self.cdf_B),
			'cdf_delta': float(self.cdf_delta),
			'cdf_bootstrap_quantile': float(self.q_cdf),
			'flat_delta': float(self.flat_delta),
			'flat_bootstrap_quantile': float(self.q_flat),
		}


class _DerivedSearchContext:
	def __init__(
		self,
		x_mix,
		x_alt,
		*,
		label,
		alpha_bounds,
		n_alpha_probes,
		alpha_tol,
		boot,
		confirm_boot=None,
		min_mix_tail,
		min_alt_tail,
		random_state,
		grid,
		candidate_cutoffs,
		max_grid_points,
		verbose,
		x0_tol,
		require_flat_tail_reject,
		use_tail_count_cutoff,
		flat_bisect_mode,
		cdf_ref_max_dist,
		shape_weight_gamma=2.0,
	):
		self.label = label
		self.verbose = bool(verbose)
		self.random_state = random_state
		self.rng = (
			random_state if isinstance(random_state, np.random.Generator)
			else np.random.default_rng(random_state)
		)
		self.x_mix = np.sort(np.asarray(x_mix, dtype=float))
		self.x_alt = np.sort(np.asarray(x_alt, dtype=float))
		if len(self.x_mix) == 0 or len(self.x_alt) == 0:
			raise ValueError("x_mix and x_alt must both be nonempty.")
		self.n = int(len(self.x_mix))
		self.m = int(len(self.x_alt))
		self.x1 = 0.0
		self.alpha_left, self.alpha_right = _check_alpha_bounds(alpha_bounds)
		self.n_alpha_probes = int(n_alpha_probes)
		if self.n_alpha_probes < 2:
			raise ValueError("n_alpha_probes must be at least 2.")
		if alpha_tol <= 0:
			raise ValueError("alpha_tol must be positive.")
		self.alpha_tol = float(alpha_tol)
		self.boot = boot
		self.confirm_boot = boot if confirm_boot is None else confirm_boot
		self.x0_tol = 0.0 if x0_tol is None else float(x0_tol)
		if not np.isfinite(self.x0_tol) or self.x0_tol < 0.0:
			raise ValueError("x0_tol must be finite and nonnegative.")
		self.require_flat_tail_reject = bool(require_flat_tail_reject)
		self.use_tail_count_cutoff = bool(use_tail_count_cutoff)
		self.flat_bisect_mode = str(flat_bisect_mode)
		if self.flat_bisect_mode not in {"three_way", "guarded"}:
			raise ValueError("flat_bisect_mode must be 'three_way' or 'guarded'.")
		self.min_mix_tail = int(min_mix_tail)
		self.min_alt_tail = int(min_alt_tail)
		if self.use_tail_count_cutoff:
			if self.min_mix_tail < 1 or self.min_alt_tail < 1:
				raise ValueError("min_mix_tail and min_alt_tail must be positive when use_tail_count_cutoff=True.")
		else:
			if self.min_mix_tail < 0 or self.min_alt_tail < 0:
				raise ValueError("min_mix_tail and min_alt_tail must be nonnegative.")
		probe_spacing = (self.alpha_right - self.alpha_left) / max(1, self.n_alpha_probes - 1)
		self.cdf_ref_max_dist = (
			probe_spacing if cdf_ref_max_dist is None else float(cdf_ref_max_dist)
		)
		if not np.isfinite(self.cdf_ref_max_dist) or self.cdf_ref_max_dist < 0.0:
			raise ValueError("cdf_ref_max_dist must be finite and nonnegative.")
		self.shape_weight_gamma = float(shape_weight_gamma)
		if not np.isfinite(self.shape_weight_gamma) or self.shape_weight_gamma < 0.0:
			raise ValueError("shape_weight_gamma must be finite and nonnegative.")

		self.grid_x = self._make_grid(grid, candidate_cutoffs, max_grid_points)
		self.candidate_indices = self._make_candidate_indices(candidate_cutoffs)
		self.Fn = _ecdf_on_grid(self.x_mix, self.grid_x)
		self.Gm = _ecdf_on_grid(self.x_alt, self.grid_x)

		self.setup_error = None
		self._apply_count_filter()

		self.alpha_checks = 0
		self.cdf_checks = 0
		self.cutoff_checks = 0
		self.valid_g_checks = 0
		self.shape_r_checks = 0
		self.flat_tail_checks = 0
		self.g_cache = {}
		self.shape_cache = {}
		self.flat_cache = {}
		self.cdf_cache = {}
		self.shape_refs = []
		self.cutoff_update_history = []
		self.working_pos = None
		self.x_g_pos = None
		self.g_anchor_diag = None
		self.G_hat_anchor = None

	def _make_grid(self, grid, candidate_cutoffs, max_grid_points):
		if grid is None:
			x_left = float(np.min([self.x_mix.min(), self.x_alt.min()]))
			grid_x = _build_grid(self.x_mix, self.x_alt, x_left, self.x1, max_points=max_grid_points)
		else:
			grid_x = np.sort(np.asarray(grid, dtype=float))
			grid_x = grid_x[np.isfinite(grid_x)]
			if len(grid_x) == 0:
				raise ValueError("grid must contain at least one finite value.")
		if candidate_cutoffs is not None:
			cutoff_values = np.sort(np.asarray(candidate_cutoffs, dtype=float))
			cutoff_values = cutoff_values[np.isfinite(cutoff_values)]
			cutoff_values = cutoff_values[cutoff_values < self.x1]
			self._candidate_values_from_input = cutoff_values
			grid_x = np.unique(np.concatenate([grid_x, cutoff_values, np.array([self.x1], dtype=float)]))
		else:
			self._candidate_values_from_input = None
			grid_x = np.unique(np.concatenate([grid_x, np.array([self.x1], dtype=float)]))
		grid_x = grid_x[grid_x <= self.x1]
		if len(grid_x) == 0:
			raise ValueError("grid has no points at or below 0.")
		if grid_x[-1] < self.x1:
			grid_x = np.concatenate([grid_x, [self.x1]])
		return grid_x

	def _make_candidate_indices(self, candidate_cutoffs):
		if candidate_cutoffs is None:
			candidate_values = self.grid_x[self.grid_x < self.x1]
		else:
			candidate_values = self._candidate_values_from_input
		candidate_values = np.unique(candidate_values[candidate_values < self.x1])
		if len(candidate_values) == 0:
			return np.array([], dtype=int)
		indices = np.array(
			[int(np.searchsorted(self.grid_x, x, side='left')) for x in candidate_values],
			dtype=int
		)
		indices = np.unique(indices)
		return indices[self.grid_x[indices] < self.x1]

	def _apply_count_filter(self):
		if len(self.candidate_indices) == 0:
			self.setup_error = {
				'reason': 'no_candidate_cutoffs',
				'Fn_at_grid_end': float(self.Fn[-1]) if len(self.Fn) else np.nan,
				'Gm_at_grid_end': float(self.Gm[-1]) if len(self.Gm) else np.nan,
			}
			self.candidate_x = np.array([], dtype=float)
			self.x_count_pos = None
			return
		candidate_x = self.grid_x[self.candidate_indices]
		mix_tail_counts = np.array([np.sum(self.x_mix >= x0) for x0 in candidate_x], dtype=int)
		alt_tail_counts = np.array([np.sum(self.x_alt >= x0) for x0 in candidate_x], dtype=int)
		if self.use_tail_count_cutoff:
			count_ok = (mix_tail_counts >= self.min_mix_tail) & (alt_tail_counts >= self.min_alt_tail)
			if not np.any(count_ok):
				self.setup_error = {
					'reason': 'no_count_admissible_cutoff',
					'min_mix_tail': int(self.min_mix_tail),
					'min_alt_tail': int(self.min_alt_tail),
					'max_mix_tail': int(np.max(mix_tail_counts)) if len(mix_tail_counts) else 0,
					'max_alt_tail': int(np.max(alt_tail_counts)) if len(alt_tail_counts) else 0,
				}
				self.candidate_indices = np.array([], dtype=int)
				self.candidate_x = np.array([], dtype=float)
				self.x_count_pos = None
				return
			self.candidate_indices = self.candidate_indices[count_ok]
		self.candidate_x = self.grid_x[self.candidate_indices]
		self.x_count_pos = len(self.candidate_indices) - 1

	def vprint(self, *args, **kwargs):
		if self.verbose:
			print(*args, **kwargs)

	def total_alpha_checks(self):
		return int(self.alpha_checks + self.cdf_checks)

	def candidate_value(self, pos):
		return float(self.grid_x[self.candidate_indices[int(pos)]])

	def tail_counts_at_pos(self, pos):
		x0 = self.candidate_value(pos)
		return int(np.sum(self.x_mix >= x0)), int(np.sum(self.x_alt >= x0))

	def cutoff_width(self, lo_pos, hi_pos):
		return float(self.candidate_value(hi_pos) - self.candidate_value(lo_pos))

	def base_info(self, mode):
		info = {
			'mode': mode,
			'alpha_checks': int(self.alpha_checks),
			'cdf_checks': int(self.cdf_checks),
			'total_alpha_checks': int(self.total_alpha_checks()),
			'cutoff_checks': int(self.cutoff_checks),
			'valid_g_checks': int(self.valid_g_checks),
			'shape_r_checks': int(self.shape_r_checks),
			'flat_tail_checks': int(self.flat_tail_checks),
			'alpha_tol': float(self.alpha_tol),
			'alpha_bounds': (float(self.alpha_left), float(self.alpha_right)),
			'n_alpha_probes': int(self.n_alpha_probes),
			'x0_tol': float(self.x0_tol),
			'cdf_ref_max_dist': float(self.cdf_ref_max_dist),
			'shape_statistic': 'weighted_ise',
			'shape_weight_gamma': float(self.shape_weight_gamma),
			'require_flat_tail_reject': bool(self.require_flat_tail_reject),
			'use_tail_count_cutoff': bool(self.use_tail_count_cutoff),
			'flat_bisect_mode': self.flat_bisect_mode,
			'n_grid': int(len(self.grid_x)),
			'n_candidate_cutoffs': int(len(self.candidate_indices)),
			'n_mix': int(self.n),
			'n_alt': int(self.m),
			'min_mix_tail': int(self.min_mix_tail),
			'min_alt_tail': int(self.min_alt_tail),
			'cutoff_update_history': self.cutoff_update_history,
		}
		info.update(self.boot.info())
		if self.x_count_pos is not None:
			info['x_count'] = self.candidate_value(self.x_count_pos)
		if self.x_g_pos is not None:
			info['x_G'] = self.candidate_value(self.x_g_pos)
		if self.g_anchor_diag is not None:
			info.update({
				'G_anchor_x0': float(self.g_anchor_diag['x0']),
				'G_convex_gap': float(self.g_anchor_diag['convex_gap']),
				'G_convex_tol': float(self.g_anchor_diag['convex_tol']),
				'G_anchor_valid': bool(self.g_anchor_diag['valid']),
				'G_distance_ks': float(self.g_anchor_diag['distance_ks']),
				'G_distance_ise': float(self.g_anchor_diag['distance_ise']),
			})
		return info

	def failure_info(self, mode, reason, extra=None):
		info = {
			'success': False,
			'reason': reason,
			'alpha': np.nan,
			'x0': None,
		}
		info.update(self.base_info(mode))
		if self.setup_error:
			info.update(self.setup_error)
		if extra:
			info.update(extra)
		return info

	def setup_print(self, tag, max_checks, gap_tol, extra=""):
		self.vprint(
			f"[{tag}] n={self.n}, m={self.m}, grid={len(self.grid_x)}, "
			f"boot_seed={_random_state_label(self.random_state)}, "
			f"alpha_bounds=({self.alpha_left:.6f}, {self.alpha_right:.6f}), "
			f"alpha_tol={self.alpha_tol:.3g}, x0_tol={self.x0_tol:.3g}, "
			f"gap_tol={float(gap_tol):.3g}, max_checks={int(max_checks)}, "
			f"B={self.boot.B}, delta={self.boot.delta:.3g}, q={self.boot.q:.6f}, "
			f"cdf_B={self.boot.cdf_B}, cdf_delta={self.boot.cdf_delta:.3g}, "
			f"q_cdf={self.boot.q_cdf:.6f}, flat_delta={self.boot.flat_delta:.3g}, "
			f"q_flat={self.boot.q_flat:.6f}{extra}"
		)
		if self.setup_error is not None:
			self.vprint(f"[{tag}] setup failure: {self.setup_error['reason']}")
		elif self.use_tail_count_cutoff:
			mix_tail, alt_tail = self.tail_counts_at_pos(self.x_count_pos)
			self.vprint(
				f"[count] admissible_cutoffs={len(self.candidate_indices)}, "
				f"x_min={self.candidate_value(0):.6g}, x_count={self.candidate_value(self.x_count_pos):.6g}, "
				f"tail_at_x_count=({mix_tail},{alt_tail})"
			)
		else:
			mix_tail, alt_tail = self.tail_counts_at_pos(self.x_count_pos)
			self.vprint(
				f"[count] tail-count cutoff disabled; candidate_cutoffs={len(self.candidate_indices)}, "
				f"x_min={self.candidate_value(0):.6g}, x_right={self.candidate_value(self.x_count_pos):.6g}, "
				f"tail_at_right=({mix_tail},{alt_tail})"
			)

	def valid_g(self, pos):
		pos = int(pos)
		if pos in self.g_cache:
			return self.g_cache[pos]
		self.cutoff_checks += 1
		self.valid_g_checks += 1
		x0 = self.candidate_value(pos)
		G_hat = _as_valid_cdf(_convex_projection_on_interval(self.grid_x, self.Gm, x0, self.x1))
		T_obs = _convex_gap_T(self.grid_x, self.Gm, x0, self.x1)
		Tstar = np.empty(self.boot.B, dtype=float)
		for b in range(self.boot.B):
			x_alt_b = _sample_from_cdf(self.grid_x, G_hat, self.m, self.rng)
			x_alt_b.sort(kind='mergesort')
			Gm_b = _ecdf_on_grid(x_alt_b, self.grid_x)
			Tstar[b] = _convex_gap_T(self.grid_x, Gm_b, x0, self.x1)
		tol = float(np.quantile(Tstar, self.boot.q, method='higher'))
		diff = self.Gm - G_hat
		mix_tail, alt_tail = self.tail_counts_at_pos(pos)
		diag = {
			'valid': bool(T_obs <= tol),
			'x0': x0,
			'pos': pos,
			'G_hat': G_hat,
			'convex_gap': float(T_obs),
			'convex_tol': tol,
			'bootstrap_quantile': float(self.boot.q),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail,
			'distance_ks': float(np.max(np.abs(diff))),
			'distance_ise': _ise_on_grid(self.grid_x, diff),
		}
		self.g_cache[pos] = diag
		self.vprint(
			f"[ValidG] pos={pos}, x0={x0:.6g}, tails=({mix_tail},{alt_tail}), "
			f"T={T_obs:.3g}, tol={tol:.3g}: {_ok_text(diag['valid'])}"
		)
		return diag

	def first_valid_g_pos(self):
		self.vprint("[ValidG-search] locating leftmost convex-compatible matched cutoff")
		right = self.x_count_pos
		if self.valid_g(0)['valid']:
			self.vprint(f"[ValidG-search] leftmost cutoff passes: pos=0, x_G={self.candidate_value(0):.6g}")
			return 0
		if not self.valid_g(right)['valid']:
			self.vprint(
				f"[ValidG-search] rightmost count-admissible cutoff fails: "
				f"pos={right}, x_count={self.candidate_value(right):.6g}"
			)
			return None
		left = 0
		while right - left > 1:
			mid = (left + right) // 2
			if self.valid_g(mid)['valid']:
				right = mid
			else:
				left = mid
		self.vprint(f"[ValidG-search] x_G pos={right}, x_G={self.candidate_value(right):.6g}")
		return right

	def initialize_g_anchor(self):
		if self.setup_error is not None:
			return False
		self.x_g_pos = self.first_valid_g_pos()
		if self.x_g_pos is None:
			return False
		self.g_anchor_diag = self.valid_g(self.x_g_pos)
		self.G_hat_anchor = self.g_anchor_diag['G_hat']
		self.vprint(
			f"[ValidG-anchor] using x_G={self.g_anchor_diag['x0']:.6g} "
			"for the matched bootstrap component"
		)
		return True

	def shape_r(self, alpha, pos, label="ShapeR"):
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in self.shape_cache:
			diag = self.shape_cache[key]
			self.vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, x0={float(diag['x0']):.6g}, "
				f"tails=({int(diag['mix_tail_count'])},{int(diag['alt_tail_count'])}), "
				f"T={float(diag['shape_gap']):.3g}/{float(diag['shape_tol']):.3g}, "
				f"ShapeR={_ok_text(diag['shape_ok'])}"
			)
			return diag
		self.cutoff_checks += 1
		self.shape_r_checks += 1
		x0 = self.candidate_value(pos)
		den = max(1e-12, 1.0 - alpha)
		R_obs = (self.Fn - alpha*self.Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_lcm = _concave_projection_on_interval(self.grid_x, R_obs, x0, self.x1)
		H_hat = _as_valid_cdf(H_lcm)
		T_obs = _weighted_lcm_gap_from_fit(
			self.grid_x, R_obs, H_lcm, x0, self.x1,
			gamma=self.shape_weight_gamma,
		)
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*self.G_hat_anchor)
		boot = self.confirm_boot
		Tstar = np.empty(boot.B, dtype=float)
		cdf_range_star = np.empty(boot.B, dtype=float)
		cdf_mon_star = np.empty(boot.B, dtype=float)
		for b in range(boot.B):
			x_mix_b = _sample_from_cdf(self.grid_x, F_hat, self.n, self.rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(self.grid_x, self.G_hat_anchor, self.m, self.rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, self.grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, self.grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			H_b_lcm = _concave_projection_on_interval(self.grid_x, R_b, x0, self.x1)
			Tstar[b] = _weighted_lcm_gap_from_fit(
				self.grid_x, R_b, H_b_lcm, x0, self.x1,
				gamma=self.shape_weight_gamma,
			)
			cdf_b = _cdf_violation_stats(R_b)
			cdf_range_star[b] = cdf_b['range']
			cdf_mon_star[b] = cdf_b['mon']
		shape_tol = float(np.quantile(Tstar, boot.q, method='higher'))
		shape_ok = bool(T_obs <= shape_tol)
		mix_tail, alt_tail = self.tail_counts_at_pos(pos)
		diag = {
			'alpha': alpha,
			'pos': pos,
			'x0': x0,
			'shape_ok': shape_ok,
			'shape_gap': float(T_obs),
			'shape_tol': shape_tol,
			'shape_statistic': 'weighted_ise',
			'shape_weight_gamma': float(self.shape_weight_gamma),
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_mon_violation': float(cdf_obs['mon']),
			'H_hat': H_hat,
			'G_hat': self.G_hat_anchor,
			'R_obs': R_obs,
			'cdf_bootstrap_range_star': cdf_range_star,
			'cdf_bootstrap_mon_star': cdf_mon_star,
			'cdf_bootstrap_B': int(boot.B),
			'mix_tail_count': mix_tail,
			'alt_tail_count': alt_tail,
		}
		self.shape_cache[key] = diag
		self.vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"tails=({mix_tail},{alt_tail}), T={T_obs:.3g}/{shape_tol:.3g}, "
			f"ShapeR={_ok_text(shape_ok)}"
		)
		return diag

	def flat_reject(self, alpha, pos, shape_diag, label="FlatReject", allow_shape_fail=False):
		alpha = float(alpha)
		pos = int(pos)
		key = (alpha, pos)
		if key in self.flat_cache:
			flat_diag = self.flat_cache[key]
			shape_diag.update(flat_diag)
			self.vprint(
				f"[{label} cached] alpha={alpha:.6f}, pos={pos}, x0={float(shape_diag['x0']):.6g}, "
				f"U={float(flat_diag['flat_stat']):.3g}/{float(flat_diag['flat_tol']):.3g}, "
				f"FlatReject={_ok_text(flat_diag['flat_reject'])}"
				f"{', ShapeR=NO directional' if flat_diag.get('flat_computed_with_shape_fail', False) else ''}"
			)
			return flat_diag
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not allow_shape_fail:
			flat_diag = {
				'flat_available': False,
				'flat_reject': False,
				'flat_stat': np.nan,
				'flat_tol': np.nan,
				'flat_delta': float(self.confirm_boot.flat_delta),
				'flat_bootstrap_quantile': float(self.confirm_boot.q_flat),
				'flat_reason': 'shape_failed',
			}
			shape_diag.update(flat_diag)
			return flat_diag
		self.cutoff_checks += 1
		self.flat_tail_checks += 1
		x0 = self.candidate_value(pos)
		den = max(1e-12, 1.0 - alpha)
		U_obs, H_flat = _flat_tail_stat(self.grid_x, shape_diag['H_hat'], x0, self.x1)
		H_flat = _as_valid_cdf(H_flat)
		F_flat = _as_valid_cdf((1.0 - alpha)*H_flat + alpha*self.G_hat_anchor)
		boot = self.confirm_boot
		Ustar = np.empty(boot.B, dtype=float)
		for b in range(boot.B):
			x_mix_b = _sample_from_cdf(self.grid_x, F_flat, self.n, self.rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(self.grid_x, self.G_hat_anchor, self.m, self.rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, self.grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, self.grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			H_b = _as_valid_cdf(_concave_projection_on_interval(self.grid_x, R_b, x0, self.x1))
			Ustar[b], _ = _flat_tail_stat(self.grid_x, H_b, x0, self.x1)
		flat_tol = float(np.quantile(Ustar, boot.q_flat, method='higher'))
		flat_reject = bool(U_obs > flat_tol)
		flat_diag = {
			'flat_available': True,
			'flat_reject': flat_reject,
			'flat_stat': float(U_obs),
			'flat_tol': flat_tol,
			'flat_delta': float(boot.flat_delta),
			'flat_bootstrap_quantile': float(boot.q_flat),
			'flat_reason': 'computed',
			'flat_computed_with_shape_fail': bool(not shape_ok),
			'H_flat': H_flat,
		}
		shape_diag.update(flat_diag)
		self.flat_cache[key] = flat_diag
		self.vprint(
			f"[{label} #{self.flat_tail_checks}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"U={U_obs:.3g}/{flat_tol:.3g}, FlatReject={_ok_text(flat_reject)}"
			f"{', ShapeR=NO directional' if not shape_ok else ''}"
		)
		return flat_diag

	def cutoff_selection_ok(self, alpha, pos, shape_diag, label="cutoff"):
		if not shape_diag.get('shape_ok', False):
			return False
		if not self.require_flat_tail_reject:
			return True
		flat_diag = self.flat_reject(alpha, pos, shape_diag, label=label)
		return bool(flat_diag['flat_reject'])

	def cutoff_state(self, alpha, pos, shape_diag, label="cutoff"):
		shape_ok = bool(shape_diag.get('shape_ok', False))
		if not shape_ok and not self.require_flat_tail_reject:
			return 'shape_fail'
		if not shape_ok and self.flat_bisect_mode != "guarded":
			return 'shape_fail'
		if not self.require_flat_tail_reject:
			return 'pass'
		flat_diag = self.flat_reject(
			alpha, pos, shape_diag, label=label,
			allow_shape_fail=(self.flat_bisect_mode == "guarded")
		)
		flat_ok = bool(flat_diag['flat_reject'])
		if shape_ok and flat_ok:
			return 'pass'
		if not shape_ok and flat_ok:
			return 'shape_fail_flat_ok'
		if not shape_ok:
			return 'shape_fail_flat_fail'
		return 'flat_fail'

	def derive_leftmost_cutoff(self, alpha, label="derive"):
		start_pos = int(self.x_g_pos)
		if start_pos > self.x_count_pos:
			return False, None, None, {'status': 'start_cutoff_exceeds_count_cutoff'}
		start_diag = self.shape_r(alpha, start_pos, label=f"{label}-xG")
		start_state = self.cutoff_state(alpha, start_pos, start_diag, label=f"{label}-flat-xG")
		if start_state == 'pass':
			return True, start_pos, start_diag, {
				'status': 'x_G_passes', 'stop': 'leftmost',
				'left_pos': start_pos, 'right_pos': start_pos, 'width': 0.0,
			}
		if start_state == 'flat_fail' and self.flat_bisect_mode != "guarded":
			return False, None, start_diag, {
				'status': 'x_G_flat_fails', 'stop': None,
				'left_pos': start_pos, 'right_pos': start_pos, 'width': 0.0,
			}
		if start_pos == self.x_count_pos:
			return False, None, start_diag, {
				'status': 'single_cutoff_fails', 'stop': None,
				'left_pos': start_pos, 'right_pos': start_pos, 'width': 0.0,
			}
		right_diag = self.shape_r(alpha, self.x_count_pos, label=f"{label}-xcount")
		right_state = self.cutoff_state(alpha, self.x_count_pos, right_diag, label=f"{label}-flat-xcount")
		if right_state in {'shape_fail', 'shape_fail_flat_ok'}:
			return False, None, right_diag, {
				'status': 'x_count_shape_fails', 'stop': None,
				'left_pos': start_pos, 'right_pos': self.x_count_pos,
				'width': self.cutoff_width(start_pos, self.x_count_pos),
			}
		left = start_pos
		right = self.x_count_pos
		best_pos = self.x_count_pos if right_state == 'pass' else None
		best_diag = right_diag if right_state == 'pass' else None
		while right - left > 1 and self.cutoff_width(left, right) > self.x0_tol:
			mid = (left + right) // 2
			mid_diag = self.shape_r(alpha, mid, label=f"{label}-bisect")
			mid_state = self.cutoff_state(alpha, mid, mid_diag, label=f"{label}-flat-bisect")
			if mid_state == 'shape_fail':
				if self.require_flat_tail_reject and self.flat_bisect_mode == "guarded":
					right = mid
					right_diag = mid_diag
					self.vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-fail; move hi")
				else:
					left = mid
					self.vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail; move lo")
			elif mid_state == 'shape_fail_flat_ok':
				left = mid
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} shape-fail/flat-ok; move lo")
			elif mid_state in {'flat_fail', 'shape_fail_flat_fail'}:
				right = mid
				right_diag = mid_diag
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; move hi")
			else:
				right = mid
				right_diag = mid_diag
				best_pos = mid
				best_diag = mid_diag
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} passes; move hi")
		stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		if best_diag is None:
			return False, None, right_diag, {
				'status': 'no_shape_flat_overlap' if self.require_flat_tail_reject else 'no_shape_passing_cutoff',
				'stop': stop_reason, 'left_pos': int(left), 'right_pos': int(right),
				'width': self.cutoff_width(left, right),
			}
		if best_diag['pos'] != best_pos:
			best_diag = self.shape_r(alpha, best_pos, label=f"{label}-best")
			self.cutoff_selection_ok(alpha, best_pos, best_diag, label=f"{label}-flat-best")
		return True, int(best_pos), best_diag, {
			'status': 'found', 'stop': stop_reason,
			'left_pos': int(left), 'right_pos': int(best_pos),
			'width': self.cutoff_width(left, int(best_pos)),
		}

	def derive_rightmost_cutoff(self, alpha, label="derive-rightmost"):
		def rightmost_state(pos, state_label):
			diag = self.shape_r(alpha, pos, label=state_label)
			shape_ok = bool(diag.get('shape_ok', False))
			if not self.require_flat_tail_reject:
				return ('pass' if shape_ok else 'shape_fail'), diag
			flat_diag = self.flat_reject(
				alpha, pos, diag,
				label=state_label.replace("-bisect", "-flat-bisect").replace("-xG", "-flat-xG").replace("-xcount", "-flat-xcount"),
				allow_shape_fail=True,
			)
			flat_ok = bool(flat_diag.get('flat_reject', False))
			if shape_ok and flat_ok:
				return 'pass', diag
			if (not shape_ok) and flat_ok:
				return 'shape_fail_flat_ok', diag
			if not shape_ok:
				return 'shape_fail_flat_fail', diag
			return 'flat_fail', diag

		start_pos = int(self.x_g_pos)
		if start_pos > self.x_count_pos:
			return False, None, None, {'status': 'start_cutoff_exceeds_count_cutoff'}
		start_diag = self.shape_r(alpha, start_pos, label=f"{label}-xG")
		if start_diag.get('shape_ok', False):
			start_state, start_diag = rightmost_state(start_pos, f"{label}-xG-selection")
		else:
			start_state = 'shape_fail'
		right_diag = self.shape_r(alpha, self.x_count_pos, label=f"{label}-xcount")
		right_shape_ok = bool(right_diag.get('shape_ok', False))
		if (not start_diag.get('shape_ok', False)) and (not right_shape_ok):
			return False, None, right_diag, {
				'status': 'endpoint_shape_fails',
				'start_state': start_state,
				'right_shape_ok': bool(right_shape_ok),
				'stop': None,
				'left_pos': int(start_pos),
				'right_pos': int(self.x_count_pos),
				'width': self.cutoff_width(int(start_pos), int(self.x_count_pos)),
			}
		if right_shape_ok:
			right_state, right_diag = rightmost_state(self.x_count_pos, f"{label}-xcount-selection")
			if right_state == 'pass':
				return True, int(self.x_count_pos), right_diag, {
					'status': 'x_count_passes',
					'stop': 'rightmost',
					'left_pos': int(start_pos),
					'right_pos': int(self.x_count_pos),
					'width': self.cutoff_width(int(start_pos), int(self.x_count_pos)),
				}
		left = int(start_pos)
		right = int(self.x_count_pos)
		best_pos = int(start_pos) if start_state == 'pass' else None
		best_diag = start_diag if start_state == 'pass' else None
		stop_reason = None
		ambiguous_pos = None
		while right - left > 1 and self.cutoff_width(left, right) > self.x0_tol:
			mid = (left + right) // 2
			mid_state, mid_diag = rightmost_state(mid, f"{label}-bisect")
			if mid_state == 'pass':
				left = mid
				best_pos = mid
				best_diag = mid_diag
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} passes; move lo")
			elif mid_state in {'shape_fail', 'shape_fail_flat_ok'}:
				left = mid
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; move lo")
			elif mid_state == 'shape_fail_flat_fail':
				ambiguous_pos = mid
				stop_reason = 'shape_flat_double_fail'
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; stop ambiguous")
				break
			else:
				right = mid
				self.vprint(f"[{label}-cutoff-bisect] pos={mid} {mid_state}; move hi")
		if stop_reason is None:
			stop_reason = 'adjacent' if right - left <= 1 else 'x0_tol'
		right_bound_pos = int(ambiguous_pos) if ambiguous_pos is not None else int(right)
		if best_diag is None:
			return False, None, right_diag, {
				'status': 'shape_flat_double_fail_no_cutoff' if ambiguous_pos is not None else 'no_rightmost_passing_cutoff',
				'stop': stop_reason,
				'left_pos': int(left),
				'right_pos': right_bound_pos,
				'ambiguous_pos': int(ambiguous_pos) if ambiguous_pos is not None else None,
				'width': self.cutoff_width(int(left), right_bound_pos),
			}
		return True, int(best_pos), best_diag, {
			'status': 'rightmost_found_before_double_fail' if ambiguous_pos is not None else 'rightmost_found',
			'stop': stop_reason,
			'left_pos': int(start_pos),
			'right_pos': int(best_pos),
			'right_bound_pos': right_bound_pos,
			'ambiguous_pos': int(ambiguous_pos) if ambiguous_pos is not None else None,
			'width': self.cutoff_width(int(best_pos), right_bound_pos) if right_bound_pos >= best_pos else 0.0,
		}

	def compact_shape(self, diag):
		if diag is None:
			return None
		out = {
			'alpha': float(diag['alpha']),
			'pos': int(diag['pos']),
			'x0': float(diag['x0']),
			'shape_ok': bool(diag['shape_ok']),
			'cutoff_selection_ok': bool(
				diag.get('shape_ok', False) and
				(not self.require_flat_tail_reject or bool(diag.get('flat_reject', False)))
			),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'shape_statistic': diag.get('shape_statistic', 'weighted_ise'),
			'shape_weight_gamma': float(diag.get('shape_weight_gamma', self.shape_weight_gamma)),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'mix_tail_count': int(diag['mix_tail_count']),
			'alt_tail_count': int(diag['alt_tail_count']),
		}
		if self.require_flat_tail_reject or 'flat_reject' in diag:
			out.update({
				'flat_available': bool(diag.get('flat_available', False)),
				'flat_reject': None if diag.get('flat_reject') is None else bool(diag.get('flat_reject')),
				'flat_stat': float(diag.get('flat_stat', np.nan)),
				'flat_tol': float(diag.get('flat_tol', np.nan)),
				'flat_delta': float(diag.get('flat_delta', self.confirm_boot.flat_delta)),
				'flat_bootstrap_quantile': float(diag.get('flat_bootstrap_quantile', self.confirm_boot.q_flat)),
				'flat_reason': diag.get('flat_reason'),
				'flat_confirm_B': None if diag.get('flat_confirm_B') is None else int(diag.get('flat_confirm_B')),
				'flat_computed_with_shape_fail': bool(diag.get('flat_computed_with_shape_fail', False)),
			})
		return out

	def compact_cdf(self, diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'cdf_ok': bool(diag['cdf_ok']),
			'cdf_available': bool(diag.get('cdf_available', True)),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_B': int(diag.get('cdf_B', self.confirm_boot.cdf_B)),
			'cdf_bootstrap_reused': int(diag.get('cdf_bootstrap_reused', 0)),
			'cdf_bootstrap_drawn': int(diag.get('cdf_bootstrap_drawn', self.confirm_boot.cdf_B)),
			'cdf_delta': float(diag.get('cdf_delta', self.confirm_boot.cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', self.confirm_boot.q_cdf)),
			'alpha_ref': None if diag.get('alpha_ref') is None else float(diag['alpha_ref']),
			'x_ref': None if diag.get('x_ref') is None else float(diag['x_ref']),
			'ref_distance': None if diag.get('ref_distance') is None else float(diag['ref_distance']),
			'weak_ref_calibration': bool(diag.get('weak_ref_calibration', False)),
		}

	def record_shape_ref(self, alpha, pos, diag, source):
		rec = {
			'alpha': float(alpha),
			'pos': int(pos),
			'x0': self.candidate_value(pos),
			'source': source,
			'shape_diag': diag,
		}
		self.shape_refs.append(rec)
		return rec

	def nearest_shape_ref(self, alpha):
		if not self.shape_refs:
			return None
		alpha = float(alpha)
		return min(self.shape_refs, key=lambda rec: abs(float(rec['alpha']) - alpha))

	def evaluate_shape_path(self, alpha, label="shape-alpha", cutoff_strategy="leftmost"):
		alpha = float(alpha)
		self.alpha_checks += 1
		old_pos = self.working_pos
		self.vprint(
			f"[{label} #{self.alpha_checks}] alpha={alpha:.6f}, "
			f"current_x0={'None' if self.working_pos is None else f'{self.candidate_value(self.working_pos):.6g}'}"
		)
		if alpha >= 1.0 - 1e-12:
			return False, old_pos, None, {
				'updated': False, 'old_pos': old_pos, 'new_pos': old_pos,
				'search': {'status': 'alpha_too_close_to_one'},
			}
		if self.working_pos is not None:
			current_diag = self.shape_r(alpha, self.working_pos, label=f"{label}-current")
			if self.cutoff_selection_ok(alpha, self.working_pos, current_diag, label=f"{label}-flat-current"):
				self.record_shape_ref(alpha, self.working_pos, current_diag, source='current')
				return True, self.working_pos, current_diag, {
					'updated': False, 'old_pos': old_pos, 'new_pos': self.working_pos,
					'search': {'status': 'current_cutoff_passes'},
				}
		if cutoff_strategy == "rightmost":
			ok, new_pos, new_diag, search = self.derive_rightmost_cutoff(alpha, label=f"{label}-derive-rightmost")
		else:
			ok, new_pos, new_diag, search = self.derive_leftmost_cutoff(alpha, label=f"{label}-derive")
		if not ok:
			return False, old_pos, new_diag, {
				'updated': False, 'old_pos': old_pos, 'new_pos': old_pos,
				'search': search,
			}
		self.working_pos = int(new_pos)
		self.record_shape_ref(alpha, self.working_pos, new_diag, source='derived')
		update = {
			'alpha': float(alpha),
			'old_pos': None if old_pos is None else int(old_pos),
			'old_x0': None if old_pos is None else self.candidate_value(old_pos),
			'new_pos': int(self.working_pos),
			'new_x0': self.candidate_value(self.working_pos),
			'search': search,
			'shape_diag': self.compact_shape(new_diag),
		}
		if old_pos != self.working_pos:
			self.cutoff_update_history.append(update)
		return True, self.working_pos, new_diag, {
			'updated': old_pos != self.working_pos,
			'old_pos': old_pos,
			'new_pos': self.working_pos,
			'search': search,
		}

	def cdf_at(self, alpha, pos, alpha_ref=None, label="CDF"):
		alpha = float(alpha)
		pos = int(pos)
		alpha_ref = alpha if alpha_ref is None else float(alpha_ref)
		key = (alpha, pos, alpha_ref)
		if key in self.cdf_cache:
			diag = self.cdf_cache[key]
			self.vprint(
				f"[{label} cached] alpha={alpha:.6f}, x_ref={diag.get('x_ref'):.6g}, "
				f"CDFOK={_ok_text(diag.get('cdf_ok', False))}"
			)
			return diag
		self.cdf_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha, 'pos': pos, 'x_ref': self.candidate_value(pos),
				'alpha_ref': alpha_ref, 'ref_distance': abs(alpha - alpha_ref),
				'weak_ref_calibration': abs(alpha - alpha_ref) > self.cdf_ref_max_dist,
				'cdf_available': True, 'cdf_ok': False,
				'cdf_range_violation': np.inf, 'cdf_low_violation': np.inf,
				'cdf_high_violation': np.inf, 'cdf_range_tol': np.nan,
				'cdf_mon_violation': np.inf, 'cdf_mon_tol': np.nan,
				'H_hat': None, 'R_obs': None,
			}
			self.cdf_cache[key] = diag
			return diag
		x_ref = self.candidate_value(pos)
		den = max(1e-12, 1.0 - alpha)
		R_obs = (self.Fn - alpha*self.Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		shape_diag = self.shape_cache.get((alpha, pos))
		H_hat = (
			shape_diag['H_hat'] if shape_diag is not None
			else _as_valid_cdf(_concave_projection_on_interval(self.grid_x, R_obs, x_ref, self.x1))
		)
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*self.G_hat_anchor)
		boot = self.confirm_boot
		range_star = np.empty(boot.cdf_B, dtype=float)
		mon_star = np.empty(boot.cdf_B, dtype=float)
		n_reused = 0
		if shape_diag is not None:
			shape_range_star = shape_diag.get('cdf_bootstrap_range_star')
			shape_mon_star = shape_diag.get('cdf_bootstrap_mon_star')
			if shape_range_star is not None and shape_mon_star is not None:
				n_reused = min(
					boot.cdf_B,
					len(shape_range_star),
					len(shape_mon_star)
				)
				if n_reused > 0:
					range_star[:n_reused] = shape_range_star[:n_reused]
					mon_star[:n_reused] = shape_mon_star[:n_reused]
		for b in range(n_reused, boot.cdf_B):
			x_mix_b = _sample_from_cdf(self.grid_x, F_hat, self.n, self.rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(self.grid_x, self.G_hat_anchor, self.m, self.rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, self.grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, self.grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
		range_tol = float(np.quantile(range_star, boot.q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, boot.q_cdf, method='higher'))
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		ref_distance = abs(alpha - alpha_ref)
		diag = {
			'alpha': alpha, 'pos': pos, 'x_ref': x_ref,
			'alpha_ref': alpha_ref, 'ref_distance': float(ref_distance),
			'weak_ref_calibration': bool(ref_distance > self.cdf_ref_max_dist),
			'cdf_available': True, 'cdf_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_B': int(boot.cdf_B),
			'cdf_bootstrap_reused': int(n_reused),
			'cdf_bootstrap_drawn': int(boot.cdf_B - n_reused),
			'cdf_delta': float(boot.cdf_delta),
			'cdf_bootstrap_quantile': float(boot.q_cdf),
			'H_hat': H_hat,
			'R_obs': R_obs,
			'mix_tail_count': int(np.sum(self.x_mix >= x_ref)),
			'alt_tail_count': int(np.sum(self.x_alt >= x_ref)),
		}
		self.cdf_cache[key] = diag
		self.vprint(
			f"[{label} #{self.cdf_checks}] alpha={alpha:.6f}, x_ref={x_ref:.6g} "
			f"(alpha_ref={alpha_ref:.6f}, dist={ref_distance:.3g}), "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, CDFOK={_ok_text(cdf_ok)}"
		)
		return diag

	def evaluate_cdf(self, alpha, label="cdf-alpha", pos=None, alpha_ref=None):
		if pos is None:
			ref = self.nearest_shape_ref(alpha)
			if ref is None:
				alpha = float(alpha)
				return {
					'alpha': alpha, 'pos': None, 'x_ref': None,
					'alpha_ref': None, 'ref_distance': None,
					'weak_ref_calibration': True, 'cdf_available': False,
					'cdf_ok': False, 'cdf_range_violation': np.nan,
					'cdf_low_violation': np.nan, 'cdf_high_violation': np.nan,
					'cdf_range_tol': np.nan, 'cdf_mon_violation': np.nan,
					'cdf_mon_tol': np.nan, 'H_hat': None, 'R_obs': None,
				}
			pos = ref['pos']
			alpha_ref = ref['alpha']
		elif alpha_ref is None:
			alpha_ref = alpha
		return self.cdf_at(alpha, pos, alpha_ref, label=label)


def _lcm_gap_from_fit(x, y, fit, a, b):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	fit = np.asarray(fit, dtype=float)
	mask = (x >= min(a, b)) & (x <= max(a, b))
	if int(np.sum(mask)) < 2:
		return 0.0
	return _ise_on_grid(x[mask], fit[mask] - y[mask])


def _weighted_lcm_gap_from_fit(x, y, fit, a, b, gamma=2.0):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	fit = np.asarray(fit, dtype=float)
	mask = (x >= min(a, b)) & (x <= max(a, b))
	if int(np.sum(mask)) < 2:
		return 0.0
	xs = x[mask]
	diff = fit[mask] - y[mask]
	x_left = float(xs[0])
	x_right = float(xs[-1])
	width = x_right - x_left
	if width <= 0.0:
		return 0.0
	gamma = float(gamma)
	t = np.clip((xs - x_left) / width, 0.0, 1.0)
	weights = np.power(t, gamma)
	weights = weights / max(float(np.mean(weights)), 1e-12)
	dx = np.diff(xs)
	return float(np.sum(
		0.5 * (
			weights[:-1] * diff[:-1]**2
			+ weights[1:] * diff[1:]**2
		) * dx
	))


def _empirical_quantile_indices(n_items, n_keep):
	"""Return indices closest to equally spaced empirical quantiles."""
	n_items = int(n_items)
	n_keep = int(n_keep)
	if n_items <= 0:
		return np.array([], dtype=int)
	if n_keep >= n_items:
		return np.arange(n_items, dtype=int)
	if n_keep <= 1:
		return np.array([0], dtype=int)
	ranks = np.arange(n_items, dtype=float)
	idx = np.quantile(ranks, np.linspace(0.0, 1.0, num=n_keep), method='nearest')
	return np.unique(idx.astype(int))


def _slope_violation_stat(x, y, a, b, n_blocks=None):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	mask = (x >= min(a, b)) & (x <= max(a, b))
	xs = x[mask]
	ys = y[mask]
	if len(xs) < 3:
		return 0.0
	if n_blocks is not None:
		n_blocks = int(n_blocks)
		if n_blocks < 2:
			raise ValueError("slope_blocks must be at least 2 when supplied.")
		max_points = n_blocks + 1
		if len(xs) > max_points:
			idx = _empirical_quantile_indices(len(xs), max_points)
			xs = xs[idx]
			ys = ys[idx]
			if len(xs) < 3:
				return 0.0
	dx = np.diff(xs)
	keep = dx > 0
	if int(np.sum(keep)) < 2:
		return 0.0
	slopes = np.diff(ys)[keep] / dx[keep]
	left = xs[:-1][keep]
	right = xs[1:][keep]
	mids = 0.5 * (left + right)
	viol = np.maximum(0.0, np.diff(slopes))
	if len(viol) == 0:
		return 0.0
	weights = np.diff(mids)
	weights = np.maximum(weights, 0.0)
	return float(np.sum(weights * viol**2))


def _interp_on_grid(x, y, value):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	value = float(value)
	if value <= x[0]:
		return float(y[0])
	if value >= x[-1]:
		return float(y[-1])
	return float(np.interp(value, x, y))


def _rightmost_crossing_x(x, y, upper_x, target):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	mask = x <= float(upper_x)
	xs = x[mask]
	ys = y[mask]
	if len(xs) == 0:
		return np.nan
	below = np.where(ys <= float(target))[0]
	if len(below) == 0:
		return np.nan
	i = int(below[-1])
	if i >= len(xs) - 1:
		return float(xs[i])
	x0, x1 = float(xs[i]), float(xs[i + 1])
	y0, y1 = float(ys[i]), float(ys[i + 1])
	if y1 == y0:
		return x0
	t = (float(target) - y0) / (y1 - y0)
	t = min(1.0, max(0.0, t))
	return float(x0 + t*(x1 - x0))


def _raw_tail_guard_diag(x, R, x0, x1=0.0):
	"""Deterministic raw-residual tail guard based on adjacent chord slopes."""
	x = np.asarray(x, dtype=float)
	R = np.asarray(R, dtype=float)
	x0 = float(x0)
	x1 = float(x1)
	x_left = max(float(x[0]), 2.0*x0 - x1)
	R_right = _interp_on_grid(x, R, x1)
	R_x0 = _interp_on_grid(x, R, x0)
	R_left = _interp_on_grid(x, R, x_left)
	right_mass = R_right - R_x0
	left_mass = R_x0 - R_left
	right_width = x1 - x0
	left_width = x0 - x_left
	left_slope = left_mass / left_width if left_width > 0.0 else np.nan
	right_slope = right_mass / right_width if right_width > 0.0 else np.nan
	if right_slope > 0.0 and np.isfinite(left_slope):
		slope_ratio = left_slope / right_slope
	else:
		slope_ratio = np.nan
	equal_mass_target = R_x0 - right_mass
	equal_mass_x = _rightmost_crossing_x(x, R, x0, equal_mass_target)
	ok = bool(
		np.isfinite(left_slope)
		and np.isfinite(right_slope)
		and right_slope > 0.0
		and left_slope >= right_slope
	)
	return {
		'tail_guard_ok': ok,
		'tail_guard_x_left': float(x_left),
		'tail_guard_R_right': float(R_right),
		'tail_guard_R_x0': float(R_x0),
		'tail_guard_R_left': float(R_left),
		'tail_guard_left_mass': float(left_mass),
		'tail_guard_right_mass': float(right_mass),
		'tail_guard_left_slope': float(left_slope),
		'tail_guard_right_slope': float(right_slope),
		'tail_guard_slope_ratio': float(slope_ratio),
		'tail_guard_equal_mass_target': float(equal_mass_target),
		'tail_guard_equal_mass_x': float(equal_mass_x),
		'tail_guard_equal_mass_ok': bool(np.isfinite(equal_mass_x) and equal_mass_x >= x_left),
	}


def _normalize_slope_blocks(slope_blocks):
	if slope_blocks is None:
		return (None,)
	if isinstance(slope_blocks, (str, bytes)):
		raise TypeError("slope_blocks must be None, an integer, or an iterable of integers.")
	try:
		values = tuple(slope_blocks)
	except TypeError:
		values = (slope_blocks,)
	if len(values) == 0:
		raise ValueError("slope_blocks iterable must contain at least one value.")
	out = []
	seen = set()
	for value in values:
		if value is None:
			block = None
			key = None
		else:
			block = int(value)
			if block < 2:
				raise ValueError("slope_blocks values must be at least 2 when supplied.")
			key = block
		if key not in seen:
			out.append(block)
			seen.add(key)
	return tuple(out)


def _slope_blocks_info(slope_blocks_values):
	if len(slope_blocks_values) == 1:
		return slope_blocks_values[0]
	return [None if block is None else int(block) for block in slope_blocks_values]


def _slope_blocks_text(slope_blocks_values):
	info = _slope_blocks_info(slope_blocks_values)
	return "all" if info is None else repr(info)


def alpha_minimize_given_x0_revisited(
	x_mix,
	x_alt,
	x0,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	cdf_B=None,
	gap_tol=None,
	near_miss_probes=5,
	max_checks=64,
	random_state=None,
	grid=None,
	max_grid_points=5000,
	verbose=False,
):
	"""
	Fixed-x0 separated shape/CDF alpha search with independent cdf_B.

	Returns
	-------
	alpha_hat : float
	H_hat     : np.ndarray or None
	G_hat     : np.ndarray
		Convex PAVA/LCM fitted matched CDF on the supplied fixed-x0 interval.
	grid_x    : np.ndarray
	info      : dict
		Diagnostics including alpha_shape, alpha_cdf, gap status, and the
		final shape/CDF statistics.
	"""
	rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
	x_mix = np.sort(np.asarray(x_mix, dtype=float))
	x_alt = np.sort(np.asarray(x_alt, dtype=float))
	if len(x_mix) == 0 or len(x_alt) == 0:
		raise ValueError("x_mix and x_alt must both be nonempty.")
	x0 = float(x0)
	if not np.isfinite(x0) or x0 >= 0.0:
		raise ValueError("x0 must be finite and strictly less than 0.")
	if alpha_tol <= 0:
		raise ValueError("alpha_tol must be positive.")
	n_alpha_probes = int(n_alpha_probes)
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	max_checks = int(max_checks)
	if max_checks < 2*n_alpha_probes:
		raise ValueError("max_checks must be at least 2*n_alpha_probes for separated shape/CDF probes.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	alpha_left, alpha_right = _check_alpha_bounds(alpha_bounds)
	boot = _BootstrapPlan.make(
		"pava3.alpha_minimize_given_x0_revisited",
		B, delta, cdf_delta=cdf_delta, cdf_B=cdf_B,
		require_flat_tail_reject=False
	)

	x1 = 0.0
	if grid is None:
		grid_x = _build_grid(x_mix, x_alt, x0, x1, max_points=max_grid_points)
	else:
		grid_x = np.sort(np.asarray(grid, dtype=float))
		grid_x = grid_x[np.isfinite(grid_x)]
		if len(grid_x) == 0:
			raise ValueError("grid must contain at least one finite value.")
	grid_x = np.unique(np.concatenate([grid_x, np.array([x0, x1], dtype=float)]))
	grid_x = np.unique(grid_x[grid_x <= x1])
	if len(grid_x) == 0:
		raise ValueError("grid has no points at or below 0.")
	if grid_x[0] > x0:
		grid_x = np.concatenate([[x0], grid_x])
	if grid_x[-1] < x1:
		grid_x = np.concatenate([grid_x, [x1]])

	Fn = _ecdf_on_grid(x_mix, grid_x)
	Gm = _ecdf_on_grid(x_alt, grid_x)
	n, m = len(x_mix), len(x_alt)
	G_hat = _as_valid_cdf(_convex_projection_on_interval(grid_x, Gm, x0, x1))
	G_diff = Gm - G_hat
	G_convex_gap = _convex_gap_T(grid_x, Gm, x0, x1)
	G_distance_ks = float(np.max(np.abs(G_diff))) if len(G_diff) else 0.0
	G_distance_ise = _ise_on_grid(grid_x, G_diff)

	def vprint(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	vprint(
		f"[setup-given-x0-2] n={n}, m={m}, grid={len(grid_x)}, x0={x0:.6g}, "
		f"boot_seed={_random_state_label(random_state)}, "
		f"alpha_bounds=({alpha_left:.6f}, {alpha_right:.6f}), "
		f"alpha_tol={float(alpha_tol):.3g}, gap_tol={float(gap_tol):.3g}, "
		f"B={boot.B}, delta={boot.delta:.3g}, q={boot.q:.6f}, "
		f"cdf_B={boot.cdf_B}, cdf_delta={boot.cdf_delta:.3g}, q_cdf={boot.q_cdf:.6f}"
	)
	vprint(
		f"[G-fit-given-x0-2] convex_gap={G_convex_gap:.3g}, "
		f"distance_ks={G_distance_ks:.3g}, distance_ise={G_distance_ise:.3g}"
	)

	shape_checks = 0
	cdf_checks = 0
	base_cache = {}
	shape_cache = {}
	cdf_cache = {}
	full_cache = {}

	def total_alpha_checks():
		return int(shape_checks + cdf_checks)

	def fit_base(alpha):
		alpha = float(alpha)
		if alpha in base_cache:
			return base_cache[alpha]
		den = max(1e-12, 1.0 - alpha)
		R_obs = (Fn - alpha*Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		H_hat = _as_valid_cdf(_concave_projection_on_interval(grid_x, R_obs, x0, x1))
		base = {
			'alpha': alpha,
			'den': den,
			'R_obs': R_obs,
			'H_hat': H_hat,
			'cdf_obs': cdf_obs,
			'shape_gap': _gap_T(grid_x, R_obs, x0, x1),
		}
		base_cache[alpha] = base
		return base

	def bootstrap_sample(alpha, H_hat):
		F_hat = _as_valid_cdf((1.0 - float(alpha))*H_hat + float(alpha)*G_hat)
		x_mix_b = _sample_from_cdf(grid_x, F_hat, n, rng)
		x_mix_b.sort(kind='mergesort')
		x_alt_b = _sample_from_cdf(grid_x, G_hat, m, rng)
		x_alt_b.sort(kind='mergesort')
		return _ecdf_on_grid(x_mix_b, grid_x), _ecdf_on_grid(x_alt_b, grid_x)

	def shape_at(alpha, label="shape"):
		nonlocal shape_checks
		alpha = float(alpha)
		if alpha in shape_cache:
			return shape_cache[alpha]
		shape_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha, 'shape_ok': False, 'shape_gap': np.inf,
				'shape_tol': np.nan, 'cdf_range_violation': np.inf,
				'cdf_low_violation': np.inf, 'cdf_high_violation': np.inf,
				'cdf_mon_violation': np.inf, 'R_obs': None, 'H_hat': None,
				'reason': 'alpha_too_close_to_one',
			}
			shape_cache[alpha] = diag
			vprint(f"[{label} #{shape_checks}] alpha={alpha:.6f}: alpha too close to 1, ShapeOK=NO")
			return diag
		base = fit_base(alpha)
		shape_star = np.empty(boot.B, dtype=float)
		for b in range(boot.B):
			Fn_b, Gm_b = bootstrap_sample(alpha, base['H_hat'])
			R_b = (Fn_b - alpha*Gm_b) / base['den']
			shape_star[b] = _gap_T(grid_x, R_b, x0, x1)
		shape_tol = float(np.quantile(shape_star, boot.q, method='higher'))
		shape_ok = bool(base['shape_gap'] <= shape_tol)
		cdf_obs = base['cdf_obs']
		diag = {
			'alpha': alpha,
			'shape_ok': shape_ok,
			'shape_gap': float(base['shape_gap']),
			'shape_tol': shape_tol,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_mon_violation': float(cdf_obs['mon']),
			'R_obs': base['R_obs'],
			'H_hat': base['H_hat'],
		}
		shape_cache[alpha] = diag
		vprint(
			f"[{label} #{shape_checks}] alpha={alpha:.6f}, "
			f"shape={base['shape_gap']:.3g}/{shape_tol:.3g}, ShapeOK={_ok_text(shape_ok)}"
		)
		return diag

	def cdf_at(alpha, label="cdf"):
		nonlocal cdf_checks
		alpha = float(alpha)
		if alpha in cdf_cache:
			return cdf_cache[alpha]
		cdf_checks += 1
		if alpha >= 1.0 - 1e-12:
			diag = {
				'alpha': alpha, 'cdf_ok': False,
				'cdf_range_violation': np.inf, 'cdf_low_violation': np.inf,
				'cdf_high_violation': np.inf, 'cdf_range_tol': np.nan,
				'cdf_mon_violation': np.inf, 'cdf_mon_tol': np.nan,
				'R_obs': None, 'H_hat': None,
				'reason': 'alpha_too_close_to_one',
			}
			cdf_cache[alpha] = diag
			vprint(f"[{label} #{cdf_checks}] alpha={alpha:.6f}: alpha too close to 1, CDFOK=NO")
			return diag
		base = fit_base(alpha)
		range_star = np.empty(boot.cdf_B, dtype=float)
		mon_star = np.empty(boot.cdf_B, dtype=float)
		for b in range(boot.cdf_B):
			Fn_b, Gm_b = bootstrap_sample(alpha, base['H_hat'])
			R_b = (Fn_b - alpha*Gm_b) / base['den']
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
		range_tol = float(np.quantile(range_star, boot.q_cdf, method='higher'))
		mon_tol = float(np.quantile(mon_star, boot.q_cdf, method='higher'))
		cdf_obs = base['cdf_obs']
		cdf_ok = bool(cdf_obs['range'] <= range_tol and cdf_obs['mon'] <= mon_tol)
		diag = {
			'alpha': alpha,
			'cdf_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_low_violation': float(cdf_obs['low']),
			'cdf_high_violation': float(cdf_obs['high']),
			'cdf_range_tol': range_tol,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': mon_tol,
			'cdf_B': int(boot.cdf_B),
			'cdf_delta': float(boot.cdf_delta),
			'cdf_bootstrap_quantile': float(boot.q_cdf),
			'R_obs': base['R_obs'],
			'H_hat': base['H_hat'],
		}
		cdf_cache[alpha] = diag
		vprint(
			f"[{label} #{cdf_checks}] alpha={alpha:.6f}, "
			f"range={cdf_obs['range']:.3g}/{range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{mon_tol:.3g}, CDFOK={_ok_text(cdf_ok)}"
		)
		return diag

	def full_diag(alpha, label="full"):
		alpha = float(alpha)
		if alpha in full_cache:
			return full_cache[alpha]
		shape_diag = shape_at(alpha, label=f"{label}-shape")
		cdf_diag = cdf_at(alpha, label=f"{label}-cdf")
		full_ok = bool(shape_diag['shape_ok'] and cdf_diag['cdf_ok'])
		diag = {
			'alpha': alpha,
			'shape_ok': bool(shape_diag['shape_ok']),
			'cdf_ok': bool(cdf_diag['cdf_ok']),
			'full_ok': full_ok,
			'shape_gap': float(shape_diag['shape_gap']),
			'shape_tol': float(shape_diag['shape_tol']),
			'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
			'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
			'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
			'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
			'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
			'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
			'cdf_B': int(cdf_diag.get('cdf_B', boot.cdf_B)),
			'cdf_delta': float(cdf_diag.get('cdf_delta', boot.cdf_delta)),
			'cdf_bootstrap_quantile': float(cdf_diag.get('cdf_bootstrap_quantile', boot.q_cdf)),
			'R_obs': shape_diag.get('R_obs'),
			'H_hat': shape_diag.get('H_hat'),
		}
		full_cache[alpha] = diag
		vprint(f"[{label}] alpha={alpha:.6f}, full={_ok_text(full_ok)}")
		return diag

	def compact_shape(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'shape_ok': bool(diag['shape_ok']),
			'shape_gap': float(diag['shape_gap']),
			'shape_tol': float(diag['shape_tol']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
		}

	def compact_cdf(diag):
		if diag is None:
			return None
		return {
			'alpha': float(diag['alpha']),
			'cdf_ok': bool(diag['cdf_ok']),
			'cdf_range_violation': float(diag['cdf_range_violation']),
			'cdf_low_violation': float(diag['cdf_low_violation']),
			'cdf_high_violation': float(diag['cdf_high_violation']),
			'cdf_range_tol': float(diag['cdf_range_tol']),
			'cdf_mon_violation': float(diag['cdf_mon_violation']),
			'cdf_mon_tol': float(diag['cdf_mon_tol']),
			'cdf_B': int(diag.get('cdf_B', boot.cdf_B)),
			'cdf_delta': float(diag.get('cdf_delta', boot.cdf_delta)),
			'cdf_bootstrap_quantile': float(diag.get('cdf_bootstrap_quantile', boot.q_cdf)),
		}

	def compact_diag(diag):
		if diag is None:
			return None
		out = compact_shape(diag)
		out.update(compact_cdf(diag))
		out['full_ok'] = bool(diag['full_ok'])
		return out

	def make_info(success, reason, final_diag=None, extra=None):
		info = {
			'success': bool(success),
			'reason': reason,
			'mode': 'fixed_x0_separated_bootstrap_pava3',
			'alpha': float(final_diag['alpha']) if final_diag is not None else np.nan,
			'x0': float(x0),
			'alpha_checks': int(total_alpha_checks()),
			'total_alpha_checks': int(total_alpha_checks()),
			'shape_checks': int(shape_checks),
			'cdf_checks': int(cdf_checks),
			'alpha_tol': float(alpha_tol),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
			'max_checks': int(max_checks),
			'n_alpha_probes': int(n_alpha_probes),
			'alpha_bounds': (float(alpha_left), float(alpha_right)),
			'n_grid': int(len(grid_x)),
			'n_mix': int(n),
			'n_alt': int(m),
			'G_convex_gap': float(G_convex_gap),
			'G_distance_ks': float(G_distance_ks),
			'G_distance_ise': float(G_distance_ise),
		}
		info.update(boot.info())
		if final_diag is not None:
			info.update(compact_diag(final_diag))
			info['alpha'] = float(final_diag['alpha'])
			info['residual_lcm_gap'] = float(final_diag['shape_gap'])
			info['residual_lcm_tol'] = float(final_diag['shape_tol'])
		if extra:
			info.update(extra)
		return info

	def refine_shape(fail_diag, pass_diag):
		lo = fail_diag
		hi = pass_diag
		status = 'alpha_tol_reached'
		while (hi['alpha'] - lo['alpha']) > alpha_tol and total_alpha_checks() < max_checks:
			mid = 0.5 * (lo['alpha'] + hi['alpha'])
			mid_diag = shape_at(mid, label="shape-bisect2")
			if mid_diag['shape_ok']:
				hi = mid_diag
			else:
				lo = mid_diag
		if (hi['alpha'] - lo['alpha']) > alpha_tol:
			status = 'max_checks_reached'
		return hi, {'status': status, 'lo': float(lo['alpha']), 'hi': float(hi['alpha']), 'width': float(hi['alpha'] - lo['alpha'])}

	probe_alphas = np.linspace(alpha_left, alpha_right, num=n_alpha_probes)
	shape_probe_table = []
	last_shape_fail = None
	first_shape_pass = None
	vprint("[shape-scan2] evaluating shape boundary probes")
	for alpha_probe in probe_alphas:
		if total_alpha_checks() >= max_checks:
			break
		shape_diag = shape_at(alpha_probe, label="shape-probe2")
		shape_probe_table.append(compact_shape(shape_diag))
		if shape_diag['shape_ok']:
			first_shape_pass = {'alpha': float(alpha_probe), 'diag': shape_diag}
			vprint(f"[shape-scan2] first shape pass at alpha={float(alpha_probe):.6f}")
			break
		last_shape_fail = {'alpha': float(alpha_probe), 'diag': shape_diag}
	if first_shape_pass is None:
		final_alpha = float(last_shape_fail['alpha']) if last_shape_fail is not None else float(alpha_left)
		final_diag = full_diag(final_alpha, label="no-shape-final2")
		probe_table = _merge_alpha_probe_tables(shape_probe_table, [])
		info = make_info(
			False, 'no_shape_passing_probe', final_diag=final_diag,
			extra={'shape_probe_table': shape_probe_table, 'cdf_probe_table': [], 'coarse_probe_table': probe_table}
		)
		return np.nan, None, G_hat, grid_x, info

	if last_shape_fail is None:
		alpha_shape_diag = first_shape_pass['diag']
		shape_boundary = {
			'status': 'left_endpoint_shape_passing',
			'lo': float(alpha_left),
			'hi': float(alpha_shape_diag['alpha']),
			'width': 0.0,
		}
	else:
		alpha_shape_diag, shape_boundary = refine_shape(last_shape_fail['diag'], first_shape_pass['diag'])
	alpha_shape = float(alpha_shape_diag['alpha'])
	full_at_shape = full_diag(alpha_shape, label="direct-shape2")
	if full_at_shape['full_ok']:
		cdf_probe_table = [compact_cdf(full_at_shape)]
		probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
		shared_extra = {
			'alpha_shape': alpha_shape,
			'alpha_cdf': alpha_shape,
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {'status': 'direct_alpha_shape_pass', 'lo': float(alpha_shape), 'hi': float(alpha_shape), 'width': 0.0},
			'shape_profile_has_holes': False,
			'cdf_profile_has_holes': False,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': compact_cdf(full_at_shape),
		}
		info = make_info(True, 'overlap_shape_boundary', final_diag=full_at_shape, extra=shared_extra)
		return alpha_shape, full_at_shape['H_hat'], G_hat, grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=alpha_right,
		alpha_tol=alpha_tol,
		max_checks=max_checks,
		total_checks=total_alpha_checks,
		evaluate_cdf=cdf_at,
		compact_cdf=compact_cdf,
		vprint=vprint,
		stop_after_pass_at=alpha_shape,
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
	if not cdf_boundary_result['success']:
		first_cdf_fail = cdf_boundary_result.get('first_cdf_fail')
		final_alpha = float(first_cdf_fail['alpha']) if first_cdf_fail is not None else float(alpha_shape)
		final_diag = full_diag(final_alpha, label="no-cdf-final2")
		extra = {
			'alpha_shape': alpha_shape,
			'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': None,
		}
		info = make_info(False, 'no_cdf_passing_probe', final_diag=final_diag, extra=extra)
		return np.nan, None, G_hat, grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = {
		'alpha_shape': alpha_shape,
		'alpha_cdf': alpha_cdf,
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary_result['cdf_boundary'],
		'shape_profile_has_holes': False,
		'cdf_profile_has_holes': False,
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'coarse_probe_table': probe_table,
		'alpha_shape_diag': compact_shape(alpha_shape_diag),
		'alpha_cdf_diag': compact_cdf(alpha_cdf_diag),
	}

	def scan_for_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or total_alpha_checks() >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan_diags = []
		for a in alphas:
			if total_alpha_checks() >= max_checks:
				break
			diag = full_diag(a, label=label)
			scan_diags.append(diag)
			if diag['full_ok']:
				return diag, scan_diags
		return None, scan_diags

	if gap <= 0.0:
		full_at_shape = full_diag(alpha_shape, label="overlap-shape2")
		if full_at_shape['full_ok']:
			info = make_info(True, 'overlap_shape_boundary', final_diag=full_at_shape, extra=shared_extra)
			return alpha_shape, full_at_shape['H_hat'], G_hat, grid_x, info
		pass_diag, scan_diags = scan_for_full_pass(alpha_shape, alpha_cdf, "overlap-scan2")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = [compact_diag(d) for d in scan_diags]
		if pass_diag is not None:
			info = make_info(True, 'overlap_local_full_pass', final_diag=pass_diag, extra=extra)
			return float(pass_diag['alpha']), pass_diag['H_hat'], G_hat, grid_x, info
		info = make_info(False, 'overlap_without_full_passing_probe', final_diag=full_at_shape, extra=extra)
		return np.nan, None, G_hat, grid_x, info
	if gap <= gap_tol:
		pass_diag, scan_diags = scan_for_full_pass(alpha_cdf, alpha_shape, "near-miss2")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = [compact_diag(d) for d in scan_diags]
		if pass_diag is not None:
			info = make_info(True, 'near_miss_full_passing_probe', final_diag=pass_diag, extra=extra)
			return float(pass_diag['alpha']), pass_diag['H_hat'], G_hat, grid_x, info
		full_at_shape = full_diag(alpha_shape, label="near-miss-shape2")
		info = make_info(False, 'near_miss_no_overlap', final_diag=full_at_shape, extra=extra)
		return np.nan, None, G_hat, grid_x, info
	full_at_shape = full_diag(alpha_shape, label="gap-shape2")
	info = make_info(False, 'separated_boundaries_gap', final_diag=full_at_shape, extra=shared_extra)
	return np.nan, None, G_hat, grid_x, info


def alpha_minimize_fixed_derived_x0(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=2.0,
	x_count_weight=0.25,
	require_flat_tail_reject=False,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5,
	_cutoff_strategy="weighted",
	_fixed_shape_requires_flat=False,
):
	"""
	Helper-based bracketing-derived fixed-cutoff search.

	This is the pava3 version of alpha_minimize_fixed_derived_x0.  Shape,
	CDF, flat-tail, and cutoff-derivation evaluations share one context; CDF
	validity uses cdf_B bootstrap replicates while shape and flat-tail use B.
	"""
	if max_checks < 1:
		raise ValueError("max_checks must be at least 1.")
	if n_alpha_probes < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")
	_cutoff_strategy = str(_cutoff_strategy)
	if _cutoff_strategy not in {"weighted", "rightmost"}:
		raise ValueError("_cutoff_strategy must be 'weighted' or 'rightmost'.")
	_fixed_shape_requires_flat = bool(_fixed_shape_requires_flat)
	x_count_weight = float(x_count_weight)
	if not np.isfinite(x_count_weight) or not (0.0 <= x_count_weight <= 1.0):
		raise ValueError("x_count_weight must be finite and in [0, 1].")
	x_count_weight_effective = 0.0 if _cutoff_strategy == "rightmost" else x_count_weight
	if not use_tail_count_cutoff and x_count_weight != 0.0:
		warnings.warn(
			"alpha_minimize_fixed_derived_x0: x_count_weight is ignored when "
			"use_tail_count_cutoff=False because x_count is not count-derived.",
			RuntimeWarning
		)
		x_count_weight_effective = 0.0

	boot = _BootstrapPlan.make(
		"pava3.alpha_minimize_fixed_derived_x0",
		B, delta, cdf_delta=cdf_delta, cdf_B=cdf_B,
		flat_delta=flat_delta, require_flat_tail_reject=require_flat_tail_reject
	)
	ctx = _DerivedSearchContext(
		x_mix, x_alt,
		label="fixed_derived_x0",
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		alpha_tol=alpha_tol,
		boot=boot,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		shape_weight_gamma=shape_weight_gamma,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		flat_bisect_mode="three_way",
		cdf_ref_max_dist=None,
	)
	ctx.setup_print(
		"setup-fixed-derived2", max_checks, gap_tol,
		extra=(
			f", x_count_weight={x_count_weight:.3g}, "
			f"x_count_weight_eff={x_count_weight_effective:.3g}, "
			f"shape_weight_gamma={float(shape_weight_gamma):.3g}, "
			f"cutoff_strategy={_cutoff_strategy}, fixed_shape_requires_flat={_fixed_shape_requires_flat}"
		)
	)
	mode = 'fixed_derived_x0_bootstrap_pava3'
	if ctx.setup_error is not None:
		return np.nan, None, None, None, ctx.grid_x, ctx.failure_info(mode, ctx.setup_error['reason'])
	if not ctx.initialize_g_anchor():
		info = ctx.failure_info(mode, 'no_convex_compatible_matched_cutoff')
		return np.nan, None, None, None, ctx.grid_x, info

	def shape_probe_record(diag, meta):
		out = ctx.compact_shape(diag)
		if out is None:
			out = {}
		out.update({
			'search': meta.get('search') if meta else None,
			'updated': bool(meta.get('updated', False)) if meta else False,
		})
		return out

	shape_bracket = _run_shape_alpha_bracket(
		alpha_left=ctx.alpha_left,
		alpha_right=ctx.alpha_right,
		n_alpha_probes=ctx.n_alpha_probes,
		alpha_tol=ctx.alpha_tol,
		max_checks=max_checks,
		total_checks=ctx.total_alpha_checks,
		evaluate_shape_path=lambda a, label: ctx.evaluate_shape_path(a, label=label, cutoff_strategy=_cutoff_strategy),
		shape_probe_record=shape_probe_record,
		vprint=ctx.vprint,
		refine=False,
	)
	if not shape_bracket['success']:
		last_fail = shape_bracket.get('last_shape_fail')
		last_diag = last_fail.get('diag') if isinstance(last_fail, dict) else None
		info = ctx.failure_info(
			mode,
			'max_checks_exhausted_before_shape_bracket' if ctx.total_alpha_checks() >= max_checks else shape_bracket['reason'],
			extra={
				'shape_probe_table': shape_bracket['shape_probe_table'],
				'alpha_shape_diag': ctx.compact_shape(last_diag) if last_diag is not None else None,
			}
		)
		return np.nan, None, None, None, ctx.grid_x, info

	alpha_br = float(shape_bracket['alpha_shape'])
	x_r_br_pos = int(shape_bracket['alpha_shape_pos'])
	br_diag = shape_bracket['alpha_shape_diag']
	x_r_br = ctx.candidate_value(x_r_br_pos)
	x_count = ctx.candidate_value(ctx.x_count_pos)
	ctx.vprint(
		f"[derive2] shape boundary alpha_br={alpha_br:.6f}, "
		f"x_br={x_r_br:.6g}, cutoff_strategy={_cutoff_strategy}"
	)

	moved_back = False
	moved_back_steps = 0
	target_pos = None
	target_valid_initial = None
	target_selection_initial = None
	x_target = None
	if _cutoff_strategy == "rightmost":
		x0_star_pos = x_r_br_pos
		x0_star_diag = br_diag
		ctx.vprint(
			f"[derive-x0-rightmost2] x_right_br={x_r_br:.6g}, "
			f"x_count={x_count:.6g}, target_pos={x0_star_pos}"
		)
	else:
		x_target = (1.0 - x_count_weight_effective)*x_r_br + x_count_weight_effective*x_count
		target_span = np.arange(x_r_br_pos, ctx.x_count_pos + 1, dtype=int)
		target_values = ctx.grid_x[ctx.candidate_indices[target_span]]
		target_pos = int(target_span[int(np.argmin(np.abs(target_values - x_target)))])
		ctx.vprint(
			f"[derive-x02] x_R_br={x_r_br:.6g}, x_count={x_count:.6g}, "
			f"x_target={x_target:.6g}, target_pos={target_pos}"
		)
		target_diag = br_diag if target_pos == x_r_br_pos else ctx.shape_r(alpha_br, target_pos, label="derive-x0-target2")
		target_valid_initial = bool(target_diag['shape_ok'])
		target_selection_initial = bool(
			ctx.cutoff_selection_ok(alpha_br, target_pos, target_diag, label="derive-flat-target2")
		)
		x0_star_pos = target_pos
		x0_star_diag = target_diag
		if not target_selection_initial:
			ctx.vprint("[derive-x02] target cutoff fails selection; moving back toward x_R_br")
			for pos in range(target_pos - 1, x_r_br_pos - 1, -1):
				cand_diag = br_diag if pos == x_r_br_pos else ctx.shape_r(alpha_br, pos, label="derive-x0-fallback2")
				moved_back_steps += 1
				if ctx.cutoff_selection_ok(alpha_br, pos, cand_diag, label="derive-flat-fallback2"):
					x0_star_pos = pos
					x0_star_diag = cand_diag
					moved_back = True
					break
	if not ctx.cutoff_selection_ok(alpha_br, x0_star_pos, x0_star_diag, label="derive-flat-final2"):
		info = ctx.failure_info(mode, 'derived_x0_validation_failed', extra={
			'alpha_br': float(alpha_br),
			'x_R_br': float(x_r_br),
			'x_target': None if x_target is None else float(x_target),
			'target_valid_initial': target_valid_initial,
			'target_selection_initial': target_selection_initial,
			'x0_moved_back': bool(moved_back),
			'x0_moved_back_steps': int(moved_back_steps),
		})
		return np.nan, None, None, None, ctx.grid_x, info

	x0_star = ctx.candidate_value(x0_star_pos)
	ctx.vprint(
		f"[derive-x02] fixed x0_star={x0_star:.6g}, "
		f"target_initial={'None' if target_pos is None else f'{ctx.candidate_value(target_pos):.6g}'}, "
		f"moved_back={moved_back}"
	)

	derive_extra = {
		'alpha_br': float(alpha_br),
		'cutoff_strategy': _cutoff_strategy,
		'fixed_shape_requires_flat': bool(_fixed_shape_requires_flat),
		'derive_shape_boundary': shape_bracket['shape_boundary'],
		'derive_shape_probe_table': shape_bracket['shape_probe_table'],
		'alpha_br_diag': ctx.compact_shape(br_diag),
		'n_shape_refs': int(len(ctx.shape_refs)),
		'x_R_br': float(x_r_br),
		'x_right_br': float(x_r_br) if _cutoff_strategy == "rightmost" else None,
		'x_target': None if x_target is None else float(x_target),
		'x_count_weight': float(x_count_weight),
		'x_count_weight_effective': float(x_count_weight_effective),
		'x0_target_initial': None if target_pos is None else ctx.candidate_value(target_pos),
		'target_pos': None if target_pos is None else int(target_pos),
		'target_valid_initial': None if target_valid_initial is None else bool(target_valid_initial),
		'target_selection_initial': None if target_selection_initial is None else bool(target_selection_initial),
		'x0_moved_back': bool(moved_back),
		'x0_moved_back_steps': int(moved_back_steps),
	}

	fixed_shape_cache = {}
	fixed_full_cache = {}

	def fixed_shape_ok(diag):
		return bool(diag.get('fixed_shape_ok', diag.get('shape_ok', False)))

	def fixed_shape_msg(diag):
		if _fixed_shape_requires_flat and ctx.require_flat_tail_reject:
			return (
				f"ShapeR={_ok_text(diag['shape_ok'])}, "
				f"FlatReject={_ok_text(diag.get('flat_reject', False))}, "
				f"FixedShape={_ok_text(fixed_shape_ok(diag))}"
			)
		return f"ShapeR={_ok_text(diag['shape_ok'])}"

	def fixed_shape_at(alpha, label="fixed-shape"):
		alpha = float(alpha)
		if alpha in fixed_shape_cache:
			diag = fixed_shape_cache[alpha]
			ctx.vprint(
				f"[{label} cached] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
				f"{fixed_shape_msg(diag)}"
			)
			return diag
		ctx.alpha_checks += 1
		diag = dict(ctx.shape_r(alpha, x0_star_pos, label=f"{label}-eval"))
		if _fixed_shape_requires_flat and ctx.require_flat_tail_reject:
			flat_diag = ctx.flat_reject(alpha, x0_star_pos, diag, label=f"{label}-flat")
			diag['fixed_shape_ok'] = bool(diag['shape_ok'] and flat_diag['flat_reject'])
		else:
			diag['fixed_shape_ok'] = bool(diag['shape_ok'])
		fixed_shape_cache[alpha] = diag
		ctx.vprint(
			f"[{label} #{ctx.alpha_checks}] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
			f"{fixed_shape_msg(diag)}"
		)
		return diag

	def fixed_cdf_at(alpha, label="fixed-cdf"):
		diag = ctx.cdf_at(alpha, x0_star_pos, alpha_ref=alpha, label=label)
		return diag

	def merge_fixed_diag(shape_diag, cdf_diag):
		diag = dict(shape_diag)
		diag.update({
			'cdf_ok': bool(cdf_diag['cdf_ok']),
			'full_ok': bool(fixed_shape_ok(shape_diag) and cdf_diag['cdf_ok']),
			'valid': bool(fixed_shape_ok(shape_diag) and cdf_diag['cdf_ok']),
			'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
			'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
			'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
			'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
			'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
			'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
			'cdf_B': int(cdf_diag.get('cdf_B', boot.cdf_B)),
			'cdf_delta': float(cdf_diag.get('cdf_delta', boot.cdf_delta)),
			'cdf_bootstrap_quantile': float(cdf_diag.get('cdf_bootstrap_quantile', boot.q_cdf)),
		})
		return diag

	def fixed_full_at(alpha, label="fixed-full"):
		alpha = float(alpha)
		if alpha in fixed_full_cache:
			diag = fixed_full_cache[alpha]
			ctx.vprint(
				f"[{label} cached] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
				f"{fixed_shape_msg(diag)}, CDFR={_ok_text(diag['cdf_ok'])}, Full={_ok_text(diag['valid'])}"
			)
			return diag
		shape_diag = fixed_shape_at(alpha, label=f"{label}-shape")
		cdf_diag = fixed_cdf_at(alpha, label=f"{label}-cdf")
		diag = merge_fixed_diag(shape_diag, cdf_diag)
		fixed_full_cache[alpha] = diag
		ctx.vprint(
			f"[{label}] alpha={alpha:.6f}, x0_star={x0_star:.6g}, "
			f"{fixed_shape_msg(diag)}, CDFR={_ok_text(diag['cdf_ok'])}, Full={_ok_text(diag['valid'])}"
		)
		return diag

	def compact_fixed_shape(diag):
		out = ctx.compact_shape(diag)
		if out is not None:
			out['fixed_shape_ok'] = bool(fixed_shape_ok(diag))
		return out

	def compact_fixed_cdf(diag):
		return ctx.compact_cdf(diag)

	def compact_fixed_diag(diag):
		if diag is None:
			return None
		out = compact_fixed_shape(diag)
		out.update(compact_fixed_cdf(diag))
		out['full_ok'] = bool(diag['valid'])
		return out

	def make_info(success, reason, alpha=None, pos=None, r_diag=None,
			alpha_lo=None, alpha_hi=None, extra=None):
		x0 = ctx.candidate_value(pos) if pos is not None else None
		info = {
			'success': bool(success),
			'reason': reason,
			'alpha': float(alpha) if alpha is not None else np.nan,
			'x0': x0,
			'alpha_lo': float(alpha_lo) if alpha_lo is not None else None,
			'alpha_hi': float(alpha_hi) if alpha_hi is not None else None,
			'max_checks': int(max_checks),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
			'cutoff_strategy': _cutoff_strategy,
			'fixed_shape_requires_flat': bool(_fixed_shape_requires_flat),
			'x_count_weight': float(x_count_weight),
			'x_count_weight_effective': float(x_count_weight_effective),
		}
		info.update(ctx.base_info(mode))
		if r_diag is not None:
			info.update({
				'shape_ok': bool(r_diag.get('shape_ok', False)),
				'fixed_shape_ok': bool(r_diag.get('fixed_shape_ok', r_diag.get('shape_ok', False))),
				'residual_lcm_gap': float(r_diag['shape_gap']),
				'residual_lcm_tol': float(r_diag['shape_tol']),
				'cdf_range_violation': float(r_diag['cdf_range_violation']),
				'cdf_low_violation': float(r_diag['cdf_low_violation']),
				'cdf_high_violation': float(r_diag['cdf_high_violation']),
				'cdf_mon_violation': float(r_diag['cdf_mon_violation']),
				'mix_tail_count': int(r_diag['mix_tail_count']),
				'alt_tail_count': int(r_diag['alt_tail_count']),
			})
			if 'cdf_range_tol' in r_diag:
				info.update({
					'cdf_ok': bool(r_diag.get('cdf_ok', False)),
					'full_ok': bool(r_diag.get('valid', False)),
					'cdf_range_tol': float(r_diag['cdf_range_tol']),
					'cdf_mon_tol': float(r_diag['cdf_mon_tol']),
				})
			if 'flat_reject' in r_diag:
				info.update({
					'flat_available': bool(r_diag.get('flat_available', False)),
					'flat_reject': None if r_diag.get('flat_reject') is None else bool(r_diag.get('flat_reject')),
					'flat_stat': float(r_diag.get('flat_stat', np.nan)),
					'flat_tol': float(r_diag.get('flat_tol', np.nan)),
				})
		if extra:
			info.update(extra)
		return info

	def refine_fixed_shape(fail_diag, pass_diag):
		lo = fail_diag
		hi = pass_diag
		status = 'alpha_tol_reached'
		iter_idx = 0
		while (hi['alpha'] - lo['alpha']) > ctx.alpha_tol and ctx.total_alpha_checks() < max_checks:
			iter_idx += 1
			mid = 0.5 * (lo['alpha'] + hi['alpha'])
			ctx.vprint(f"[fixed-shape-bisect2 #{iter_idx:02d}] lo={lo['alpha']:.6f}, hi={hi['alpha']:.6f}, mid={mid:.6f}")
			mid_diag = fixed_shape_at(mid, label="fixed-shape-bisect2")
			if fixed_shape_ok(mid_diag):
				hi = mid_diag
			else:
				lo = mid_diag
		if (hi['alpha'] - lo['alpha']) > ctx.alpha_tol:
			status = 'max_checks_reached'
		return hi, {'status': status, 'lo': float(lo['alpha']), 'hi': float(hi['alpha']), 'width': float(hi['alpha'] - lo['alpha'])}

	probe_alphas = np.linspace(ctx.alpha_left, ctx.alpha_right, num=ctx.n_alpha_probes)
	shape_probe_table = []
	last_shape_fail = None
	first_shape_pass = None
	ctx.vprint("[fixed-shape-scan2] evaluating shape boundary probes at derived fixed cutoff")
	for alpha_probe in probe_alphas:
		if ctx.total_alpha_checks() >= max_checks:
			break
		diag = fixed_shape_at(alpha_probe, label="fixed-shape-probe2")
		shape_probe_table.append(compact_fixed_shape(diag))
		if fixed_shape_ok(diag):
			first_shape_pass = {'alpha': float(alpha_probe), 'diag': diag}
			ctx.vprint(f"[fixed-shape-scan2] first shape pass alpha={float(alpha_probe):.6f}")
			break
		last_shape_fail = {'alpha': float(alpha_probe), 'diag': diag}
	if first_shape_pass is None:
		final_diag = last_shape_fail['diag'] if last_shape_fail is not None else x0_star_diag
		extra = dict(derive_extra)
		extra.update({
			'gap_status': 'no_shape_passing_probe',
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': [],
			'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, []),
			'alpha_shape_diag': None,
			'alpha_cdf_diag': None,
		})
		info = make_info(False, 'no_fixed_shape_passing_probe', alpha=np.nan, pos=x0_star_pos, r_diag=final_diag, extra=extra)
		return np.nan, None, None, None, ctx.grid_x, info
	if last_shape_fail is None:
		alpha_shape_diag = first_shape_pass['diag']
		shape_boundary = {'status': 'left_endpoint_shape_passing', 'lo': float(ctx.alpha_left), 'hi': float(alpha_shape_diag['alpha']), 'width': 0.0}
	else:
		alpha_shape_diag, shape_boundary = refine_fixed_shape(last_shape_fail['diag'], first_shape_pass['diag'])

	alpha_shape = float(alpha_shape_diag['alpha'])
	full_at_shape = fixed_full_at(alpha_shape, label="direct-shape2")
	if full_at_shape['valid']:
		cdf_probe_table = [compact_fixed_cdf(full_at_shape)]
		shared_extra = dict(derive_extra)
		shared_extra.update({
			'alpha_shape': float(alpha_shape), 'alpha_cdf': float(alpha_shape),
			'boundary_gap': 0.0, 'overlap_width': 0.0, 'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {'status': 'direct_alpha_shape_pass', 'lo': float(alpha_shape), 'hi': float(alpha_shape), 'width': 0.0},
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table),
			'alpha_shape_diag': compact_fixed_shape(alpha_shape_diag),
			'alpha_cdf_diag': compact_fixed_cdf(full_at_shape),
		})
		info = make_info(True, 'overlap_shape_boundary_fixed_derived_x0', alpha=alpha_shape, pos=x0_star_pos, r_diag=full_at_shape, alpha_lo=shape_boundary.get('lo'), alpha_hi=alpha_shape, extra=shared_extra)
		ctx.vprint(f"[done-fixed-derived2] alpha_hat={alpha_shape:.6f}, x0={info['x0']:.6g}, reason={info['reason']}")
		return alpha_shape, info['x0'], full_at_shape['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=ctx.alpha_right,
		alpha_tol=ctx.alpha_tol,
		max_checks=max_checks,
		total_checks=ctx.total_alpha_checks,
		evaluate_cdf=lambda a, label: fixed_cdf_at(a, label=label),
		compact_cdf=compact_fixed_cdf,
		vprint=ctx.vprint,
		stop_after_pass_at=alpha_shape,
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	probe_table = _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table)
	if not cdf_boundary_result['success']:
		first_cdf_fail = cdf_boundary_result.get('first_cdf_fail')
		if first_cdf_fail is not None:
			fail_alpha = float(first_cdf_fail['alpha'])
			fail_shape_diag = fixed_shape_at(fail_alpha, label="cdf-fail-shape2")
			final_diag = merge_fixed_diag(fail_shape_diag, first_cdf_fail['diag'])
		else:
			final_diag = alpha_shape_diag
		extra = dict(derive_extra)
		extra.update({
			'alpha_shape': alpha_shape, 'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': probe_table,
			'alpha_shape_diag': compact_fixed_shape(alpha_shape_diag),
			'alpha_cdf_diag': None,
		})
		info = make_info(False, 'no_fixed_cdf_passing_probe', alpha=np.nan, pos=x0_star_pos, r_diag=final_diag, extra=extra)
		return np.nan, None, None, None, ctx.grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = dict(derive_extra)
	shared_extra.update({
		'alpha_shape': float(alpha_shape),
		'alpha_cdf': float(alpha_cdf),
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary_result['cdf_boundary'],
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'coarse_probe_table': probe_table,
		'alpha_shape_diag': compact_fixed_shape(alpha_shape_diag),
		'alpha_cdf_diag': compact_fixed_cdf(alpha_cdf_diag),
	})
	ctx.vprint(
		f"[fixed-boundaries2] alpha_shape={alpha_shape:.6f}, alpha_cdf={alpha_cdf:.6f}, "
		f"gap={gap:.3g}, status={shared_extra['gap_status']}"
	)

	def scan_for_fixed_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or ctx.total_alpha_checks() >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan = []
		for a in alphas:
			if ctx.total_alpha_checks() >= max_checks:
				break
			diag = fixed_full_at(a, label=label)
			scan.append(compact_fixed_diag(diag))
			if diag['valid']:
				return diag, scan
		return None, scan

	full_at_shape = fixed_full_at(alpha_shape, label="fixed-at-shape2")
	if gap <= 0.0:
		if full_at_shape['valid']:
			info = make_info(True, 'overlap_shape_boundary_fixed_derived_x0', alpha=alpha_shape, pos=x0_star_pos, r_diag=full_at_shape, alpha_lo=shape_boundary.get('lo'), alpha_hi=alpha_shape, extra=shared_extra)
			return alpha_shape, info['x0'], full_at_shape['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		pass_diag, scan = scan_for_fixed_full_pass(alpha_shape, alpha_cdf, "fixed-overlap-scan2")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = scan
		if pass_diag is not None:
			info = make_info(True, 'overlap_local_full_pass_fixed_derived_x0', alpha=pass_diag['alpha'], pos=x0_star_pos, r_diag=pass_diag, alpha_lo=shape_boundary.get('lo'), alpha_hi=pass_diag['alpha'], extra=extra)
			return float(pass_diag['alpha']), info['x0'], pass_diag['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		info = make_info(False, 'overlap_without_full_passing_probe_fixed_derived_x0', alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape, extra=extra)
		return np.nan, None, None, None, ctx.grid_x, info
	if gap <= gap_tol:
		pass_diag, scan = scan_for_fixed_full_pass(alpha_cdf, alpha_shape, "fixed-near-miss2")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = scan
		if pass_diag is not None:
			info = make_info(True, 'near_miss_full_passing_probe_fixed_derived_x0', alpha=pass_diag['alpha'], pos=x0_star_pos, r_diag=pass_diag, alpha_lo=alpha_cdf, alpha_hi=alpha_shape, extra=extra)
			return float(pass_diag['alpha']), info['x0'], pass_diag['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		info = make_info(False, 'near_miss_no_overlap_fixed_derived_x0', alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape, extra=extra)
		return np.nan, None, None, None, ctx.grid_x, info
	info = make_info(False, 'separated_boundaries_gap_fixed_derived_x0', alpha=np.nan, pos=x0_star_pos, r_diag=full_at_shape, extra=shared_extra)
	return np.nan, None, None, None, ctx.grid_x, info


def alpha_minimize_fixed_derived_x0_rightmost(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=2.0,
	require_flat_tail_reject=True,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5,
):
	"""Rightmost-cutoff wrapper over the helper-based fixed-derived search."""
	return alpha_minimize_fixed_derived_x0(
		x_mix,
		x_alt,
		alpha_tol=alpha_tol,
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		B=B,
		delta=delta,
		cdf_delta=cdf_delta,
		cdf_B=cdf_B,
		flat_delta=flat_delta,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		max_checks=max_checks,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		shape_weight_gamma=shape_weight_gamma,
		x_count_weight=0.0,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		gap_tol=gap_tol,
		near_miss_probes=near_miss_probes,
		_cutoff_strategy="rightmost",
			_fixed_shape_requires_flat=True,
		)


def alpha_minimize_rightmost_delta_ladder(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	deltas=(0.25, 0.05, 0.01, 0.005),
	cdf_delta=None,
	cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=32,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=6.0,
	require_flat_tail_reject=True,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5,
):
	"""
	Try the rightmost derived-x0 search over a sequence of shape-test deltas.

	Each rung uses the same ``random_state`` value, matching the sweep scripts'
	paired-fallback convention.  The first successful rung is returned.  The
	return shape is the derived-x0 six-tuple:

		(alpha_hat, x0_hat, H_hat, G_hat, grid_x, info)
	"""
	if deltas is None:
		raise ValueError("deltas must be a nonempty iterable of positive values.")
	deltas = tuple(float(delta_value) for delta_value in deltas)
	if len(deltas) == 0:
		raise ValueError("deltas must be a nonempty iterable of positive values.")
	if any((not np.isfinite(delta_value)) or delta_value <= 0.0 or delta_value >= 1.0 for delta_value in deltas):
		raise ValueError("all deltas must be finite values in (0, 1).")

	attempts = []
	last_result = None
	for rung_index, delta_value in enumerate(deltas):
		result = alpha_minimize_fixed_derived_x0_rightmost(
			x_mix,
			x_alt,
			alpha_tol=alpha_tol,
			alpha_bounds=alpha_bounds,
			n_alpha_probes=n_alpha_probes,
			B=B,
			delta=delta_value,
			cdf_delta=cdf_delta,
			cdf_B=cdf_B,
			flat_delta=flat_delta,
			min_mix_tail=min_mix_tail,
			min_alt_tail=min_alt_tail,
			max_checks=max_checks,
			random_state=random_state,
			grid=grid,
			candidate_cutoffs=candidate_cutoffs,
			max_grid_points=max_grid_points,
			verbose=verbose,
			x0_tol=x0_tol,
			shape_weight_gamma=shape_weight_gamma,
			require_flat_tail_reject=require_flat_tail_reject,
			use_tail_count_cutoff=use_tail_count_cutoff,
			gap_tol=gap_tol,
			near_miss_probes=near_miss_probes,
		)
		last_result = result
		alpha_hat, x0_hat, H_hat, G_hat, grid_x, info = result
		info_dict = info if isinstance(info, dict) else {}
		success = bool(info_dict.get('success', False)) and _diag_float(alpha_hat) is not None
		attempts.append({
			'rung': int(rung_index),
			'delta': float(delta_value),
			'success': bool(success),
			'reason': info_dict.get('reason'),
			'alpha_hat': _diag_float(alpha_hat),
			'x0_hat': _diag_float(x0_hat),
			'alpha_checks': _diag_float(info_dict.get('alpha_checks')),
			'total_alpha_checks': _diag_float(info_dict.get('total_alpha_checks')),
		})
		if success:
			out_info = dict(info_dict)
			out_info.update({
				'mode': 'rightmost_delta_ladder_pava3',
				'x0_source': 'derived_rightmost_ladder',
				'ladder_success': True,
				'ladder_success_rung': int(rung_index),
				'ladder_delta': float(delta_value),
				'ladder_deltas': tuple(float(d) for d in deltas),
				'ladder_attempts': attempts,
				'shape_weight_gamma': float(shape_weight_gamma),
			})
			return alpha_hat, x0_hat, H_hat, G_hat, grid_x, out_info

	alpha_hat, x0_hat, H_hat, G_hat, grid_x, info = last_result
	out_info = dict(info) if isinstance(info, dict) else {}
	out_info.update({
		'success': False,
		'mode': 'rightmost_delta_ladder_pava3',
		'x0_source': 'derived_rightmost_ladder',
		'ladder_success': False,
		'ladder_success_rung': None,
		'ladder_delta': None,
		'ladder_deltas': tuple(float(d) for d in deltas),
		'ladder_attempts': attempts,
		'shape_weight_gamma': float(shape_weight_gamma),
		'reason': out_info.get('reason') or 'no_ladder_rung_succeeded',
	})
	return np.nan, None, H_hat, G_hat, grid_x, out_info


def alpha_minimize(
	x_mix,
	x_alt,
	x0=None,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=None,
	deltas=None,
	cdf_delta=None,
	cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=None,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=6.0,
	require_flat_tail_reject=True,
	use_tail_count_cutoff=True,
	gap_tol=None,
	near_miss_probes=5,
):
	"""
	Uniform pava3 entry point for either given-x0 or derived-x0 estimation.

	If ``x0`` is supplied, this calls ``alpha_minimize_given_x0_revisited`` and
	returns ``x0`` as the second output.  That fixed-x0 path currently uses the
	unweighted pava2-style shape statistic.

	If ``x0`` is omitted, this calls ``alpha_minimize_rightmost_delta_ladder``.
	By default the ladder uses ``deltas=(0.25, 0.05, 0.01, 0.005)`` and
	``shape_weight_gamma=6``.

	Returns
	-------
	alpha_hat, x0_hat, H_hat, G_hat, grid_x, info
	"""
	if x0 is not None:
		fixed_delta = 0.05 if delta is None else float(delta)
		fixed_max_checks = 64 if max_checks is None else int(max_checks)
		alpha_hat, H_hat, G_hat, grid_x, info = alpha_minimize_given_x0_revisited(
			x_mix,
			x_alt,
			x0,
			alpha_tol=alpha_tol,
			alpha_bounds=alpha_bounds,
			n_alpha_probes=n_alpha_probes,
			B=B,
			delta=fixed_delta,
			cdf_delta=cdf_delta,
			cdf_B=cdf_B,
			gap_tol=gap_tol,
			near_miss_probes=near_miss_probes,
			max_checks=fixed_max_checks,
			random_state=random_state,
			grid=grid,
			max_grid_points=max_grid_points,
			verbose=verbose,
		)
		out_info = dict(info) if isinstance(info, dict) else {}
		out_info.update({
			'wrapper': 'pava3.alpha_minimize',
			'x0_source': 'given',
			'x0': float(x0),
			'fixed_x0_delta': float(fixed_delta),
			'shape_statistic': out_info.get('shape_statistic', 'unweighted_ise'),
			'shape_weight_gamma': None,
			'derived_options_unused': {
				'deltas': None if deltas is None else tuple(float(d) for d in deltas),
				'flat_delta': None if flat_delta is None else float(flat_delta),
				'min_mix_tail': int(min_mix_tail),
				'min_alt_tail': int(min_alt_tail),
				'x0_tol': float(x0_tol),
				'require_flat_tail_reject': bool(require_flat_tail_reject),
				'use_tail_count_cutoff': bool(use_tail_count_cutoff),
				'shape_weight_gamma': float(shape_weight_gamma),
			},
		})
		return alpha_hat, float(x0), H_hat, G_hat, grid_x, out_info

	if deltas is None:
		deltas = (0.25, 0.05, 0.01, 0.005) if delta is None else (float(delta),)
	derived_max_checks = 32 if max_checks is None else int(max_checks)
	alpha_hat, x0_hat, H_hat, G_hat, grid_x, info = alpha_minimize_rightmost_delta_ladder(
		x_mix,
		x_alt,
		alpha_tol=alpha_tol,
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		B=B,
		deltas=deltas,
		cdf_delta=cdf_delta,
		cdf_B=cdf_B,
		flat_delta=flat_delta,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		max_checks=derived_max_checks,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		shape_weight_gamma=shape_weight_gamma,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		gap_tol=gap_tol,
		near_miss_probes=near_miss_probes,
	)
	out_info = dict(info) if isinstance(info, dict) else {}
	out_info.update({
		'wrapper': 'pava3.alpha_minimize',
		'x0_source': out_info.get('x0_source', 'derived_rightmost_ladder'),
	})
	return alpha_hat, x0_hat, H_hat, G_hat, grid_x, out_info


def alpha_minimize_screened_derived_x0(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	cdf_B=None,
	confirm_B=None,
	confirm_cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=64,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=2.0,
	require_flat_tail_reject=True,
	confirm_tail_mode=None,
	use_tail_count_cutoff=True,
	screen_B=100,
	screen_delta=0.25,
	screen_cdf_delta=None,
	screen_flat_delta=None,
	screen_tail_mode="flat",
	screen_slope_delta=None,
	screen_k_success=5,
	screen_max_cutoffs=200,
	screen_max_confirm_candidates=3,
	screen_bracket_mode="linear",
	screen_direction_majority=0.7,
	confirm_mode="full",
	screen_confirm_mode="run",
	screen_confirm_cdf_delta=None,
	screen_confirm_flat_delta=None,
	screen_confirm_slope_delta=None,
	slope_blocks=50,
	screen_confirm_slope_blocks=None,
	confirm_delta=None,
	confirm_cdf_delta=None,
	confirm_flat_delta=None,
	locate_mode="screen",
	proper_k_success=None,
):
	"""
	Screened derived-cutoff search.

	For each tested alpha, a cheap cutoff screen scans candidate suffixes from
	right to left and collects non-overlapping runs of at least screen_k_success
	consecutive passing cutoffs.  The centered cutoff from each run is checked
	according to confirm_mode: full fitted-model confirmation, screen-only
	confirmation, or intermediate PAVA shape/CDF confirmation.  Candidate
	confirmation uses confirm_B, confirm_cdf_B, and confirm_*_delta values,
	defaulting to the legacy B/cdf_B and delta/cdf_delta/flat controls.
	During alpha bisection, the cutoff
	from the current passing endpoint is tried first as a warm start; failed warm
	starts fall back to a fresh screened cutoff search.  For screen-only runs,
	screen_confirm_mode="run" preserves the original inherited-run retest, while
	screen_confirm_mode="selected" checks only the selected cutoff using the
	screen_confirm_* deltas before falling back to a fresh cutoff search.  In
	selected mode, fresh screen searches also apply the selected-cutoff
	confirmation to each located run and continue up to
	screen_max_confirm_candidates if earlier located runs fail.  slope_blocks
	may be a single block count or an iterable of block counts; iterable values
	run a multiscale slope screen and require all scales to pass.
	screen_confirm_slope_blocks optionally overrides the block scale used by
	selected-cutoff confirmation; by default it inherits slope_blocks.
	screen_tail_mode controls the tail guard used during screened cutoff
	location: "flat" is the original bootstrap raw-flat rejection screen,
	"deterministic" uses the raw-residual adjacent chord-slope guard, and
	"none" disables the tail guard.  confirm_tail_mode controls the
	confirmation-stage tail guard: "flat" preserves the previous
	fitted/bootstrap flat-tail rejection, "deterministic" uses the raw-residual
	adjacent chord-slope guard, and "none" disables the confirmation tail
	guard.  require_flat_tail_reject=True/False is kept as a backward-compatible
	alias for confirm_tail_mode="flat"/"none" when confirm_tail_mode is not
	supplied.  See screened_derived_x0_options.md for an implementation-level
	mode guide.
	"""
	confirm_tail_mode = _normalize_confirm_tail_mode(
		confirm_tail_mode, require_flat_tail_reject
	)
	require_flat_tail_reject = confirm_tail_mode != "none"
	confirm_tail_uses_bootstrap_flat = confirm_tail_mode == "flat"
	locate_mode = str(locate_mode)
	if locate_mode not in {"screen", "proper"}:
		raise ValueError("locate_mode must be 'screen' or 'proper'.")
	screen_tail_mode = str(screen_tail_mode)
	if screen_tail_mode not in {"flat", "deterministic", "none"}:
		raise ValueError("screen_tail_mode must be 'flat', 'deterministic', or 'none'.")
	screen_bracket_mode = str(screen_bracket_mode)
	if screen_bracket_mode not in {"linear", "directional"}:
		raise ValueError("screen_bracket_mode must be 'linear' or 'directional'.")
	confirm_mode = str(confirm_mode)
	if confirm_mode not in {"full", "screen_only", "pava_shape_cdf"}:
		raise ValueError("confirm_mode must be 'full', 'screen_only', or 'pava_shape_cdf'.")
	if locate_mode == "proper" and confirm_mode == "screen_only":
		raise ValueError("locate_mode='proper' requires confirm_mode='full' or 'pava_shape_cdf'.")
	screen_confirm_mode = str(screen_confirm_mode)
	if screen_confirm_mode not in {"run", "selected"}:
		raise ValueError("screen_confirm_mode must be 'run' or 'selected'.")
	screen_direction_majority = float(screen_direction_majority)
	if not (0.5 < screen_direction_majority <= 1.0):
		raise ValueError("screen_direction_majority must be in (0.5, 1].")
	if max_checks < 1:
		raise ValueError("max_checks must be at least 1.")
	if int(n_alpha_probes) < 2:
		raise ValueError("n_alpha_probes must be at least 2.")
	screen_B = int(screen_B)
	if screen_B < 1:
		raise ValueError("screen_B must be at least 1.")
	screen_k_success = int(screen_k_success)
	if screen_k_success < 1:
		raise ValueError("screen_k_success must be at least 1.")
	proper_k_success = screen_k_success if proper_k_success is None else int(proper_k_success)
	if proper_k_success < 1:
		raise ValueError("proper_k_success must be at least 1.")
	if screen_max_cutoffs is not None:
		screen_max_cutoffs = int(screen_max_cutoffs)
		if screen_max_cutoffs < screen_k_success:
			raise ValueError("screen_max_cutoffs must be at least screen_k_success.")
	screen_max_confirm_candidates = int(screen_max_confirm_candidates)
	if screen_max_confirm_candidates < 1:
		raise ValueError("screen_max_confirm_candidates must be at least 1.")
	confirm_B = int(B) if confirm_B is None else int(confirm_B)
	if confirm_B < 1:
		raise ValueError("confirm_B must be at least 1.")
	if confirm_cdf_B is None:
		confirm_cdf_B = confirm_B if cdf_B is None else int(cdf_B)
	else:
		confirm_cdf_B = int(confirm_cdf_B)
	if confirm_cdf_B < 1:
		raise ValueError("confirm_cdf_B must be at least 1.")
	slope_blocks_values = _normalize_slope_blocks(slope_blocks)
	slope_blocks_for_info = _slope_blocks_info(slope_blocks_values)
	slope_blocks_for_print = _slope_blocks_text(slope_blocks_values)
	screen_confirm_slope_blocks_values = (
		slope_blocks_values if screen_confirm_slope_blocks is None
		else _normalize_slope_blocks(screen_confirm_slope_blocks)
	)
	screen_confirm_slope_blocks_for_info = _slope_blocks_info(screen_confirm_slope_blocks_values)
	screen_confirm_slope_blocks_for_print = _slope_blocks_text(screen_confirm_slope_blocks_values)
	screen_delta = float(screen_delta)
	screen_cdf_delta = screen_delta if screen_cdf_delta is None else float(screen_cdf_delta)
	screen_flat_delta = screen_delta if screen_flat_delta is None else float(screen_flat_delta)
	screen_slope_delta = screen_delta if screen_slope_delta is None else float(screen_slope_delta)
	screen_confirm_cdf_delta = (
		screen_cdf_delta if screen_confirm_cdf_delta is None
		else float(screen_confirm_cdf_delta)
	)
	screen_confirm_flat_delta = (
		screen_flat_delta if screen_confirm_flat_delta is None
		else float(screen_confirm_flat_delta)
	)
	screen_confirm_slope_delta = (
		screen_slope_delta if screen_confirm_slope_delta is None
		else float(screen_confirm_slope_delta)
	)
	confirm_delta = delta if confirm_delta is None else float(confirm_delta)
	confirm_cdf_delta = (
		(delta if cdf_delta is None else cdf_delta)
		if confirm_cdf_delta is None else float(confirm_cdf_delta)
	)
	confirm_flat_delta = (
		(
			screen_confirm_flat_delta if confirm_mode == "pava_shape_cdf"
			else (delta if flat_delta is None else flat_delta)
		)
		if confirm_flat_delta is None else float(confirm_flat_delta)
	)
	q_screen_cdf = _bootstrap_quantile_level(screen_cdf_delta, screen_B, "pava3.alpha_minimize_screened_derived_x0.cdf_screen")
	q_screen_flat = _bootstrap_quantile_level(screen_flat_delta, screen_B, "pava3.alpha_minimize_screened_derived_x0.raw_flat_screen")
	q_screen_slope = _bootstrap_quantile_level(screen_slope_delta, screen_B, "pava3.alpha_minimize_screened_derived_x0.slope_screen")
	q_screen_confirm_cdf = (
		q_screen_cdf if screen_confirm_cdf_delta == screen_cdf_delta
		else _bootstrap_quantile_level(
			screen_confirm_cdf_delta, screen_B,
			"pava3.alpha_minimize_screened_derived_x0.confirm_cdf_screen"
		)
	)
	q_screen_confirm_flat = (
		q_screen_flat if screen_confirm_flat_delta == screen_flat_delta
		else _bootstrap_quantile_level(
			screen_confirm_flat_delta, screen_B,
			"pava3.alpha_minimize_screened_derived_x0.confirm_raw_flat_screen"
		)
	)
	q_screen_confirm_slope = (
		q_screen_slope if screen_confirm_slope_delta == screen_slope_delta
		else _bootstrap_quantile_level(
			screen_confirm_slope_delta, screen_B,
			"pava3.alpha_minimize_screened_derived_x0.confirm_slope_screen"
		)
	)
	q_pava_confirm_flat = _bootstrap_quantile_level(
		confirm_flat_delta,
		confirm_B,
		"pava3.alpha_minimize_screened_derived_x0.pava_shape_cdf.raw_flat"
	)

	boot = _BootstrapPlan.make(
		"pava3.alpha_minimize_screened_derived_x0",
		B, delta, cdf_delta=cdf_delta, cdf_B=cdf_B,
		flat_delta=flat_delta, require_flat_tail_reject=confirm_tail_uses_bootstrap_flat
	)
	confirm_boot = _BootstrapPlan.make(
		"pava3.alpha_minimize_screened_derived_x0.confirm",
		confirm_B, confirm_delta, cdf_delta=confirm_cdf_delta, cdf_B=confirm_cdf_B,
		flat_delta=confirm_flat_delta, require_flat_tail_reject=confirm_tail_uses_bootstrap_flat
	)
	ctx = _DerivedSearchContext(
		x_mix, x_alt,
		label="screened_derived_x0",
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		alpha_tol=alpha_tol,
		boot=boot,
		confirm_boot=confirm_boot,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		shape_weight_gamma=shape_weight_gamma,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		flat_bisect_mode="three_way",
		cdf_ref_max_dist=None,
	)
	mode = 'screened_derived_x0_bootstrap_pava3'
	ctx.setup_print(
		"setup-screened-x0-2", max_checks, 0.0,
		extra=(
			f", screen_B={screen_B}, screen_delta={screen_delta:.3g}, "
			f"shape_weight_gamma={float(shape_weight_gamma):.3g}, "
			f"screen_cdf_delta={screen_cdf_delta:.3g}, screen_flat_delta={screen_flat_delta:.3g}, "
			f"screen_tail_mode={screen_tail_mode}, "
			f"screen_slope_delta={screen_slope_delta:.3g}, screen_k_success={screen_k_success}, "
			f"locate_mode={locate_mode}, proper_k_success={proper_k_success}, "
			f"screen_max_cutoffs={screen_max_cutoffs}, screen_max_confirm_candidates={screen_max_confirm_candidates}, "
			f"confirm_B={confirm_B}, confirm_cdf_B={confirm_cdf_B}, "
			f"confirm_delta={confirm_delta:.3g}, confirm_cdf_delta={confirm_cdf_delta:.3g}, "
			f"confirm_flat_delta={confirm_flat_delta:.3g}, "
			f"screen_bracket_mode={screen_bracket_mode}, screen_direction_majority={screen_direction_majority:.3g}, "
			f"confirm_mode={confirm_mode}, screen_confirm_mode={screen_confirm_mode}, "
			f"screen_confirm_cdf_delta={screen_confirm_cdf_delta:.3g}, "
			f"screen_confirm_flat_delta={screen_confirm_flat_delta:.3g}, "
			f"screen_confirm_slope_delta={screen_confirm_slope_delta:.3g}, "
			f"slope_blocks={slope_blocks_for_print}, "
			f"screen_confirm_slope_blocks={screen_confirm_slope_blocks_for_print}, "
			f"confirm_tail_mode={confirm_tail_mode}"
		)
	)
	if ctx.setup_error is not None:
		return np.nan, None, None, None, ctx.grid_x, ctx.failure_info(mode, ctx.setup_error['reason'])
	if not ctx.initialize_g_anchor():
		info = ctx.failure_info(mode, 'no_convex_compatible_matched_cutoff')
		return np.nan, None, None, None, ctx.grid_x, info

	screen_candidate_checks = 0
	screen_cdf_checks = 0
	screen_flat_checks = 0
	screen_slope_checks = 0
	proper_candidate_checks = 0
	alpha_eval_table = []

	def screen_positions():
		positions = np.arange(int(ctx.x_count_pos), int(ctx.x_g_pos) - 1, -1, dtype=int)
		if screen_max_cutoffs is not None and len(positions) > screen_max_cutoffs:
			idx = _empirical_quantile_indices(len(positions), screen_max_cutoffs)
			positions = positions[idx]
		return positions

	def sample_ecdfs(F_cdf):
		x_mix_b = _sample_from_cdf(ctx.grid_x, F_cdf, ctx.n, ctx.rng)
		x_mix_b.sort(kind='mergesort')
		x_alt_b = _sample_from_cdf(ctx.grid_x, ctx.G_hat_anchor, ctx.m, ctx.rng)
		x_alt_b.sort(kind='mergesort')
		return _ecdf_on_grid(x_mix_b, ctx.grid_x), _ecdf_on_grid(x_alt_b, ctx.grid_x)

	def screen_candidate(alpha, pos, R_obs, cdf_obs, den, label,
			q_cdf=None, q_flat=None, q_slope=None, role="locate"):
		nonlocal screen_candidate_checks, screen_cdf_checks, screen_flat_checks, screen_slope_checks
		screen_candidate_checks += 1
		q_cdf = q_screen_cdf if q_cdf is None else float(q_cdf)
		q_flat = q_screen_flat if q_flat is None else float(q_flat)
		q_slope = q_screen_slope if q_slope is None else float(q_slope)
		active_slope_blocks = (
			screen_confirm_slope_blocks_values
			if role == "confirm" else slope_blocks_values
		)
		active_slope_blocks_info = _slope_blocks_info(active_slope_blocks)
		pos = int(pos)
		x0 = ctx.candidate_value(pos)
		H_lcm = _concave_projection_on_interval(ctx.grid_x, R_obs, x0, ctx.x1)
		T_obs = _lcm_gap_from_fit(ctx.grid_x, R_obs, H_lcm, x0, ctx.x1)
		H_hat = _as_valid_cdf(H_lcm)
		F_hat = _as_valid_cdf((1.0 - alpha)*H_hat + alpha*ctx.G_hat_anchor)

		range_star = np.empty(screen_B, dtype=float)
		mon_star = np.empty(screen_B, dtype=float)
		slope_star = np.empty((len(active_slope_blocks), screen_B), dtype=float)
		for b in range(screen_B):
			Fn_b, Gm_b = sample_ecdfs(F_hat)
			R_b = (Fn_b - alpha*Gm_b) / den
			cdf_b = _cdf_violation_stats(R_b)
			range_star[b] = cdf_b['range']
			mon_star[b] = cdf_b['mon']
			for i, block in enumerate(active_slope_blocks):
				slope_star[i, b] = _slope_violation_stat(ctx.grid_x, R_b, x0, ctx.x1, n_blocks=block)
		screen_cdf_checks += 1
		screen_slope_checks += 1
		cdf_range_tol = float(np.quantile(range_star, q_cdf, method='higher'))
		cdf_mon_tol = float(np.quantile(mon_star, q_cdf, method='higher'))
		cdf_range_tail_p = float(np.mean(range_star >= cdf_obs['range']))
		cdf_mon_tail_p = float(np.mean(mon_star >= cdf_obs['mon']))
		cdf_ok = bool(cdf_obs['range'] <= cdf_range_tol and cdf_obs['mon'] <= cdf_mon_tol)
		slope_stats = np.array([
			_slope_violation_stat(ctx.grid_x, R_obs, x0, ctx.x1, n_blocks=block)
			for block in active_slope_blocks
		], dtype=float)
		slope_tols = np.array([
			np.quantile(slope_star[i], q_slope, method='higher')
			for i in range(len(active_slope_blocks))
		], dtype=float)
		slope_tail_ps = np.array([
			np.mean(slope_star[i] >= slope_stats[i])
			for i in range(len(active_slope_blocks))
		], dtype=float)
		slope_ok_by_scale = slope_stats <= slope_tols
		slope_ok = bool(np.all(slope_ok_by_scale))
		with np.errstate(divide='ignore', invalid='ignore'):
			slope_ratios = np.where(
				slope_tols > 0.0,
				slope_stats / slope_tols,
				np.where(slope_stats <= 0.0, 0.0, np.inf)
			)
		worst_i = int(np.argmax(slope_ratios))
		slope_stat = float(slope_stats[worst_i])
		slope_tol = float(slope_tols[worst_i])
		slope_tail_p = float(np.min(slope_tail_ps))
		slope_worst_blocks = active_slope_blocks[worst_i]
		slope_worst_blocks_text = "all" if slope_worst_blocks is None else str(slope_worst_blocks)
		slope_worst_ratio = float(slope_ratios[worst_i])

		tail_diag = _raw_tail_guard_diag(ctx.grid_x, R_obs, x0, ctx.x1)
		if screen_tail_mode == "flat":
			U_raw, _ = _flat_tail_stat(ctx.grid_x, R_obs, x0, ctx.x1)
			_, H_flat = _flat_tail_stat(ctx.grid_x, H_hat, x0, ctx.x1)
			H_flat = _as_valid_cdf(H_flat)
			F_flat = _as_valid_cdf((1.0 - alpha)*H_flat + alpha*ctx.G_hat_anchor)
			flat_star = np.empty(screen_B, dtype=float)
			for b in range(screen_B):
				Fn_b, Gm_b = sample_ecdfs(F_flat)
				R_b = (Fn_b - alpha*Gm_b) / den
				flat_star[b], _ = _flat_tail_stat(ctx.grid_x, R_b, x0, ctx.x1)
			screen_flat_checks += 1
			raw_flat_tol = float(np.quantile(flat_star, q_flat, method='higher'))
			raw_flat_tail_p = float(np.mean(flat_star >= U_raw))
			raw_flat_ok = bool(U_raw > raw_flat_tol)
			tail_guard_ok = raw_flat_ok
		elif screen_tail_mode == "deterministic":
			U_raw = np.nan
			raw_flat_tol = np.nan
			raw_flat_tail_p = np.nan
			raw_flat_ok = bool(tail_diag['tail_guard_ok'])
			tail_guard_ok = raw_flat_ok
			screen_flat_checks += 1
		else:
			U_raw = np.nan
			raw_flat_tol = np.nan
			raw_flat_tail_p = np.nan
			raw_flat_ok = True
			tail_guard_ok = True

		mix_tail, alt_tail = ctx.tail_counts_at_pos(pos)
		screen_ok = bool(cdf_ok and tail_guard_ok and slope_ok)
		row = {
			'alpha': float(alpha),
			'pos': pos,
			'x0': float(x0),
			'mix_tail_count': int(mix_tail),
			'alt_tail_count': int(alt_tail),
			'screen_role': role,
			'screen_ok': screen_ok,
			'cdf_screen_ok': cdf_ok,
			'cdf_range_violation': float(cdf_obs['range']),
			'cdf_range_tol': cdf_range_tol,
			'cdf_range_tail_p': cdf_range_tail_p,
			'cdf_mon_violation': float(cdf_obs['mon']),
			'cdf_mon_tol': cdf_mon_tol,
			'cdf_mon_tail_p': cdf_mon_tail_p,
			'raw_flat_ok': raw_flat_ok,
			'raw_flat_stat': float(U_raw),
			'raw_flat_tol': raw_flat_tol,
			'raw_flat_tail_p': raw_flat_tail_p,
			'tail_guard_mode': screen_tail_mode,
			'tail_guard_ok': bool(tail_guard_ok),
			'tail_guard_x_left': float(tail_diag['tail_guard_x_left']),
			'tail_guard_left_mass': float(tail_diag['tail_guard_left_mass']),
			'tail_guard_right_mass': float(tail_diag['tail_guard_right_mass']),
			'tail_guard_left_slope': float(tail_diag['tail_guard_left_slope']),
			'tail_guard_right_slope': float(tail_diag['tail_guard_right_slope']),
			'tail_guard_slope_ratio': float(tail_diag['tail_guard_slope_ratio']),
			'tail_guard_equal_mass_x': float(tail_diag['tail_guard_equal_mass_x']),
			'slope_screen_ok': slope_ok,
			'slope_stat': slope_stat,
			'slope_tol': slope_tol,
			'slope_tail_p': slope_tail_p,
			'slope_blocks': active_slope_blocks_info,
			'slope_worst_blocks': slope_worst_blocks,
			'slope_worst_ratio': slope_worst_ratio,
			'slope_stats_by_block': [
				float(value) for value in slope_stats
			],
			'slope_tols_by_block': [
				float(value) for value in slope_tols
			],
			'slope_tail_ps_by_block': [
				float(value) for value in slope_tail_ps
			],
			'slope_ok_by_block': [
				bool(value) for value in slope_ok_by_scale
			],
			'observed_lcm_gap': float(T_obs),
		}
		if screen_tail_mode == "flat":
			tail_msg = f"RawFlat={_ok_text(raw_flat_ok)}, U={U_raw:.3g}/{raw_flat_tol:.3g}"
		elif screen_tail_mode == "deterministic":
			tail_msg = f"TailGuard={_ok_text(tail_guard_ok)}, ratio={tail_diag['tail_guard_slope_ratio']:.3g}"
		else:
			tail_msg = "TailGuard=SKIP"
		ctx.vprint(
			f"[{label}] alpha={alpha:.3f}, x0={x0:.3f}, "
			f"Screen={_ok_text(screen_ok)}, CDF={_ok_text(cdf_ok)} "
			f"range={cdf_obs['range']:.3g}/{cdf_range_tol:.3g}, "
			f"mon={cdf_obs['mon']:.3g}/{cdf_mon_tol:.3g}, "
			f"{tail_msg}, Slope={_ok_text(slope_ok)}, "
			f"S={slope_stat:.3g}/{slope_tol:.3g}@{slope_worst_blocks_text}"
		)
		return row, {'H_hat': H_hat, 'T_obs': float(T_obs)}

	def classify_screen_direction(screen):
		table = [] if screen is None else screen.get('screen_table', [])
		if not table:
			return {
				'direction': 'none',
				'n_direction_rows': 0,
				'range_blocked_frac': np.nan,
				'shape_blocked_frac': np.nan,
				'median_range_tail_p': np.nan,
				'median_slope_tail_p': np.nan,
			}
		range_p = np.asarray([row.get('cdf_range_tail_p', np.nan) for row in table], dtype=float)
		slope_p = np.asarray([row.get('slope_tail_p', np.nan) for row in table], dtype=float)
		range_valid = np.isfinite(range_p)
		slope_valid = np.isfinite(slope_p)
		if not np.any(range_valid) and any('proper_cdf_blocked' in row for row in table):
			range_blocked_rows = np.asarray(
				[row.get('proper_cdf_blocked', False) for row in table],
				dtype=bool
			)
			range_frac = float(np.mean(range_blocked_rows))
		else:
			range_frac = (
				float(np.mean(range_p[range_valid] <= screen_cdf_delta))
				if np.any(range_valid) else np.nan
			)
		if not np.any(slope_valid) and any('proper_shape_blocked' in row for row in table):
			shape_blocked_rows = np.asarray(
				[row.get('proper_shape_blocked', False) for row in table],
				dtype=bool
			)
			shape_frac = float(np.mean(shape_blocked_rows))
		else:
			shape_frac = (
				float(np.mean(slope_p[slope_valid] <= screen_slope_delta))
				if np.any(slope_valid) else np.nan
			)
		range_blocked = bool(np.isfinite(range_frac) and range_frac >= screen_direction_majority)
		shape_blocked = bool(np.isfinite(shape_frac) and shape_frac >= screen_direction_majority)
		if range_blocked and not shape_blocked:
			direction = 'range_blocked'
		elif shape_blocked and not range_blocked:
			direction = 'shape_blocked'
		elif range_blocked and shape_blocked:
			direction = 'mixed_blocked'
		else:
			direction = 'directionless'
		return {
			'direction': direction,
			'n_direction_rows': int(len(table)),
			'range_blocked_frac': range_frac,
			'shape_blocked_frac': shape_frac,
			'median_range_tail_p': (
				float(np.nanmedian(range_p)) if np.any(range_valid) else np.nan
			),
			'median_slope_tail_p': (
				float(np.nanmedian(slope_p)) if np.any(slope_valid) else np.nan
			),
		}

	def screen_search(alpha, label="screen", confirm_candidates=False,
			selected_confirm_candidates=False, stop_at_first_run=False):
		alpha = float(alpha)
		if alpha >= 1.0 - 1e-12:
			return {
				'success': False, 'status': 'alpha_too_close_to_one',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 0, 'passing_run': [], 'candidate_runs': [],
				'n_confirm_candidates': 0, 'candidate_confirmations': [],
				'confirmed': False, 'confirm': None, 'screen_table': []
			}
		den = max(1e-12, 1.0 - alpha)
		R_obs = (ctx.Fn - alpha*ctx.Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		run = []
		table = []
		fits = {}
		candidate_runs = []
		candidate_confirmations = []
		confirmed_result = None
		positions = screen_positions()

		def maybe_record_run():
			nonlocal confirmed_result
			if len(run) < screen_k_success:
				return None
			eligible_run = run[:screen_k_success]
			chosen_row = eligible_run[len(eligible_run)//2]
			chosen_pos = int(chosen_row['pos'])
			candidate = {
				'candidate_index': int(len(candidate_runs) + 1),
				'selected_pos': chosen_pos,
				'selected_x0': ctx.candidate_value(chosen_pos),
				'passing_run': eligible_run.copy(),
				'selected_screen': chosen_row,
				'selected_fit': fits[chosen_pos],
			}
			candidate_runs.append(candidate)
			ctx.vprint(
				f"[{label}-run #{len(candidate_runs)}] alpha={alpha:.6f}, "
				f"len={len(eligible_run)}, selected_pos={chosen_pos}, x0={ctx.candidate_value(chosen_pos):.6g}"
			)
			if confirm_candidates:
				confirm = confirm_at(
					alpha, chosen_pos,
					label=f"{label}-fresh-run{int(candidate['candidate_index'])}"
				)
				candidate_confirmations.append({
					'candidate_index': int(candidate['candidate_index']),
					'selected_pos': chosen_pos,
					'selected_x0': ctx.candidate_value(chosen_pos),
					'confirm': compact_confirm(confirm),
				})
				if confirm['full_ok']:
					confirmed_result = confirm
					return 'confirmed'
				if confirm.get('reason') == 'max_checks_reached':
					confirmed_result = confirm
					return 'max_checks_reached'
			elif selected_confirm_candidates:
				confirm_row, confirm_fit = screen_candidate(
					alpha, chosen_pos, R_obs, cdf_obs, den,
					label=f"{label}-fresh-run{int(candidate['candidate_index'])}-selected-confirm",
					q_cdf=q_screen_confirm_cdf,
					q_flat=q_screen_confirm_flat,
					q_slope=q_screen_confirm_slope,
					role="confirm",
				)
				candidate_confirmations.append({
					'candidate_index': int(candidate['candidate_index']),
					'selected_pos': chosen_pos,
					'selected_x0': ctx.candidate_value(chosen_pos),
					'screen_confirm': confirm_row,
				})
				if confirm_row['screen_ok']:
					candidate['selected_screen'] = confirm_row
					candidate['selected_fit'] = confirm_fit
					candidate['confirm_screen'] = confirm_row
					confirmed_result = {
						'full_ok': True,
						'pos': chosen_pos,
						'x0': ctx.candidate_value(chosen_pos),
						'reason': 'selected_screen_confirmed',
					}
					return 'selected_confirmed'
			if stop_at_first_run:
				return 'screen_run_found'
			if len(candidate_runs) >= screen_max_confirm_candidates:
				return 'candidate_cap'
			return None

		stop_reason = None
		for pos in positions:
			row, fit = screen_candidate(alpha, pos, R_obs, cdf_obs, den, label=f"{label}-candidate")
			table.append(row)
			fits[int(pos)] = fit
			if row['screen_ok']:
				run.append(row)
				if len(run) >= screen_k_success:
					stop_reason = maybe_record_run()
					if stop_reason is not None:
						break
					run = []
			else:
				run = []
		else:
			stop_reason = 'x_G_reached'

		if candidate_runs:
			if confirmed_result is not None and confirmed_result.get('full_ok', False):
				selected = next(
					candidate for candidate in candidate_runs
					if int(candidate['selected_pos']) == int(confirmed_result['pos'])
				)
				status = (
					'screen_selected_confirmed'
					if selected_confirm_candidates else 'screen_confirmed'
				)
			else:
				selected = candidate_runs[0]
				status = (
					'max_checks_reached'
					if stop_reason == 'max_checks_reached'
					else 'screen_selected_confirm_candidates_failed'
					if selected_confirm_candidates
					else 'screen_confirm_candidates_failed'
					if confirm_candidates else 'screen_runs_found'
				)
			success = bool(
				(confirmed_result is not None and confirmed_result.get('full_ok', False))
				if selected_confirm_candidates else True
			)
			return {
				'success': success,
				'status': status,
				'stop_reason': stop_reason,
				'selected_pos': int(selected['selected_pos']),
				'selected_x0': float(selected['selected_x0']),
				'n_screened': int(len(table)),
				'passing_run': selected['passing_run'],
				'selected_screen': selected['selected_screen'],
				'selected_fit': selected['selected_fit'],
				'candidate_runs': candidate_runs,
				'n_confirm_candidates': int(len(candidate_runs)),
				'candidate_confirmations': candidate_confirmations,
				'confirmed': bool(confirmed_result is not None and confirmed_result.get('full_ok', False)),
				'confirm': confirmed_result,
				'screen_table': table,
			}
		return {
			'success': False,
			'status': 'no_screen_passing_run',
			'stop_reason': stop_reason,
			'selected_pos': None,
			'selected_x0': None,
			'n_screened': int(len(table)),
			'passing_run': [],
			'selected_screen': None,
			'selected_fit': None,
			'candidate_runs': [],
			'n_confirm_candidates': 0,
			'candidate_confirmations': [],
			'confirmed': False,
			'confirm': None,
			'screen_table': table,
		}

	def proper_search(alpha, label="proper"):
		nonlocal proper_candidate_checks
		alpha = float(alpha)
		if alpha >= 1.0 - 1e-12:
			return {
				'success': False, 'status': 'alpha_too_close_to_one',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 0, 'passing_run': [], 'candidate_runs': [],
				'n_confirm_candidates': 0, 'candidate_confirmations': [],
				'confirmed': False, 'confirm': None, 'screen_table': []
			}
		run = []
		table = []
		positions = screen_positions()
		stop_reason = None
		for pos in positions:
			if ctx.total_alpha_checks() >= max_checks:
				stop_reason = 'max_checks_reached'
				break
			proper_candidate_checks += 1
			confirm = confirm_at(alpha, int(pos), label=f"{label}-candidate")
			shape_diag = confirm.get('shape_diag') or {}
			cdf_diag = confirm.get('cdf_diag') or {}
			row = {
				'alpha': float(alpha),
				'pos': int(pos),
				'x0': ctx.candidate_value(pos),
				'screen_role': 'proper',
				'screen_ok': bool(confirm.get('full_ok', False)),
				'proper_ok': bool(confirm.get('full_ok', False)),
				'proper_shape_ok': bool(confirm.get('shape_ok', False)),
				'proper_confirm_shape_ok': bool(confirm.get('confirm_shape_ok', False)),
				'proper_cdf_ok': bool(confirm.get('cdf_ok', False)),
				'proper_shape_blocked': not bool(confirm.get('confirm_shape_ok', False)),
				'proper_cdf_blocked': not bool(confirm.get('cdf_ok', False)),
				'cdf_screen_ok': bool(confirm.get('cdf_ok', False)),
				'cdf_range_violation': float(cdf_diag.get('cdf_range_violation', np.nan)),
				'cdf_range_tol': float(cdf_diag.get('cdf_range_tol', np.nan)),
				'cdf_mon_violation': float(cdf_diag.get('cdf_mon_violation', np.nan)),
				'cdf_mon_tol': float(cdf_diag.get('cdf_mon_tol', np.nan)),
				'raw_flat_ok': bool(shape_diag.get('flat_reject', True) if require_flat_tail_reject else True),
				'raw_flat_stat': float(shape_diag.get('flat_stat', np.nan)),
				'raw_flat_tol': float(shape_diag.get('flat_tol', np.nan)),
				'slope_screen_ok': bool(confirm.get('shape_ok', False)),
				'observed_lcm_gap': float(shape_diag.get('shape_gap', np.nan)),
				'observed_lcm_tol': float(shape_diag.get('shape_tol', np.nan)),
				'mix_tail_count': int(shape_diag.get('mix_tail_count', 0)),
				'alt_tail_count': int(shape_diag.get('alt_tail_count', 0)),
				'confirm': confirm,
			}
			table.append(row)
			ctx.vprint(
				f"[{label}-candidate] alpha={alpha:.6f}, pos={int(pos)}, "
				f"x0={ctx.candidate_value(pos):.6g}, "
				f"Shape={_ok_text(confirm.get('shape_ok', False))}, "
				f"Flat={_ok_text(row['raw_flat_ok'])}, "
				f"CDF={_ok_text(confirm.get('cdf_ok', False))}, "
				f"Full={_ok_text(confirm.get('full_ok', False))}"
			)
			if confirm.get('reason') == 'max_checks_reached':
				stop_reason = 'max_checks_reached'
				break
			if confirm.get('full_ok', False):
				run.append(row)
				if len(run) >= proper_k_success:
					eligible_run = run[:proper_k_success]
					chosen_row = eligible_run[len(eligible_run)//2]
					chosen_pos = int(chosen_row['pos'])
					chosen_confirm = chosen_row['confirm']
					selected_fit = {
						'H_hat': chosen_confirm['shape_diag'].get('H_hat')
						if chosen_confirm and chosen_confirm.get('shape_diag') else None,
						'T_obs': (
							chosen_confirm['shape_diag'].get('shape_gap', np.nan)
							if chosen_confirm and chosen_confirm.get('shape_diag') else np.nan
						),
					}
					candidate = {
						'candidate_index': 1,
						'selected_pos': chosen_pos,
						'selected_x0': ctx.candidate_value(chosen_pos),
						'passing_run': eligible_run.copy(),
						'selected_screen': chosen_row,
						'selected_fit': selected_fit,
					}
					return {
						'success': True,
						'status': 'proper_confirmed_run',
						'stop_reason': 'proper_run_found',
						'selected_pos': chosen_pos,
						'selected_x0': ctx.candidate_value(chosen_pos),
						'n_screened': int(len(table)),
						'passing_run': eligible_run,
						'selected_screen': chosen_row,
						'selected_fit': selected_fit,
						'candidate_runs': [candidate],
						'n_confirm_candidates': 1,
						'candidate_confirmations': [{
							'candidate_index': 1,
							'selected_pos': chosen_pos,
							'selected_x0': ctx.candidate_value(chosen_pos),
							'confirm': compact_confirm(chosen_confirm),
						}],
						'confirmed': True,
						'confirm': chosen_confirm,
						'screen_table': table,
					}
			else:
				run = []
		if stop_reason is None:
			stop_reason = 'x_G_reached'
		return {
			'success': False,
			'status': 'no_proper_passing_run',
			'stop_reason': stop_reason,
			'selected_pos': None,
			'selected_x0': None,
			'n_screened': int(len(table)),
			'passing_run': [],
			'selected_screen': None,
			'selected_fit': None,
			'candidate_runs': [],
			'n_confirm_candidates': 0,
			'candidate_confirmations': [],
			'confirmed': False,
			'confirm': None,
			'screen_table': table,
		}

	def screen_existing_run(alpha, passing_run, label="screen-block"):
		alpha = float(alpha)
		if not passing_run or alpha >= 1.0 - 1e-12:
			return {
				'success': False, 'status': 'inherited_screen_unavailable',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 0, 'passing_run': [],
				'candidate_runs': [], 'n_confirm_candidates': 0,
				'candidate_confirmations': [], 'confirmed': False,
				'confirm': None, 'screen_table': []
			}
		den = max(1e-12, 1.0 - alpha)
		R_obs = (ctx.Fn - alpha*ctx.Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		table = []
		fits = {}
		for old_row in passing_run:
			pos = int(old_row['pos'])
			row, fit = screen_candidate(alpha, pos, R_obs, cdf_obs, den, label=f"{label}-candidate")
			table.append(row)
			fits[pos] = fit
		if len(table) < screen_k_success or not all(row['screen_ok'] for row in table):
			return {
				'success': False, 'status': 'inherited_screen_block_failed',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': int(len(table)), 'passing_run': table,
				'candidate_runs': [], 'n_confirm_candidates': 0,
				'candidate_confirmations': [], 'confirmed': False,
				'confirm': None, 'screen_table': table,
			}
		eligible_run = table[:screen_k_success]
		chosen_row = eligible_run[len(eligible_run)//2]
		chosen_pos = int(chosen_row['pos'])
		candidate = {
			'candidate_index': 1,
			'selected_pos': chosen_pos,
			'selected_x0': ctx.candidate_value(chosen_pos),
			'passing_run': eligible_run.copy(),
			'selected_screen': chosen_row,
			'selected_fit': fits[chosen_pos],
		}
		return {
			'success': True,
			'status': 'inherited_screen_block_passes',
			'stop_reason': 'inherited_screen',
			'selected_pos': chosen_pos,
			'selected_x0': ctx.candidate_value(chosen_pos),
			'n_screened': int(len(table)),
			'passing_run': eligible_run,
			'selected_screen': chosen_row,
			'selected_fit': fits[chosen_pos],
			'candidate_runs': [candidate],
			'n_confirm_candidates': 1,
			'candidate_confirmations': [],
			'confirmed': False,
			'confirm': None,
			'screen_table': table,
		}

	def screen_selected_cutoff(alpha, inherited_screen, label="screen-selected"):
		alpha = float(alpha)
		if inherited_screen is None or alpha >= 1.0 - 1e-12:
			return {
				'success': False, 'status': 'inherited_selected_unavailable',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 0, 'passing_run': [],
				'candidate_runs': [], 'n_confirm_candidates': 0,
				'candidate_confirmations': [], 'confirmed': False,
				'confirm': None, 'screen_table': []
			}
		pos = inherited_screen.get('selected_pos')
		if pos is None:
			selected_screen = inherited_screen.get('selected_screen')
			if isinstance(selected_screen, dict):
				pos = selected_screen.get('pos')
		if pos is None:
			return {
				'success': False, 'status': 'inherited_selected_unavailable',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 0, 'passing_run': [],
				'candidate_runs': [], 'n_confirm_candidates': 0,
				'candidate_confirmations': [], 'confirmed': False,
				'confirm': None, 'screen_table': []
			}
		pos = int(pos)
		den = max(1e-12, 1.0 - alpha)
		R_obs = (ctx.Fn - alpha*ctx.Gm) / den
		cdf_obs = _cdf_violation_stats(R_obs)
		row, fit = screen_candidate(
			alpha, pos, R_obs, cdf_obs, den,
			label=f"{label}-candidate",
			q_cdf=q_screen_confirm_cdf,
			q_flat=q_screen_confirm_flat,
			q_slope=q_screen_confirm_slope,
			role="confirm",
		)
		if not row['screen_ok']:
			return {
				'success': False, 'status': 'inherited_selected_confirm_failed',
				'selected_pos': None, 'selected_x0': None,
				'n_screened': 1, 'passing_run': [row],
				'candidate_runs': [], 'n_confirm_candidates': 0,
				'candidate_confirmations': [], 'confirmed': False,
				'confirm': None, 'screen_table': [row],
			}
		candidate = {
			'candidate_index': 1,
			'selected_pos': pos,
			'selected_x0': ctx.candidate_value(pos),
			'passing_run': [row],
			'selected_screen': row,
			'selected_fit': fit,
		}
		return {
			'success': True,
			'status': 'inherited_selected_confirm_passes',
			'stop_reason': 'inherited_selected',
			'selected_pos': pos,
			'selected_x0': ctx.candidate_value(pos),
			'n_screened': 1,
			'passing_run': [row],
			'selected_screen': row,
			'selected_fit': fit,
			'candidate_runs': [candidate],
			'n_confirm_candidates': 1,
			'candidate_confirmations': [],
			'confirmed': False,
			'confirm': None,
			'screen_table': [row],
		}

	def raw_flat_confirm_at(alpha, pos, shape_diag, label="raw-flat-confirm"):
		alpha = float(alpha)
		pos = int(pos)
		x0 = ctx.candidate_value(pos)
		den = max(1e-12, 1.0 - alpha)
		R_obs = shape_diag.get('R_obs')
		if R_obs is None:
			R_obs = (ctx.Fn - alpha*ctx.Gm) / den
		H_hat = shape_diag.get('H_hat')
		if H_hat is None:
			H_hat = _as_valid_cdf(_concave_projection_on_interval(ctx.grid_x, R_obs, x0, ctx.x1))
		U_obs, _ = _flat_tail_stat(ctx.grid_x, R_obs, x0, ctx.x1)
		_, H_flat = _flat_tail_stat(ctx.grid_x, H_hat, x0, ctx.x1)
		H_flat = _as_valid_cdf(H_flat)
		F_flat = _as_valid_cdf((1.0 - alpha)*H_flat + alpha*ctx.G_hat_anchor)
		flat_star = np.empty(confirm_B, dtype=float)
		for b in range(confirm_B):
			x_mix_b = _sample_from_cdf(ctx.grid_x, F_flat, ctx.n, ctx.rng)
			x_mix_b.sort(kind='mergesort')
			x_alt_b = _sample_from_cdf(ctx.grid_x, ctx.G_hat_anchor, ctx.m, ctx.rng)
			x_alt_b.sort(kind='mergesort')
			Fn_b = _ecdf_on_grid(x_mix_b, ctx.grid_x)
			Gm_b = _ecdf_on_grid(x_alt_b, ctx.grid_x)
			R_b = (Fn_b - alpha*Gm_b) / den
			flat_star[b], _ = _flat_tail_stat(ctx.grid_x, R_b, x0, ctx.x1)
		flat_tol = float(np.quantile(flat_star, q_pava_confirm_flat, method='higher'))
		flat_reject = bool(U_obs > flat_tol)
		flat_diag = {
			'flat_available': True,
			'flat_reject': flat_reject,
			'flat_stat': float(U_obs),
			'flat_tol': flat_tol,
			'flat_delta': float(confirm_flat_delta),
			'flat_bootstrap_quantile': float(q_pava_confirm_flat),
			'flat_reason': 'raw_confirm',
			'flat_confirm_B': int(confirm_B),
			'H_flat': H_flat,
		}
		shape_diag.update(flat_diag)
		ctx.vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"U={U_obs:.3g}/{flat_tol:.3g}, RawFlat={_ok_text(flat_reject)}"
		)
		return flat_diag

	def deterministic_tail_confirm_at(alpha, pos, shape_diag, label="tail-confirm"):
		alpha = float(alpha)
		pos = int(pos)
		x0 = ctx.candidate_value(pos)
		den = max(1e-12, 1.0 - alpha)
		R_obs = shape_diag.get('R_obs')
		if R_obs is None:
			R_obs = (ctx.Fn - alpha*ctx.Gm) / den
		tail_diag = _raw_tail_guard_diag(ctx.grid_x, R_obs, x0, ctx.x1)
		tail_ok = bool(tail_diag['tail_guard_ok'])
		flat_diag = {
			'flat_available': True,
			'flat_reject': tail_ok,
			'flat_stat': float(tail_diag['tail_guard_slope_ratio']),
			'flat_tol': 1.0,
			'flat_delta': np.nan,
			'flat_bootstrap_quantile': np.nan,
			'flat_reason': 'deterministic_raw_tail_guard',
			'flat_confirm_B': 0,
		}
		flat_diag.update(tail_diag)
		shape_diag.update(flat_diag)
		ctx.vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={x0:.6g}, "
			f"ratio={tail_diag['tail_guard_slope_ratio']:.3g}, "
			f"TailGuard={_ok_text(tail_ok)}"
		)
		return flat_diag

	def confirm_at(alpha, pos, label="confirm"):
		if ctx.total_alpha_checks() >= max_checks:
			return {
				'alpha': float(alpha), 'pos': int(pos), 'x0': ctx.candidate_value(pos),
				'full_ok': False, 'shape_ok': False, 'confirm_shape_ok': False,
				'cdf_ok': False, 'reason': 'max_checks_reached',
				'shape_diag': None, 'cdf_diag': None,
			}
		ctx.alpha_checks += 1
		shape_diag = dict(ctx.shape_r(alpha, pos, label=f"{label}-shape"))
		confirm_shape_ok = bool(shape_diag.get('shape_ok', False))
		if confirm_tail_mode == "flat" and confirm_mode == "full":
			flat_diag = ctx.flat_reject(alpha, pos, shape_diag, label=f"{label}-flat")
			confirm_shape_ok = bool(confirm_shape_ok and flat_diag.get('flat_reject', False))
		elif confirm_tail_mode == "flat" and confirm_mode == "pava_shape_cdf":
			flat_diag = raw_flat_confirm_at(alpha, pos, shape_diag, label=f"{label}-raw-flat")
			confirm_shape_ok = bool(confirm_shape_ok and flat_diag.get('flat_reject', False))
		elif confirm_tail_mode == "deterministic":
			flat_diag = deterministic_tail_confirm_at(alpha, pos, shape_diag, label=f"{label}-tail")
			confirm_shape_ok = bool(confirm_shape_ok and flat_diag.get('flat_reject', False))
		cdf_diag = ctx.cdf_at(alpha, pos, alpha_ref=alpha, label=f"{label}-cdf")
		full_ok = bool(confirm_shape_ok and cdf_diag.get('cdf_ok', False))
		ctx.record_shape_ref(alpha, pos, shape_diag, source=label)
		ctx.vprint(
			f"[{label}] alpha={alpha:.6f}, pos={pos}, x0={ctx.candidate_value(pos):.6g}, "
			f"Shape={_ok_text(shape_diag.get('shape_ok', False))}, "
			f"Tail={_ok_text(shape_diag.get('flat_reject', True) if require_flat_tail_reject else True)}, "
			f"CDF={_ok_text(cdf_diag.get('cdf_ok', False))}, Full={_ok_text(full_ok)}"
		)
		return {
			'alpha': float(alpha),
			'pos': int(pos),
			'x0': ctx.candidate_value(pos),
			'full_ok': full_ok,
			'shape_ok': bool(shape_diag.get('shape_ok', False)),
			'confirm_shape_ok': bool(confirm_shape_ok),
			'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
			'shape_diag': shape_diag,
			'cdf_diag': cdf_diag,
			'reason': 'confirmed' if full_ok else 'confirmation_failed',
		}

	def compact_confirm(result):
		if result is None:
			return None
		return {
			'alpha': float(result['alpha']),
			'pos': None if result.get('pos') is None else int(result['pos']),
			'x0': result.get('x0'),
			'full_ok': bool(result.get('full_ok', False)),
			'shape_ok': bool(result.get('shape_ok', False)),
			'confirm_shape_ok': bool(result.get('confirm_shape_ok', False)),
			'cdf_ok': bool(result.get('cdf_ok', False)),
			'reason': result.get('reason'),
			'shape_diag': ctx.compact_shape(result.get('shape_diag')),
			'cdf_diag': ctx.compact_cdf(result.get('cdf_diag')) if result.get('cdf_diag') is not None else None,
		}

	def evaluate_alpha(alpha, inherited_pos=None, inherited_result=None, label="alpha"):
		alpha = float(alpha)
		if confirm_mode == "screen_only" and inherited_result is not None and inherited_result.get('screen') is not None:
			if screen_confirm_mode == "selected":
				ctx.vprint(
					f"[{label}] alpha={alpha:.6f}, trying inherited selected cutoff"
				)
				screen = screen_selected_cutoff(
					alpha,
					inherited_result['screen'],
					label=f"{label}-inherited-selected"
				)
			else:
				ctx.vprint(
					f"[{label}] alpha={alpha:.6f}, trying inherited screen block"
				)
				screen = screen_existing_run(
					alpha,
					inherited_result['screen'].get('passing_run', []),
					label=f"{label}-inherited-screen"
				)
			if screen['success']:
				direction = (
					{'direction': 'selected_confirmed', 'n_direction_rows': 1}
					if screen_confirm_mode == "selected"
					else classify_screen_direction(screen)
				)
				return {
					'alpha': alpha, 'ok': True, 'source': 'inherited_screen',
					'pos': int(screen['selected_pos']), 'screen': screen,
					'confirm': None, 'reason': 'screen_only',
					'selected_fit': screen.get('selected_fit'),
					'direction': direction,
				}
		if confirm_mode in {"full", "pava_shape_cdf"} and inherited_pos is not None:
			ctx.vprint(
				f"[{label}] alpha={alpha:.6f}, trying inherited pos={int(inherited_pos)}, "
				f"x0={ctx.candidate_value(inherited_pos):.6g}"
			)
			inherited_confirm = confirm_at(alpha, int(inherited_pos), label=f"{label}-inherited")
			if inherited_confirm['full_ok']:
				return {
					'alpha': alpha, 'ok': True, 'source': 'inherited',
					'pos': int(inherited_pos), 'screen': None,
					'confirm': inherited_confirm,
				}
			if inherited_confirm.get('reason') == 'max_checks_reached':
				return {
					'alpha': alpha, 'ok': False, 'source': 'inherited',
					'pos': int(inherited_pos), 'screen': None,
					'confirm': inherited_confirm, 'reason': 'max_checks_reached',
				}
		if locate_mode == "proper":
			screen = proper_search(alpha, label=f"{label}-proper")
		else:
			screen = screen_search(
				alpha,
				label=f"{label}-screen",
				confirm_candidates=(confirm_mode in {"full", "pava_shape_cdf"}),
				selected_confirm_candidates=(
					confirm_mode == "screen_only" and screen_confirm_mode == "selected"
				),
				stop_at_first_run=(
					confirm_mode == "screen_only" and screen_confirm_mode == "run"
				)
			)
		direction = classify_screen_direction(screen)
		if not screen['success']:
			return {
				'alpha': alpha, 'ok': False, 'source': 'screen',
				'pos': None, 'screen': screen, 'confirm': None,
				'reason': screen['status'], 'direction': direction,
			}
		if confirm_mode == "screen_only":
			return {
				'alpha': alpha, 'ok': True, 'source': 'fresh_screen',
				'pos': int(screen['selected_pos']), 'screen': screen,
				'confirm': None, 'reason': 'screen_only',
				'selected_fit': screen.get('selected_fit'),
				'direction': direction,
			}
		if screen.get('confirmed', False):
			confirm = screen['confirm']
			return {
				'alpha': alpha, 'ok': True, 'source': 'fresh_screen',
				'pos': int(confirm['pos']), 'screen': screen,
				'confirm': confirm, 'reason': confirm['reason'],
				'direction': direction,
			}
		return {
			'alpha': alpha, 'ok': False, 'source': 'fresh_screen',
			'pos': None, 'screen': screen,
			'confirm': screen.get('confirm'),
			'reason': (
				'max_checks_reached'
				if screen.get('status') == 'max_checks_reached'
				else screen.get('status', 'screen_confirm_candidates_failed')
			),
			'direction': direction,
		}

	def alpha_eval_record(result):
		screen = result.get('screen')
		confirm = result.get('confirm')
		return {
			'alpha': float(result['alpha']),
			'ok': bool(result.get('ok', False)),
			'source': result.get('source'),
			'pos': None if result.get('pos') is None else int(result['pos']),
			'x0': None if result.get('pos') is None else ctx.candidate_value(result['pos']),
			'reason': result.get('reason'),
			'direction': result.get('direction'),
			'screen_status': None if screen is None else screen.get('status'),
			'n_screened': None if screen is None else int(screen.get('n_screened', 0)),
			'n_confirm_candidates': None if screen is None else int(screen.get('n_confirm_candidates', 0)),
			'confirm': compact_confirm(confirm),
		}

	def make_info(success, reason, result=None, alpha_lo=None, alpha_hi=None, extra=None):
		confirm = None if result is None else result.get('confirm')
		shape_diag = None if confirm is None else confirm.get('shape_diag')
		cdf_diag = None if confirm is None else confirm.get('cdf_diag')
		pos = None if result is None else result.get('pos')
		info = {
			'success': bool(success),
			'reason': reason,
			'alpha': np.nan if result is None or not success else float(result['alpha']),
			'x0': None if pos is None else ctx.candidate_value(pos),
			'alpha_lo': None if alpha_lo is None else float(alpha_lo),
			'alpha_hi': None if alpha_hi is None else float(alpha_hi),
			'max_checks': int(max_checks),
			'confirm_mode': confirm_mode,
			'confirm_tail_mode': confirm_tail_mode,
			'locate_mode': locate_mode,
			'screen_confirm_mode': screen_confirm_mode,
			'screen_bracket_mode': screen_bracket_mode,
			'screen_direction_majority': float(screen_direction_majority),
			'screen_B': int(screen_B),
			'screen_delta': float(screen_delta),
			'screen_cdf_delta': float(screen_cdf_delta),
			'screen_flat_delta': float(screen_flat_delta),
			'screen_tail_mode': screen_tail_mode,
			'screen_slope_delta': float(screen_slope_delta),
			'screen_confirm_cdf_delta': float(screen_confirm_cdf_delta),
			'screen_confirm_flat_delta': float(screen_confirm_flat_delta),
			'screen_confirm_slope_delta': float(screen_confirm_slope_delta),
			'screen_confirm_slope_blocks': screen_confirm_slope_blocks_for_info,
			'confirm_B': int(confirm_B),
			'confirm_cdf_B': int(confirm_cdf_B),
			'confirm_delta': float(confirm_delta),
			'confirm_cdf_delta': float(confirm_cdf_delta),
			'confirm_flat_delta': float(confirm_flat_delta),
			'screen_cdf_quantile': float(q_screen_cdf),
			'screen_flat_quantile': float(q_screen_flat),
			'screen_slope_quantile': float(q_screen_slope),
			'screen_confirm_cdf_quantile': float(q_screen_confirm_cdf),
			'screen_confirm_flat_quantile': float(q_screen_confirm_flat),
			'screen_confirm_slope_quantile': float(q_screen_confirm_slope),
			'pava_shape_cdf_raw_flat_quantile': float(q_pava_confirm_flat),
			'screen_k_success': int(screen_k_success),
			'proper_k_success': int(proper_k_success),
			'screen_max_cutoffs': None if screen_max_cutoffs is None else int(screen_max_cutoffs),
			'screen_max_confirm_candidates': int(screen_max_confirm_candidates),
			'screen_cutoff_spacing': 'empirical_quantile',
			'slope_blocks': slope_blocks_for_info,
			'slope_block_spacing': 'empirical_quantile',
			'screen_candidate_checks': int(screen_candidate_checks),
			'screen_cdf_checks': int(screen_cdf_checks),
			'screen_flat_checks': int(screen_flat_checks),
			'screen_slope_checks': int(screen_slope_checks),
			'proper_candidate_checks': int(proper_candidate_checks),
			'alpha_eval_table': alpha_eval_table,
		}
		info.update(ctx.base_info(mode))
		if result is not None and result.get('direction') is not None:
			info['final_direction'] = result.get('direction')
		if confirm is None and result is not None and bool(success) and confirm_mode == "screen_only":
			selected_fit = result.get('selected_fit')
			selected_screen = None
			if result.get('screen') is not None:
				selected_screen = result['screen'].get('selected_screen')
			info.update({
				'screen_only_ok': True,
				'full_ok': False,
				'residual_lcm_gap': float(
					selected_fit.get('T_obs', np.nan)
					if selected_fit is not None else
					(selected_screen or {}).get('observed_lcm_gap', np.nan)
				),
				'residual_lcm_tol': np.nan,
			})
		if shape_diag is not None:
			info.update({
				'shape_ok': bool(shape_diag.get('shape_ok', False)),
				'confirm_shape_ok': bool(confirm.get('confirm_shape_ok', False)),
				'residual_lcm_gap': float(shape_diag['shape_gap']),
				'residual_lcm_tol': float(shape_diag['shape_tol']),
				'mix_tail_count': int(shape_diag['mix_tail_count']),
				'alt_tail_count': int(shape_diag['alt_tail_count']),
			})
			if require_flat_tail_reject:
				info.update({
					'flat_available': bool(shape_diag.get('flat_available', False)),
					'flat_reject': bool(shape_diag.get('flat_reject', False)),
					'flat_stat': float(shape_diag.get('flat_stat', np.nan)),
					'flat_tol': float(shape_diag.get('flat_tol', np.nan)),
				})
		if cdf_diag is not None:
			info.update({
				'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
				'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
				'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
				'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
				'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
				'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
				'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
				'full_ok': bool(confirm.get('full_ok', False)) if confirm is not None else False,
			})
		if result is not None and result.get('screen') is not None:
			screen = result['screen']
			info.update({
				'final_screen_status': screen.get('status'),
				'final_screen_stop_reason': screen.get('stop_reason'),
				'final_n_screened': int(screen.get('n_screened', 0)),
				'final_n_confirm_candidates': int(screen.get('n_confirm_candidates', 0)),
				'final_screen_passing_run': screen.get('passing_run', []),
				'final_selected_screen': screen.get('selected_screen'),
				'final_screen_candidate_runs': [
					{
						'candidate_index': int(candidate['candidate_index']),
						'selected_pos': int(candidate['selected_pos']),
						'selected_x0': float(candidate['selected_x0']),
						'passing_run': candidate.get('passing_run', []),
						'selected_screen': candidate.get('selected_screen'),
					}
					for candidate in screen.get('candidate_runs', [])
				],
				'final_screen_candidate_confirmations': screen.get('candidate_confirmations', []),
				'final_screen_table': screen.get('screen_table', []),
			})
		if extra:
			info.update(extra)
		return info

	probe_alphas = np.linspace(ctx.alpha_left, ctx.alpha_right, num=ctx.n_alpha_probes)
	last_fail = None
	first_pass = None
	directional_upper = None
	initial_lo_alpha = None
	for alpha_probe in probe_alphas:
		if ctx.total_alpha_checks() >= max_checks:
			break
		result = evaluate_alpha(float(alpha_probe), inherited_pos=None, label="screen-probe2")
		alpha_eval_table.append(alpha_eval_record(result))
		if result['ok']:
			first_pass = result
			ctx.vprint(
				f"[screen-bracket2] first confirmed pass alpha={float(alpha_probe):.6f}, "
				f"x0={ctx.candidate_value(result['pos']):.6g}, source={result['source']}"
			)
			break
		direction = (result.get('direction') or {}).get('direction')
		if screen_bracket_mode == "directional" and direction == 'range_blocked' and last_fail is not None:
			directional_upper = result
			ctx.vprint(
				f"[screen-bracket2] range-blocked alpha={float(alpha_probe):.6f}; "
				f"refining interval ({float(last_fail['alpha']):.6f}, {float(alpha_probe):.6f})"
			)
			break
		last_fail = result
	if first_pass is None:
		if screen_bracket_mode == "directional" and directional_upper is not None and last_fail is not None:
			lo_alpha_dir = float(last_fail['alpha'])
			hi_alpha_dir = float(directional_upper['alpha'])
			directional_iter = 0
			while (hi_alpha_dir - lo_alpha_dir) > ctx.alpha_tol and ctx.total_alpha_checks() < max_checks:
				directional_iter += 1
				mid = 0.5 * (lo_alpha_dir + hi_alpha_dir)
				ctx.vprint(
					f"[screen-directional-refine2 #{directional_iter:02d}] "
					f"lo={lo_alpha_dir:.6f}, hi={hi_alpha_dir:.6f}, mid={mid:.6f}"
				)
				result = evaluate_alpha(mid, inherited_pos=None, label="screen-directional2")
				alpha_eval_table.append(alpha_eval_record(result))
				if result['ok']:
					first_pass = result
					initial_lo_alpha = lo_alpha_dir
					ctx.vprint(
						f"[screen-directional-refine2] first confirmed pass alpha={mid:.6f}, "
						f"x0={ctx.candidate_value(result['pos']):.6g}, source={result['source']}"
					)
					break
				direction = (result.get('direction') or {}).get('direction')
				if direction == 'range_blocked':
					hi_alpha_dir = mid
				else:
					lo_alpha_dir = mid
					last_fail = result
			if first_pass is None:
				info = make_info(
					False,
					'no_screened_confirmed_alpha_in_directional_bracket',
					result=last_fail,
					alpha_lo=lo_alpha_dir,
					alpha_hi=hi_alpha_dir,
					extra={'probe_alphas': probe_alphas, 'directional_upper': alpha_eval_record(directional_upper)}
				)
				return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info
		else:
			info = make_info(
				False,
				'no_screened_confirmed_alpha_in_probe_grid',
				result=last_fail,
				extra={'probe_alphas': probe_alphas}
			)
			return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info
	if last_fail is None:
		info = make_info(
			True, 'left_endpoint_screened_confirmed',
			result=first_pass,
			alpha_lo=ctx.alpha_left,
			alpha_hi=first_pass['alpha'],
			extra={'probe_alphas': probe_alphas, 'shape_boundary': {'status': 'left_endpoint_passing', 'lo': ctx.alpha_left, 'hi': first_pass['alpha'], 'width': 0.0}}
		)
		if confirm_mode == "screen_only":
			return float(first_pass['alpha']), info['x0'], first_pass['selected_fit']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		confirm = first_pass['confirm']
		return float(first_pass['alpha']), info['x0'], confirm['shape_diag']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info

	lo_alpha = float(initial_lo_alpha) if initial_lo_alpha is not None else float(last_fail['alpha'])
	hi_alpha = float(first_pass['alpha'])
	hi_result = first_pass
	bisect_iter = 0
	status = 'alpha_tol_reached'
	while (hi_alpha - lo_alpha) > ctx.alpha_tol and ctx.total_alpha_checks() < max_checks:
		bisect_iter += 1
		mid = 0.5 * (lo_alpha + hi_alpha)
		ctx.vprint(
			f"[screen-alpha-bisect2 #{bisect_iter:02d}] lo={lo_alpha:.6f}, "
			f"hi={hi_alpha:.6f}, mid={mid:.6f}"
		)
		result = evaluate_alpha(
			mid,
			inherited_pos=hi_result['pos'],
			inherited_result=hi_result,
			label="screen-bisect2"
		)
		alpha_eval_table.append(alpha_eval_record(result))
		if result['ok']:
			hi_alpha = mid
			hi_result = result
		else:
			lo_alpha = mid
	if (hi_alpha - lo_alpha) > ctx.alpha_tol:
		status = 'max_checks_reached'

	shape_boundary = {
		'status': status,
		'lo': float(lo_alpha),
		'hi': float(hi_alpha),
		'width': float(hi_alpha - lo_alpha),
	}
	info = make_info(
		True, 'screened_confirmed_alpha',
		result=hi_result,
		alpha_lo=lo_alpha,
		alpha_hi=hi_alpha,
		extra={
			'probe_alphas': probe_alphas,
			'shape_boundary': shape_boundary,
			'alpha_shape': float(hi_result['alpha']),
			'alpha_cdf': float(hi_result['alpha']),
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'screened_confirmed',
		}
	)
	confirm = hi_result['confirm']
	ctx.vprint(
		f"[done-screened-x0-2] alpha_hat={float(hi_result['alpha']):.6f}, "
		f"x0={info['x0']:.6g}, source={hi_result['source']}, reason={info['reason']}"
	)
	if confirm_mode == "screen_only":
		return float(hi_result['alpha']), info['x0'], hi_result['selected_fit']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
	return float(hi_result['alpha']), info['x0'], confirm['shape_diag']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info


def alpha_minimize_derived_x0_revisited(
	x_mix,
	x_alt,
	alpha_tol=1e-4,
	alpha_bounds=(0.0, 0.999999),
	n_alpha_probes=9,
	B=100,
	delta=0.05,
	cdf_delta=None,
	cdf_B=None,
	flat_delta=None,
	min_mix_tail=30,
	min_alt_tail=30,
	max_checks=64,
	random_state=None,
	grid=None,
	candidate_cutoffs=None,
	max_grid_points=5000,
	verbose=False,
	x0_tol=0.0,
	shape_weight_gamma=2.0,
	gap_tol=None,
	near_miss_probes=5,
	cdf_ref_max_dist=None,
	require_flat_tail_reject=False,
	use_tail_count_cutoff=True,
	flat_bisect_mode="three_way",
):
	"""
	Helper-based semi-adjustable derived-cutoff search.

	The cutoff may be rederived during shape-boundary bisection.  Shape and
	flat-tail bootstraps use B; CDF-validity bootstraps use cdf_B.
	"""
	if max_checks < 2*int(n_alpha_probes):
		raise ValueError("max_checks must be at least 2*n_alpha_probes for separated shape/CDF probes.")
	if gap_tol is None:
		gap_tol = 2.0 * float(alpha_tol)
	else:
		gap_tol = float(gap_tol)
	if not np.isfinite(gap_tol) or gap_tol < 0.0:
		raise ValueError("gap_tol must be finite and nonnegative.")
	near_miss_probes = int(near_miss_probes)
	if near_miss_probes < 0:
		raise ValueError("near_miss_probes must be nonnegative.")

	boot = _BootstrapPlan.make(
		"pava3.alpha_minimize_derived_x0_revisited",
		B, delta, cdf_delta=cdf_delta, cdf_B=cdf_B,
		flat_delta=flat_delta, require_flat_tail_reject=require_flat_tail_reject
	)
	ctx = _DerivedSearchContext(
		x_mix, x_alt,
		label="semi_derived_x0",
		alpha_bounds=alpha_bounds,
		n_alpha_probes=n_alpha_probes,
		alpha_tol=alpha_tol,
		boot=boot,
		min_mix_tail=min_mix_tail,
		min_alt_tail=min_alt_tail,
		random_state=random_state,
		grid=grid,
		candidate_cutoffs=candidate_cutoffs,
		max_grid_points=max_grid_points,
		verbose=verbose,
		x0_tol=x0_tol,
		shape_weight_gamma=shape_weight_gamma,
		require_flat_tail_reject=require_flat_tail_reject,
		use_tail_count_cutoff=use_tail_count_cutoff,
		flat_bisect_mode=flat_bisect_mode,
		cdf_ref_max_dist=cdf_ref_max_dist,
	)
	ctx.setup_print(
		"setup-semi-x0-2", max_checks, gap_tol,
		extra=(
			f", flat_reject={bool(require_flat_tail_reject)}, "
			f"shape_weight_gamma={float(shape_weight_gamma):.3g}, "
			f"flat_bisect_mode={str(flat_bisect_mode)}, "
			f"use_tail_count_cutoff={bool(use_tail_count_cutoff)}"
		)
	)
	mode = 'semiadjustable_x0_separated_bootstrap_pava3'
	if ctx.setup_error is not None:
		return np.nan, None, None, None, ctx.grid_x, ctx.failure_info(mode, ctx.setup_error['reason'])
	if not ctx.initialize_g_anchor():
		info = ctx.failure_info(mode, 'no_convex_compatible_matched_cutoff')
		return np.nan, None, None, None, ctx.grid_x, info

	def shape_probe_record(diag, meta):
		out = ctx.compact_shape(diag)
		if out is None:
			out = {}
		out.update({
			'search': meta.get('search') if meta else None,
			'updated': bool(meta.get('updated', False)) if meta else False,
		})
		return out

	def make_info(success, reason, alpha=None, pos=None, shape_diag=None, cdf_diag=None, extra=None):
		x0 = ctx.candidate_value(pos) if pos is not None else None
		info = {
			'success': bool(success),
			'reason': reason,
			'alpha': float(alpha) if alpha is not None else np.nan,
			'x0': x0,
			'max_checks': int(max_checks),
			'gap_tol': float(gap_tol),
			'near_miss_probes': int(near_miss_probes),
		}
		info.update(ctx.base_info(mode))
		if shape_diag is not None:
			info.update({
				'shape_ok': bool(shape_diag['shape_ok']),
				'cutoff_selection_ok': bool(
					shape_diag.get('shape_ok', False) and
					(not ctx.require_flat_tail_reject or bool(shape_diag.get('flat_reject', False)))
				),
				'residual_lcm_gap': float(shape_diag['shape_gap']),
				'residual_lcm_tol': float(shape_diag['shape_tol']),
				'mix_tail_count': int(shape_diag['mix_tail_count']),
				'alt_tail_count': int(shape_diag['alt_tail_count']),
			})
			if ctx.require_flat_tail_reject or 'flat_reject' in shape_diag:
				info.update({
					'flat_available': bool(shape_diag.get('flat_available', False)),
					'flat_reject': None if shape_diag.get('flat_reject') is None else bool(shape_diag.get('flat_reject')),
					'flat_stat': float(shape_diag.get('flat_stat', np.nan)),
					'flat_tol': float(shape_diag.get('flat_tol', np.nan)),
					'flat_reason': shape_diag.get('flat_reason'),
					'flat_computed_with_shape_fail': bool(shape_diag.get('flat_computed_with_shape_fail', False)),
				})
		if cdf_diag is not None:
			info.update({
				'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
				'cdf_available': bool(cdf_diag.get('cdf_available', True)),
				'cdf_range_violation': float(cdf_diag['cdf_range_violation']),
				'cdf_low_violation': float(cdf_diag['cdf_low_violation']),
				'cdf_high_violation': float(cdf_diag['cdf_high_violation']),
				'cdf_range_tol': float(cdf_diag['cdf_range_tol']),
				'cdf_mon_violation': float(cdf_diag['cdf_mon_violation']),
				'cdf_mon_tol': float(cdf_diag['cdf_mon_tol']),
				'cdf_alpha_ref': None if cdf_diag.get('alpha_ref') is None else float(cdf_diag['alpha_ref']),
				'cdf_x_ref': None if cdf_diag.get('x_ref') is None else float(cdf_diag['x_ref']),
				'cdf_ref_distance': None if cdf_diag.get('ref_distance') is None else float(cdf_diag['ref_distance']),
				'cdf_weak_ref_calibration': bool(cdf_diag.get('weak_ref_calibration', False)),
			})
		if extra:
			info.update(extra)
		return info

	shape_bracket = _run_shape_alpha_bracket(
		alpha_left=ctx.alpha_left,
		alpha_right=ctx.alpha_right,
		n_alpha_probes=ctx.n_alpha_probes,
		alpha_tol=ctx.alpha_tol,
		max_checks=max_checks,
		total_checks=ctx.total_alpha_checks,
		evaluate_shape_path=lambda a, label: ctx.evaluate_shape_path(a, label=label, cutoff_strategy="leftmost"),
		shape_probe_record=shape_probe_record,
		vprint=ctx.vprint,
	)
	probe_alphas = shape_bracket['probe_alphas']
	shape_probe_table = shape_bracket['shape_probe_table']
	if not shape_bracket['success']:
		info = make_info(
			False, shape_bracket['reason'],
			extra={'shape_probe_table': shape_probe_table}
		)
		return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info

	alpha_shape_diag = shape_bracket['alpha_shape_diag']
	alpha_shape_pos = shape_bracket['alpha_shape_pos']
	shape_boundary = shape_bracket['shape_boundary']
	alpha_shape = float(shape_bracket['alpha_shape'])
	cdf_at_shape = ctx.evaluate_cdf(alpha_shape, label="direct-cdf-at-shape2", pos=alpha_shape_pos, alpha_ref=alpha_shape)
	if cdf_at_shape['cdf_ok']:
		cdf_probe_table = [ctx.compact_cdf(cdf_at_shape)]
		shared_extra = {
			'alpha_shape': float(alpha_shape),
			'alpha_cdf': float(alpha_shape),
			'boundary_gap': 0.0,
			'overlap_width': 0.0,
			'gap_status': 'overlap',
			'shape_boundary': shape_boundary,
			'cdf_boundary': {
				'status': 'direct_alpha_shape_pass',
				'lo': float(alpha_shape),
				'hi': float(alpha_shape),
				'width': 0.0,
			},
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table),
			'alpha_shape_diag': ctx.compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': ctx.compact_cdf(cdf_at_shape),
			'n_shape_refs': int(len(ctx.shape_refs)),
		}
		ctx.vprint(f"[direct-alpha-shape2] alpha={alpha_shape:.6f}, ShapeR=OK, CDFOK=OK; stopping")
		info = make_info(
			True, 'overlap_shape_boundary',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=shared_extra,
		)
		return alpha_shape, info['x0'], alpha_shape_diag['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info

	cdf_boundary_result = _run_cdf_alpha_boundary(
		probe_alphas=probe_alphas,
		alpha_right=ctx.alpha_right,
		alpha_tol=ctx.alpha_tol,
		max_checks=max_checks,
		total_checks=ctx.total_alpha_checks,
		evaluate_cdf=ctx.evaluate_cdf,
		compact_cdf=ctx.compact_cdf,
		vprint=ctx.vprint,
		stop_after_pass_at=alpha_shape,
	)
	cdf_probe_table = cdf_boundary_result['cdf_probe_table']
	if not cdf_boundary_result['success']:
		cdf_diag_at_shape = ctx.evaluate_cdf(alpha_shape, label="cdf-at-shape2", pos=alpha_shape_pos, alpha_ref=alpha_shape)
		extra = {
			'alpha_shape': float(alpha_shape),
			'alpha_cdf': None,
			'gap_status': 'no_cdf_passing_probe',
			'shape_boundary': shape_boundary,
			'shape_probe_table': shape_probe_table,
			'cdf_probe_table': cdf_probe_table,
			'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table),
			'alpha_shape_diag': ctx.compact_shape(alpha_shape_diag),
			'alpha_cdf_diag': None,
		}
		info = make_info(
			False, 'no_cdf_passing_probe',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_diag_at_shape,
			extra=extra,
		)
		return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info

	alpha_cdf_diag = cdf_boundary_result['alpha_cdf_diag']
	alpha_cdf = float(cdf_boundary_result['alpha_cdf'])
	gap = alpha_shape - alpha_cdf
	shared_extra = {
		'alpha_shape': float(alpha_shape),
		'alpha_cdf': float(alpha_cdf),
		'boundary_gap': float(gap),
		'overlap_width': float(max(0.0, alpha_cdf - alpha_shape)),
		'gap_status': 'overlap' if gap <= 0.0 else ('near_miss_gap' if gap <= gap_tol else 'substantial_gap'),
		'shape_boundary': shape_boundary,
		'cdf_boundary': cdf_boundary_result['cdf_boundary'],
		'shape_probe_table': shape_probe_table,
		'cdf_probe_table': cdf_probe_table,
		'coarse_probe_table': _merge_alpha_probe_tables(shape_probe_table, cdf_probe_table),
		'alpha_shape_diag': ctx.compact_shape(alpha_shape_diag),
		'alpha_cdf_diag': ctx.compact_cdf(alpha_cdf_diag),
		'n_shape_refs': int(len(ctx.shape_refs)),
	}
	ctx.vprint(
		f"[boundaries2] alpha_shape={alpha_shape:.6f}, alpha_cdf={alpha_cdf:.6f}, "
		f"gap={gap:.3g}, status={shared_extra['gap_status']}"
	)

	def full_check(alpha, label="full"):
		ok_shape, pos, shape_diag, shape_meta = ctx.evaluate_shape_path(alpha, label=f"{label}-shape", cutoff_strategy="leftmost")
		if ok_shape:
			cdf_diag = ctx.evaluate_cdf(alpha, label=f"{label}-cdf", pos=pos, alpha_ref=alpha)
		else:
			cdf_diag = ctx.evaluate_cdf(alpha, label=f"{label}-cdf")
		return {
			'alpha': float(alpha),
			'shape_ok': bool(ok_shape),
			'cdf_ok': bool(cdf_diag.get('cdf_ok', False)),
			'full_ok': bool(ok_shape and cdf_diag.get('cdf_ok', False)),
			'pos': None if pos is None else int(pos),
			'x0': None if pos is None else ctx.candidate_value(pos),
			'shape_diag': shape_diag,
			'cdf_diag': cdf_diag,
			'shape_meta': shape_meta,
		}

	def scan_for_full_pass(lo, hi, label):
		if near_miss_probes <= 0 or ctx.total_alpha_checks() >= max_checks or hi < lo:
			return None, []
		alphas = np.linspace(float(lo), float(hi), num=near_miss_probes + 2)
		scan = []
		for a in alphas:
			if ctx.total_alpha_checks() >= max_checks:
				break
			full = full_check(a, label=label)
			scan.append({
				'alpha': float(full['alpha']),
				'shape_ok': bool(full['shape_ok']),
				'cdf_ok': bool(full['cdf_ok']),
				'full_ok': bool(full['full_ok']),
				'x0': full['x0'],
				'shape_diag': ctx.compact_shape(full['shape_diag']),
				'cdf_diag': ctx.compact_cdf(full['cdf_diag']),
			})
			if full['full_ok']:
				return full, scan
		return None, scan

	cdf_at_shape = ctx.evaluate_cdf(alpha_shape, label="cdf-at-shape2", pos=alpha_shape_pos, alpha_ref=alpha_shape)
	if gap <= 0.0:
		if cdf_at_shape['cdf_ok']:
			info = make_info(
				True, 'overlap_shape_boundary',
				alpha=alpha_shape, pos=alpha_shape_pos,
				shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
				extra=shared_extra,
			)
			return alpha_shape, info['x0'], alpha_shape_diag['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		full, scan = scan_for_full_pass(alpha_shape, alpha_cdf, "overlap-scan2")
		extra = dict(shared_extra)
		extra['overlap_scan_table'] = scan
		if full is not None:
			info = make_info(
				True, 'overlap_local_full_pass',
				alpha=full['alpha'], pos=full['pos'],
				shape_diag=full['shape_diag'], cdf_diag=full['cdf_diag'],
				extra=extra,
			)
			return float(full['alpha']), info['x0'], full['shape_diag']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		info = make_info(
			False, 'overlap_without_full_passing_probe',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=extra,
		)
		return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info
	if gap <= gap_tol:
		full, scan = scan_for_full_pass(alpha_cdf, alpha_shape, "near-miss2")
		extra = dict(shared_extra)
		extra['near_miss_scan_table'] = scan
		if full is not None:
			info = make_info(
				True, 'near_miss_full_passing_probe',
				alpha=full['alpha'], pos=full['pos'],
				shape_diag=full['shape_diag'], cdf_diag=full['cdf_diag'],
				extra=extra,
			)
			return float(full['alpha']), info['x0'], full['shape_diag']['H_hat'], ctx.G_hat_anchor, ctx.grid_x, info
		info = make_info(
			False, 'near_miss_no_overlap',
			alpha=alpha_shape, pos=alpha_shape_pos,
			shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
			extra=extra,
		)
		return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info
	info = make_info(
		False, 'separated_boundaries_gap',
		alpha=alpha_shape, pos=alpha_shape_pos,
		shape_diag=alpha_shape_diag, cdf_diag=cdf_at_shape,
		extra=shared_extra,
	)
	return np.nan, None, None, ctx.G_hat_anchor, ctx.grid_x, info

#!/usr/bin/env python3
"""
diffpair_solver.py - 2D electrostatic field solver for PCB differential pairs.

Models an edge-coupled differential pair on an outer layer over a solid
reference plane, optionally with coplanar ground pour on both sides
(grounded coplanar waveguide) and conformal solder mask. Copper thickness
IS modelled (sidewalls included), unlike closed-form calculators.

Method: finite-difference Laplace solve on a non-uniform grid, quasi-TEM.
Capacitance is computed with and without dielectrics; Z = 1/(c*sqrt(C*C0)).
Lossless, frequency-independent (fine for USB 2.0 and similar).

Requires: numpy, scipy   (pip install numpy scipy)

Examples:
  # GCPW pair, 0.125 mm ground gap, with 20 um solder mask
  python3 diffpair_solver.py -W 0.18 -S 0.125 -H 0.21 -T 0.035 --er 4.4 --gap 0.125 --mask 0.02

  # Plain coupled microstrip (no side ground), no mask
  python3 diffpair_solver.py -W 0.3 -S 0.127 -H 0.21 -T 0.035 --er 4.4

  # Sweep trace width
  python3 diffpair_solver.py -W 0.15 0.18 0.2 -S 0.125 -H 0.21 --gap 0.125 --mask 0.02

All dimensions in mm.
"""
import argparse
import itertools
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.854187817e-12
C0 = 299792458.0


def _grid(breaks, lo, hi, h0, fine_lo, fine_hi, growth=1.15, hmax=0.2):
    """1D non-uniform grid: fine (h0) near the structure, growing outward,
    with every geometric edge in `breaks` placed exactly on a node."""
    pts = sorted(set([lo, hi] + [b for b in breaks if lo < b < hi]))
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        x = a
        while b - x > 1e-12:
            if fine_lo <= x <= fine_hi:
                d = 0.0
            else:
                d = min(abs(x - fine_lo), abs(x - fine_hi))
            h = h0 if d == 0 else h0 * growth ** min(d / h0, 200)
            h = min(hmax, h, h0 + d * (growth - 1) * 3)
            x = min(b, x + max(h, h0))
            if b - x < 0.3 * h0:
                x = b
            out.append(x)
    return np.array(out)


def _charge(W, S, H, T, gap, er, mask, mask_er, h0, odd=True):
    """Charge on trace 1 (units of eps0) with trace1=+1 V, trace2=-1 V (odd)
    or +1 V (even), side grounds and reference plane at 0 V."""
    gnd = gap is not None
    G = gap if gnd else 1.0
    xs = [-S/2 - W, -S/2, S/2, S/2 + W]
    if gnd:
        xs += [-S/2 - W - G, S/2 + W + G]
    m = mask or 0.0
    xb = xs + [x + s*m for x in xs for s in (-1, 1)]
    yb = [0.0, H, H + T, H + m, H + T + m]

    span = S/2 + W + G + 0.1
    box = max(5.0, 12 * H, 3 * span)          # side walls far away
    top = max(4.0, 12 * H)                    # top lid far away
    X = _grid(xb, -box, box, h0, -span, span)
    Y = _grid(yb, 0.0, top, h0, 0.0, H + T + 0.1)
    nx, ny = len(X), len(Y)

    # permittivity per cell
    xc = (X[:-1] + X[1:]) / 2
    yc = (Y[:-1] + Y[1:]) / 2
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    eps = np.ones((nx - 1, ny - 1))
    eps[YC < H] = er

    def tr1(x): return (x >= -S/2 - W) & (x <= -S/2)
    def tr2(x): return (x >= S/2) & (x <= S/2 + W)
    def gc(x):
        if not gnd:
            return np.zeros_like(x, dtype=bool)
        return (x <= -S/2 - W - G) | (x >= S/2 + W + G)

    if mask:
        cu = tr1(XC) | tr2(XC) | gc(XC)
        copper_c = (YC > H) & (YC < H + T) & cu
        surf = np.where(cu, H + T, H)
        near = (YC > H) & (YC < surf + mask) & ~copper_c
        for e in xs:  # conformal coverage of sidewalls
            near |= (YC > H) & (YC < H + T + mask) & (np.abs(XC - e) < mask) & ~copper_c
        eps[near] = mask_er

    # fixed-potential nodes
    XN, YN = np.meshgrid(X, Y, indexing="ij")
    onc = (YN >= H - 1e-12) & (YN <= H + T + 1e-12)
    fixed = np.zeros((nx, ny), bool)
    fixed[:, 0] = fixed[:, -1] = fixed[0, :] = fixed[-1, :] = True
    t1 = onc & tr1(XN)
    t2 = onc & tr2(XN)
    fixed |= t1 | t2 | (onc & gc(XN))
    val = np.zeros((nx, ny))
    val[t1] = 1.0
    val[t2] = -1.0 if odd else 1.0

    # 5-point finite-volume Laplacian with cell-averaged permittivity
    dx, dy = np.diff(X), np.diff(Y)
    idx = np.arange(nx * ny).reshape(nx, ny)
    ep = np.pad(eps, ((0, 0), (1, 1)), constant_values=1)
    dyp = np.concatenate([[0], dy, [0]])
    gh = (ep[:, :-1] * dyp[None, :-1] / 2 + ep[:, 1:] * dyp[None, 1:] / 2) / dx[:, None]
    ep2 = np.pad(eps, ((1, 1), (0, 0)), constant_values=1)
    dxp = np.concatenate([[0], dx, [0]])
    gv = (ep2[:-1, :] * dxp[:-1, None] / 2 + ep2[1:, :] * dxp[1:, None] / 2) / dy[None, :]
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    c, d = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    I = np.concatenate([a, b, c, d])
    J = np.concatenate([b, a, d, c])
    V = np.concatenate([gh.ravel(), gh.ravel(), gv.ravel(), gv.ravel()])
    A = sp.coo_matrix((-V, (I, J)), shape=(nx * ny, nx * ny)).tocsr()
    A = A + sp.diags(-np.asarray(A.sum(axis=1)).ravel())

    f = fixed.ravel()
    u = ~f
    phi = val.ravel().copy()
    phi[u] = spla.spsolve(A[u][:, u].tocsc(), -A[u][:, f] @ phi[f])
    return (A @ phi)[t1.ravel()].sum()


def solve(W, S, H, T, er, gap=None, mask=None, mask_er=3.5, h0=0.0025):
    res = {}
    for mode, odd in (("odd", True), ("even", False)):
        C = _charge(W, S, H, T, gap, er, mask, mask_er, h0, odd) * EPS0
        Ca = _charge(W, S, H, T, gap, 1.0, mask, 1.0, h0, odd) * EPS0
        res["Z" + mode] = 1.0 / (C0 * np.sqrt(C * Ca))
        res["eeff_" + mode] = C / Ca
    res["Zdiff"] = 2 * res["Zodd"]
    res["Zcomm"] = res["Zeven"] / 2
    res["delay_odd_ps_per_mm"] = np.sqrt(res["eeff_odd"]) / C0 * 1e9  # ps/mm
    return res


def main():
    p = argparse.ArgumentParser(
        description="2D field solver for (grounded coplanar) differential pairs. "
                    "Dimensions in mm. Give several values to any of W/S/H/T/gap/mask to sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Examples:")[1])
    p.add_argument("-W", type=float, nargs="+", required=True, help="trace width")
    p.add_argument("-S", type=float, nargs="+", required=True, help="gap between the two traces")
    p.add_argument("-H", type=float, nargs="+", required=True, help="dielectric height to reference plane")
    p.add_argument("-T", type=float, nargs="+", default=[0.035], help="copper thickness (default 0.035)")
    p.add_argument("--er", type=float, default=4.4, help="dielectric constant (default 4.4)")
    p.add_argument("--gap", type=float, nargs="+", default=[None],
                   help="trace-to-side-ground gap; omit for no side ground (plain coupled microstrip)")
    p.add_argument("--mask", type=float, nargs="+", default=[None],
                   help="solder mask thickness (conformal); omit for no mask")
    p.add_argument("--mask-er", type=float, default=3.5, help="solder mask dielectric constant (default 3.5)")
    p.add_argument("--mesh", type=float, default=0.0025,
                   help="finest mesh cell in mm (default 0.0025; smaller = slower, more accurate)")
    a = p.parse_args()

    print(f"{'W':>6} {'S':>6} {'H':>6} {'T':>6} {'gap':>6} {'mask':>6} | "
          f"{'Zdiff':>7} {'Zodd':>6} {'Zeven':>6} {'Zcomm':>6} {'eeff_o':>6} {'ps/mm':>6}")
    for W, S, H, T, gap, mask in itertools.product(a.W, a.S, a.H, a.T, a.gap, a.mask):
        r = solve(W, S, H, T, a.er, gap, mask, a.mask_er, a.mesh)
        g = "-" if gap is None else f"{gap:g}"
        mk = "-" if not mask else f"{mask:g}"
        print(f"{W:6g} {S:6g} {H:6g} {T:6g} {g:>6} {mk:>6} | "
              f"{r['Zdiff']:7.1f} {r['Zodd']:6.1f} {r['Zeven']:6.1f} {r['Zcomm']:6.1f} "
              f"{r['eeff_odd']:6.2f} {r['delay_odd_ps_per_mm']:6.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

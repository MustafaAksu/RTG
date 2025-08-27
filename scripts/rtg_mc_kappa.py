#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTG MC for C_kappa (geometry factor) — v1.1.2
----------------------------------------------
- Nearest-neighbor resonance model on L^3 periodic lattice
- Move types: phase Metropolis, spin flips, optional frequency moves
- Regulators: Litim, Sharp, Gaussian
- Windowed measurement of M2, M4 in x = |Δω|/Δω*
- C_kappa^nat = (1/180) * (M4 / M2)
- x-histogram (all links vs used links), JSON + PNG outputs
- Numba-safe measurement kernel; --no-jit to disable JIT

Example:
  python rtg_mc_kappa.py --L 32 --sweeps 240 --therm 60 \
    --reg litim --seed 2 --pair-init bipartite --mu 0.45 --pair-noise 0.01 \
    --x-min 0.28 --x-max 0.70 --window-mode discard --dphi 1.0 --beta 1.0 \
    --save-every 20 --hist-file xhist_32.png --out kappa_32.json
"""

import argparse, json, time, math, sys
import numpy as np

# ---- Optional modules
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ----------------------------
# Utilities (Python versions)
# ----------------------------

def wrap_pi(a):
    """Map angles to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def litim_weight(x):
    # Litim regulator on x = |Δω|/Δω*: w(x) = (1 - x^2) for x<1 else 0
    return (1.0 - x*x) if (x < 1.0) else 0.0

def sharp_weight(x):
    return 1.0 if (x < 1.0) else 0.0

def gauss_weight(x):
    return math.exp(-x*x)

def reg_weight(x, reg_id):
    if reg_id == 0:   # litim
        return litim_weight(x)
    elif reg_id == 1: # sharp
        return sharp_weight(x)
    else:             # gauss
        return gauss_weight(x)

def gate_from_spins(si, sj):
    # Code spins σ = ±1; open gate for opposite spins → 1 - σi σj ∈ {0, 2}
    return 1 - si*sj

def A_ij(dphi, si, sj, x, reg_id):
    """Resonance amplitude with spin gate: A = (3/4)(1+cos Δφ) * G * w_reg(x)."""
    w = reg_weight(x, reg_id)
    if w == 0.0:
        return 0.0
    G = gate_from_spins(si, sj)
    if G == 0:
        return 0.0
    c = math.cos(dphi)
    return 0.75 * (1.0 + c) * G * w  # ∈ [0, 3]

def exch_weight(x, sigma_ex):
    # Smooth UV regulator for the exchange term (independent of the main regulator)
    if sigma_ex <= 0:
        return 1.0
    y = x / sigma_ex
    return math.exp(-y*y)


# ----------------------------
# Numba-safe helpers & kernel
# ----------------------------

if HAVE_NUMBA:
    @nb.njit(cache=True, nogil=True)
    def _reg_weight_nb(x, reg_id):
        if reg_id == 0:  # litim
            return (1.0 - x*x) if (x < 1.0) else 0.0
        elif reg_id == 1:  # sharp
            return 1.0 if (x < 1.0) else 0.0
        else:  # gauss
            return math.exp(-x*x)

    @nb.njit(cache=True, nogil=True)
    def _A_ij_nb(dphi, si, sj, x, reg_id):
        w = _reg_weight_nb(x, reg_id)
        if w == 0.0:
            return 0.0
        G = 1 - si*sj  # gate
        if G == 0:
            return 0.0
        c = math.cos(dphi)
        return 0.75 * (1.0 + c) * G * w

    @nb.njit(cache=True, nogil=True)
    def _measure_kernel_nb(phi, spin, omega, neigh_fwd, reg_id,
                           x_min, x_max, mode_flag,
                           hist_edges):
        Lx, Ly, Lz = phi.shape
        M2 = 0.0
        M4 = 0.0
        count_links = 0

        nbins = hist_edges.shape[0] - 1
        hist_all = np.zeros(nbins, dtype=np.int64)
        hist_used = np.zeros(nbins, dtype=np.int64)
        Hmin = hist_edges[0]
        Hmax = hist_edges[-1]
        Hdx = (Hmax - Hmin) / nbins

        for ix in range(Lx):
            for iy in range(Ly):
                for iz in range(Lz):
                    phii = float(phi[ix,iy,iz])
                    si   = int(spin[ix,iy,iz])
                    wi   = float(omega[ix,iy,iz])

                    for d in range(3):  # +x,+y,+z
                        jx = neigh_fwd[ix,iy,iz,d,0]
                        jy = neigh_fwd[ix,iy,iz,d,1]
                        jz = neigh_fwd[ix,iy,iz,d,2]

                        dp = phii - float(phi[jx,jy,jz])
                        # fast wrap of Δφ to (-π,π]
                        dp = math.atan2(math.sin(dp), math.cos(dp))
                        x = wi - float(omega[jx,jy,jz])
                        if x < 0.0:
                            x = -x

                        # Histogram: all links
                        if Hmin <= x < Hmax:
                            b = int((x - Hmin)/Hdx)
                            if 0 <= b < nbins:
                                hist_all[b] += 1

                        # Window handling
                        if mode_flag == 0:  # discard
                            if (x < x_min) or (x > x_max):
                                continue
                            x_eff = x
                        elif mode_flag == 1:  # clamp
                            if x < x_min:
                                x_eff = x_min
                            elif x > x_max:
                                x_eff = x_max
                            else:
                                x_eff = x
                        else:  # none
                            x_eff = x

                        A = _A_ij_nb(dp, si, int(spin[jx,jy,jz]), x_eff, reg_id)
                        if A == 0.0:
                            continue

                        # Beat-distance length (Δω* set to 1 here): ℓ̂ = 2π/x
                        ell_hat = (2.0*math.pi) / (x_eff + 1e-14)
                        l2 = ell_hat * ell_hat
                        M2 += A * l2
                        M4 += A * l2 * l2
                        count_links += 1

                        # Histogram: used links
                        if Hmin <= x_eff < Hmax:
                            b2 = int((x_eff - Hmin)/Hdx)
                            if 0 <= b2 < nbins:
                                hist_used[b2] += 1

        Nsites = Lx * Ly * Lz
        return (M2 / Nsites, M4 / Nsites, count_links, hist_all, hist_used)


# ----------------------------
# Model energy (Python)
# ----------------------------

def local_energy_at(site_idx, phi, spin, omega_over_star, neigh, params):
    """Compute energy contribution of a site to the Hamiltonian from 6 NN links."""
    ix, iy, iz = site_idx
    phii = float(phi[ix, iy, iz])
    si   = int(spin[ix, iy, iz])
    wi   = float(omega_over_star[ix, iy, iz])

    e = 0.0
    for d in range(6):
        jx, jy, jz = neigh[ix, iy, iz, d]
        dphi = phii - float(phi[jx, jy, jz])
        dphi = math.atan2(math.sin(dphi), math.cos(dphi))
        x = abs(wi - float(omega_over_star[jx, jy, jz]))
        A = A_ij(dphi, si, int(spin[jx, jy, jz]), x, params['reg_id'])
        e += params['Kp'] * x
        e += params['J']  * A
        if params['Jex'] != 0.0:
            e += params['Jex'] * math.sin(dphi) * exch_weight(x, params['sigma_ex'])
    return e


# ----------------------------
# Lattice topology
# ----------------------------

def build_neighbors(L):
    """Forward/backward neighbors for 3D periodic lattice."""
    neigh = np.zeros((L, L, L, 6, 3), dtype=np.int32)
    for ix in range(L):
        ixm = (ix - 1) % L ; ixp = (ix + 1) % L
        for iy in range(L):
            iym = (iy - 1) % L ; iyp = (iy + 1) % L
            for iz in range(L):
                izm = (iz - 1) % L ; izp = (iz + 1) % L
                neigh[ix,iy,iz,0] = (ixp, iy , iz )
                neigh[ix,iy,iz,1] = (ix , iyp, iz )
                neigh[ix,iy,iz,2] = (ix , iy , izp)
                neigh[ix,iy,iz,3] = (ixm, iy , iz )
                neigh[ix,iy,iz,4] = (ix , iym, iz )
                neigh[ix,iy,iz,5] = (ix , iy , izm)
    return neigh


# ----------------------------
# Main MC driver
# ----------------------------

def run_mc(args):
    rng = np.random.default_rng(args.seed)
    L = args.L

    # pick kernel
    USE_JIT = (HAVE_NUMBA and (not args.no_jit))
    # select reg
    reg_id = {"litim":0, "sharp":1, "gauss":2}[args.reg]

    # State arrays
    phi  = rng.uniform(-np.pi, np.pi, size=(L,L,L)).astype(np.float64)
    spin = rng.choice(np.array([-1,1], dtype=np.int8), size=(L,L,L))

    # ω/Δω* initialization
    if args.pair_init == "bipartite":
        base = 0.5 * args.mu
        parity = (np.indices((L,L,L)).sum(axis=0) & 1).astype(np.int8)
        omega_over_star = (base * (1 - 2*parity)).astype(np.float64)
        if args.pair_noise > 0.0:
            omega_over_star += (args.pair_noise * rng.standard_normal((L,L,L))).astype(np.float64)
    else:
        omega_over_star = (args.mu + args.width * rng.standard_normal((L,L,L))).astype(np.float64)

    # Neighbors
    neigh = build_neighbors(L)
    neigh_fwd = neigh[:,:,:,0:3,:]
    neigh_all = neigh

    # Params for energy
    params = {
        "Kp": args.Kp,
        "J": args.J,
        "Jex": args.Jex,
        "sigma_ex": args.sigma_ex,
        "reg_id": reg_id
    }

    beta = args.beta
    dphi = args.dphi
    domega = args.domega

    # Acceptance counters
    acc_phi = acc_spin = acc_w = 0
    prop_phi = prop_spin = prop_w = 0

    # Histogram setup
    nbins = args.hist_bins
    hist_edges = np.linspace(0.0, args.hist_range_max, nbins+1)
    hist_all_tot = np.zeros(nbins, dtype=np.int64)
    hist_used_tot = np.zeros(nbins, dtype=np.int64)

    # Storage for measurement snapshots
    M2_list, M4_list, ratio_list, Ck_list = [], [], [], []
    xlinks_list = []

    start = time.time()
    print(f"[init] L={L}, reg={args.reg}, beta={beta}, sweeps={args.meas_sweeps} (therm {args.therm_sweeps}) | "
          f"mu={args.mu}, width={args.width}")

    # ----- Thermalization
    for sw in range(1, args.therm_sweeps+1):
        # Phase updates
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    prop_phi += 1
                    old = phi[ix,iy,iz]
                    new = wrap_pi(old + dphi*(rng.random()*2.0 - 1.0))
                    e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    phi[ix,iy,iz] = new
                    e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    dE = e_new - e_old
                    if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                        acc_phi += 1
                    else:
                        phi[ix,iy,iz] = old

        # Spin flips
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    prop_spin += 1
                    olds = spin[ix,iy,iz]
                    news = -olds
                    e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    spin[ix,iy,iz] = news
                    e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    dE = e_new - e_old
                    if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                        acc_spin += 1
                    else:
                        spin[ix,iy,iz] = olds

        # Optional frequency moves
        if args.freq_moves:
            for ix in range(L):
                for iy in range(L):
                    for iz in range(L):
                        prop_w += 1
                        oldw = omega_over_star[ix,iy,iz]
                        neww = oldw + domega*(rng.random()*2.0 - 1.0)
                        e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                        omega_over_star[ix,iy,iz] = neww
                        e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                        dE = e_new - e_old
                        if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                            acc_w += 1
                        else:
                            omega_over_star[ix,iy,iz] = oldw

        if (sw % 10) == 0 or (sw == args.therm_sweeps):
            a_phi = acc_phi/max(1,prop_phi)
            a_spin = acc_spin/max(1,prop_spin)
            a_w = acc_w/max(1,prop_w)
            print(f"[therm {sw:4d}/{args.therm_sweeps}] acc_phi={a_phi:.2f} acc_spin={a_spin:.2f} acc_ω={a_w:.2f}")

    # ----- Measurement
    acc_phi = acc_spin = acc_w = 0
    prop_phi = prop_spin = prop_w = 0
    mode_flag = {"discard":0, "clamp":1, "none":2}[args.window_mode]

    for sw in range(1, args.meas_sweeps+1):
        # Phase updates
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    prop_phi += 1
                    old = phi[ix,iy,iz]
                    new = wrap_pi(old + dphi*(rng.random()*2.0 - 1.0))
                    e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    phi[ix,iy,iz] = new
                    e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    dE = e_new - e_old
                    if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                        acc_phi += 1
                    else:
                        phi[ix,iy,iz] = old

        # Spin flips
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    prop_spin += 1
                    olds = spin[ix,iy,iz]
                    news = -olds
                    e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    spin[ix,iy,iz] = news
                    e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                    dE = e_new - e_old
                    if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                        acc_spin += 1
                    else:
                        spin[ix,iy,iz] = olds

        # Optional frequency moves
        if args.freq_moves:
            for ix in range(L):
                for iy in range(L):
                    for iz in range(L):
                        prop_w += 1
                        oldw = omega_over_star[ix,iy,iz]
                        neww = oldw + domega*(rng.random()*2.0 - 1.0)
                        e_old = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                        omega_over_star[ix,iy,iz] = neww
                        e_new = local_energy_at((ix,iy,iz), phi, spin, omega_over_star, neigh_all, params)
                        dE = e_new - e_old
                        if (dE <= 0.0) or (rng.random() < math.exp(-beta*dE)):
                            acc_w += 1
                        else:
                            omega_over_star[ix,iy,iz] = oldw

        # Snapshot measure
        if (sw % args.save_every) == 0 or (sw == args.meas_sweeps):
            if USE_JIT:
                M2, M4, nlinks, h_all, h_used = _measure_kernel_nb(
                    phi, spin, omega_over_star, neigh_fwd, reg_id,
                    args.x_min, args.x_max, mode_flag,
                    hist_edges
                )
            else:
                # Pure-Python fallback of the same logic
                M2 = 0.0 ; M4 = 0.0 ; nlinks = 0
                nbins = hist_edges.shape[0] - 1
                h_all = np.zeros(nbins, dtype=np.int64)
                h_used = np.zeros(nbins, dtype=np.int64)
                Hmin = hist_edges[0]; Hmax = hist_edges[-1]; Hdx = (Hmax - Hmin)/nbins

                for ix in range(L):
                    for iy in range(L):
                        for iz in range(L):
                            phii = float(phi[ix,iy,iz])
                            si   = int(spin[ix,iy,iz])
                            wi   = float(omega_over_star[ix,iy,iz])

                            for d in range(3):
                                jx, jy, jz = neigh_fwd[ix,iy,iz,d]
                                dp = phii - float(phi[jx,jy,jz])
                                dp = math.atan2(math.sin(dp), math.cos(dp))
                                x = abs(wi - float(omega_over_star[jx,jy,jz]))

                                if Hmin <= x < Hmax:
                                    b = int((x - Hmin)/Hdx)
                                    if 0 <= b < nbins:
                                        h_all[b] += 1

                                # window
                                if mode_flag == 0 and not (args.x_min <= x <= args.x_max):
                                    continue
                                if mode_flag == 1:
                                    x_eff = args.x_min if x < args.x_min else (args.x_max if x > args.x_max else x)
                                else:
                                    x_eff = x

                                A = A_ij(dp, si, int(spin[jx,jy,jz]), x_eff, reg_id)
                                if A == 0.0:
                                    continue

                                ell_hat = (2.0*np.pi)/(x_eff + 1e-14)
                                l2 = ell_hat*ell_hat
                                M2 += A*l2
                                M4 += A*l2*l2
                                nlinks += 1

                                if Hmin <= x_eff < Hmax:
                                    b2 = int((x_eff - Hmin)/Hdx)
                                    if 0 <= b2 < nbins:
                                        h_used[b2] += 1

                Nsites = L*L*L
                M2 /= Nsites
                M4 /= Nsites

            ratio = (M4 / max(M2, 1e-30))
            Ck_nat = ratio / 180.0

            M2_list.append(M2)
            M4_list.append(M4)
            ratio_list.append(ratio)
            Ck_list.append(Ck_nat)
            xlinks_list.append(int(nlinks))
            hist_all_tot += h_all
            hist_used_tot += h_used

            a_phi = acc_phi/max(1,prop_phi)
            a_spin = acc_spin/max(1,prop_spin)
            a_w = acc_w/max(1,prop_w)
            print(f"[meas {sw:4d}/{args.meas_sweeps}] acc φ={a_phi:.2f} spin={a_spin:.2f} ω={a_w:.2f} | "
                  f"M2={M2:.3e} M4={M4:.3e} ratio={ratio:.3e} Cκ_nat={Ck_nat:.4e}")

    # Averages + SE
    def mean_se(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return (np.nan, np.nan)
        m = a.mean()
        se = a.std(ddof=1)/np.sqrt(max(1, a.size))
        return m, se

    M2_mean, M2_se = mean_se(M2_list)
    M4_mean, M4_se = mean_se(M4_list)
    ratio_mean, ratio_se = mean_se(ratio_list)
    Ck_mean, Ck_se = mean_se(Ck_list)

    # Optional SI mapping (pass-through; geometry factor is dimensionless)
    Ck_SI_mean = Ck_mean if args.delta_omega_star_SI else None
    Ck_SI_se   = Ck_se   if args.delta_omega_star_SI else None

    elapsed = time.time() - start

    # Histogram plot
    if args.hist_file and HAVE_PLT:
        centers = 0.5*(hist_edges[:-1]+hist_edges[1:])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.step(centers, hist_all_tot, where='mid', linewidth=1.5, label='all links')
        ax.step(centers, hist_used_tot, where='mid', linewidth=1.5, label='used (windowed)')
        ax.axvspan(args.x_min, args.x_max, alpha=0.15, hatch='//', label='window')
        ax.set_xlabel(r'$x=|\Delta\omega|/\Delta\omega^\ast$')
        ax.set_ylabel('count')
        ax.set_title(f'x-histogram L={L}, reg={args.reg}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.hist_file, dpi=150)
        plt.close(fig)

    # JSON dump
    result = {
        "L": L,
        "beta": beta,
        "reg": args.reg,
        "mu": args.mu,
        "width": args.width,
        "Kp": args.Kp,
        "J": args.J,
        "Jex": args.Jex,
        "sigma_ex": args.sigma_ex,
        "freq_moves": bool(args.freq_moves),
        "dphi": args.dphi,
        "domega": args.domega,
        "therm_sweeps": args.therm_sweeps,
        "meas_sweeps": args.meas_sweeps,
        "save_every": args.save_every,
        "seed": args.seed,
        "pair_init": args.pair_init,
        "pair_noise": args.pair_noise,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "window_mode": args.window_mode,
        "hist_bins": args.hist_bins,
        "hist_range_max": args.hist_range_max,
        "hist_edges": (hist_edges.tolist()),
        "hist_all_counts": hist_all_tot.tolist(),
        "hist_used_counts": hist_used_tot.tolist(),
        "xlinks_samples": xlinks_list,
        "M2_mean": M2_mean, "M2_se": M2_se,
        "M4_mean": M4_mean, "M4_se": M4_se,
        "ratio_mean": ratio_mean, "ratio_se": ratio_se,
        "Ckappa_nat_mean": Ck_mean, "Ckappa_nat_se": Ck_se,
        "Ckappa_SI_mean": Ck_SI_mean, "Ckappa_SI_se": Ck_SI_se,
        "elapsed_sec": elapsed,
        "jit_enabled": USE_JIT
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))

    return result


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RTG MC for C_kappa (geometry factor)")
    p.add_argument("--L", type=int, default=32, help="Linear lattice size")
    p.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    p.add_argument("--sweeps", type=int, default=200, help="Total measurement sweeps (alias of --meas-sweeps)")
    p.add_argument("--meas-sweeps", type=int, help="Measurement sweeps (overrides --sweeps)")
    p.add_argument("--therm", type=int, default=50, help="Thermalization sweeps (alias of --therm-sweeps)")
    p.add_argument("--therm-sweeps", type=int, help="Thermalization sweeps (overrides --therm)")
    p.add_argument("--save-every", type=int, default=20, help="Measure every N sweeps")
    p.add_argument("--reg", type=str, default="litim", choices=["litim","sharp","gauss"], help="Regulator")
    p.add_argument("--seed", type=int, default=1, help="PRNG seed")
    p.add_argument("--no-jit", action="store_true", help="Disable Numba JIT for measurement (pure Python)")

    # Model parameters (dimensionless here)
    p.add_argument("--mu", type=float, default=0.45, help="Target |Δω|/Δω* scale (window center)")
    p.add_argument("--width", type=float, default=0.10, help="Std of per-site ω init (gauss mode)")
    p.add_argument("--Kp", type=float, default=0.5, help="Coefficient for K' term (~|Δω|/Δω*)")
    p.add_argument("--J", type=float, default=1.0, help="Coefficient for resonance term")
    p.add_argument("--Jex", type=float, default=0.2, help="Coefficient for exchange term")
    p.add_argument("--sigma-ex", dest="sigma_ex", type=float, default=1.0, help="UV width for exchange term in x units")

    # Moves
    p.add_argument("--freq-moves", action="store_true", help="Enable ω moves")
    p.add_argument("--dphi", type=float, default=0.35, help="Phase proposal half-range")
    p.add_argument("--domega", type=float, default=0.05, help="Frequency proposal half-range in x units")

    # Pair-level initialization and measurement windowing
    p.add_argument("--pair-init", type=str, default="gauss",
                   choices=["gauss","bipartite"],
                   help="Init ω/Δω*: 'gauss' (per-site N(μ, width^2)) or 'bipartite' (NN |Δω|≈μ).")
    p.add_argument("--pair-noise", type=float, default=0.02,
                   help="Std of small per-site noise for bipartite init (in x units).")
    p.add_argument("--x-min", type=float, default=0.28, help="Measurement lower cut for x=|Δω|/Δω*")
    p.add_argument("--x-max", type=float, default=0.70, help="Measurement upper cut for x=|Δω|/Δω*")
    p.add_argument("--window-mode", type=str, default="discard",
                   choices=["discard","clamp","none"],
                   help="How to treat links outside [x_min,x_max] during measurement.")

    # Histogram and output
    p.add_argument("--hist-bins", type=int, default=60, help="Number of x-histogram bins")
    p.add_argument("--hist-range-max", type=float, default=1.2, help="x-histogram max (min is 0)")
    p.add_argument("--hist-file", type=str, default="", help="If set, save x-histogram PNG here")
    p.add_argument("--out", type=str, default="", help="If set, save JSON here")
    p.add_argument("--delta-omega-star-SI", type=float, default=None,
                   help="Optional Δω* in s^-1 for SI mapping (kept as pass-through unless defined).")

    args = p.parse_args()
    if args.meas_sweeps is None:
        args.meas_sweeps = args.sweeps
    if args.therm_sweeps is None:
        args.therm_sweeps = args.therm
    return args


def main():
    args = parse_args()
    _ = run_mc(args)


if __name__ == "__main__":
    main()

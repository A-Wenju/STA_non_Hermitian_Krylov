"""Physics core for the non-Hermitian MFIM Krylov-CD project.

  Hamiltonian / dynamics   H_of, dH_of, split, evolve, evolve2, diagnostics
  branch tracking          frame, tracked_frames   (ground branch by max
                            biorthogonal overlap with the previous step)
  Krylov ground-state AGP  arnoldi_ops, agp_exact_gs, agp_krylov_gs, Eqs. (F1)-(F3)
  optimal-ramp schedule    _fit_tau, solve, schedule   (App. F, Eq. F8)
"""
import numpy as np
import scipy.sparse as sp
from functools import lru_cache

# ------------------------------------------------------------------ Hamiltonian
sx = np.array([[0, 1], [1, 0]], float)
sz = np.array([[1, 0], [0, -1]], float)
isy = np.array([[0, 1], [-1, 0]], float)      # i*sigma^y  (real, antisymmetric)
I2 = sp.identity(2, format='csr')


def _op(L, mats):
    out = sp.identity(1, format='csr')
    for j in range(L):
        m = mats.get(j)
        out = sp.kron(out, sp.csr_matrix(m) if m is not None else I2, format='csr')
    return out


@lru_cache(maxsize=8)
def terms(L):
    XX = sum(_op(L, {j: sx, (j + 1) % L: sx}) for j in range(L)).tocsr()
    Z = sum(_op(L, {j: sz}) for j in range(L)).tocsr()
    IY = sum(_op(L, {j: isy}) for j in range(L)).tocsr()
    X = sum(_op(L, {j: sx}) for j in range(L)).tocsr()
    return XX, Z, IY, X


def H_of(L, h, gamma, J=1.0, eps=0.5):
    XX, Z, IY, X = terms(L)
    return (-J * XX + h * Z + h * gamma * IY + eps * X).astype(complex)


def dH_of(L, gamma, J=1.0, eps=0.5):
    XX, Z, IY, X = terms(L)
    return (Z + gamma * IY).astype(complex)      # d/dh


def split(L, gamma, J=1.0, eps=0.5):
    XX, Z, IY, X = terms(L)
    A = (-J * XX + eps * X).astype(complex).tocsr()
    B = (Z + gamma * IY).astype(complex).tocsr()
    return A, B


# ------------------------------------------------------------------ branch tracking
def frame(L, h, gamma, J=1.0, eps=0.5, prev=None):
    """Dense biorthogonal frame at fixed h. If `prev` (a previous-step right
    eigenvector) is given, i0 continues that branch by max overlap; otherwise
    i0 is the real-part ground state."""
    H = H_of(L, h, gamma, J, eps).toarray()
    E, R = np.linalg.eig(H)
    R = R / np.linalg.norm(R, axis=0)
    Li = np.linalg.inv(R)
    i0 = int(np.argmin(E.real)) if prev is None else int(np.argmax(np.abs(Li @ prev)))
    return H, E, R, Li, i0


def tracked_frames(L, hgrid, gamma, J=1.0, eps=0.5):
    """frame() at every h in hgrid, continuing the ground branch by max
    biorthogonal overlap with the previous step. Yields
    (H, E, R, Li, i0)."""
    prev = None
    for h in hgrid:
        H, E, R, Li, i0 = frame(L, h, gamma, J, eps, prev)
        prev = R[:, i0].copy()
        yield H, E, R, Li, i0


# ------------------------------------------------------------------ dynamics
def evolve(L, gamma, hfun, tau, psi0, nsteps, **kw):
    """Classical RK4 on i dpsi/dt = H psi, renormalising after every step."""
    dt = tau / nsteps
    psi = psi0.astype(complex).copy(); psi /= np.linalg.norm(psi)
    for k in range(nsteps):
        t = k * dt

        def f(tt, v):
            return -1j * (H_of(L, float(hfun(tt)), gamma, **kw) @ v)

        k1 = f(t, psi); k2 = f(t + dt / 2, psi + dt / 2 * k1)
        k3 = f(t + dt / 2, psi + dt / 2 * k2); k4 = f(t + dt, psi + dt * k3)
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        n = np.linalg.norm(psi)
        if not np.isfinite(n) or n == 0:
            return None
        psi /= n
    return psi


def evolve2(A, B, hfun, tau, psi0, nsteps, Acd=None):
    """RK4 for i dpsi/dt = [A + h(t)B + CD] psi.  Acd(t) -> sparse/dense CD operator or None."""
    dt = tau / nsteps
    psi = psi0.astype(complex).copy(); psi /= np.linalg.norm(psi)

    def f(tt, v):
        w = A @ v + hfun(tt) * (B @ v)
        if Acd is not None:
            w = w + Acd(tt) @ v
        return -1j * w

    for k in range(nsteps):
        t = k * dt
        k1 = f(t, psi); k2 = f(t + dt / 2, psi + dt / 2 * k1)
        k3 = f(t + dt / 2, psi + dt / 2 * k2); k4 = f(t + dt, psi + dt * k3)
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        n = np.linalg.norm(psi)
        if not np.isfinite(n) or n == 0:
            return None
        psi /= n
    return psi


def diagnostics(psi, E, R, Li, i0, H):
    c = Li @ psi                                  # <L_m|psi>
    PR = abs(c[i0]) ** 2 / np.sum(abs(c) ** 2)
    num = np.vdot(psi, H @ psi); Emean = (num / np.vdot(psi, psi))
    dE = Emean - E[i0]
    Fov = abs(np.vdot(R[:, i0], psi)) ** 2 / (np.vdot(R[:, i0], R[:, i0]).real)
    return PR, dE, Fov


# ------------------------------------------------------------------ Krylov ground-state AGP
def ip(X, Y):                       # Frobenius/Hilbert-Schmidt, normalised
    return np.vdot(X, Y) / X.shape[0]


def arnoldi_ops(H, X0, nvec):
    """orthonormal Krylov operators K_0..K_{nvec-1} for L(.)=[H,.] from X0."""
    K = [X0 / np.sqrt(ip(X0, X0).real)]
    for n in range(1, nvec):
        A = H @ K[-1] - K[-1] @ H
        for j in range(n):
            A = A - ip(K[j], A) * K[j]
        for j in range(n):         # reorthogonalise
            A = A - ip(K[j], A) * K[j]
        nrm = np.sqrt(ip(A, A).real)
        if nrm < 1e-12:
            break
        K.append(A / nrm)
    return K


def agp_exact_gs(E, R, Li, i0, dH):
    """A|R0> exactly, from Eq. (20): A = -i sum_{n!=0} (<Ln|dH|R0>/(En-E0)) |Rn><L0|."""
    v = Li @ dH @ R[:, i0]
    d = E - E[i0]; d[i0] = np.inf
    c = -1j * v / d; c[i0] = 0.0
    return np.outer(R @ c, Li[i0, :])


def agp_krylov_gs(H, dH, E, R, Li, i0, M):
    """ground-state-tailored truncated AGP, Eqs. (F1)-(F3), on the Arnoldi basis.  Returns A^{(M)}, alphas, residual norm, |v|."""
    K = arnoldi_ops(H, dH.astype(complex), 2 * M)
    odd = [K[2 * k - 1] for k in range(1, M + 1) if 2 * k - 1 < len(K)]
    Mx = len(odd)
    R0 = R[:, i0]; L0 = Li[i0, :].conj()                  # <L0| as ket

    def Q(x):
        return x - R0 * np.vdot(L0, x)

    u = [Q((H @ P - P @ H) @ R0) for P in odd]
    v = -1j * Q(dH @ R0)
    w = [Q(P @ R0) for P in odd]
    Mm = np.array([[np.vdot(w[m], u[k]) for k in range(Mx)] for m in range(Mx)])
    rhs = np.array([np.vdot(w[m], v) for m in range(Mx)])
    al = np.linalg.lstsq(Mm, rhs, rcond=None)[0]
    A = sum(al[k] * odd[k] for k in range(Mx))
    res = np.linalg.norm(sum(al[k] * u[k] for k in range(Mx)) - v)
    return A, al, res, np.linalg.norm(v)


# ------------------------------------------------------------------ optimal-ramp schedule (App. F, Eq. F8)
def _u(g, mu, nu1, nu2):
    den = np.maximum(nu1 * mu + nu2, 1e-300)
    return np.sqrt(np.maximum(g, 1e-300) / den)


def _fit_tau(hs, g, mu, nu1, tau):
    """choose nu2 so that int u dh = tau (bounded bisection in log nu2)."""
    lo, hi = -60., 60.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = np.trapezoid(_u(g, mu, nu1, np.exp(mid)), hs)
        if not np.isfinite(val):
            hi = mid; continue
        if val > tau:
            lo = mid
        else:
            hi = mid
    nu2 = np.exp(0.5 * (lo + hi))
    u = _u(g, mu, nu1, nu2)
    u *= tau / np.trapezoid(u, hs)
    return u


def solve(hs, g, mu, tau, Gtol):
    """min int g/u dh  s.t.  int u dh = tau and int mu u dh <= Gtol.
       EL:  u = sqrt( g/(nu1*mu + nu2) )."""
    mu = np.maximum(np.asarray(mu, float), 0.0)
    u0 = _fit_tau(hs, g, mu, 0.0, tau); G0 = np.trapezoid(mu * u0, hs)
    if G0 <= Gtol:
        return u0, G0, 0.0
    lo, hi = -20., 80.                       # log nu1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        u = _fit_tau(hs, g, mu, np.exp(mid), tau)
        G = np.trapezoid(mu * u, hs)
        if G > Gtol:
            lo = mid
        else:
            hi = mid
    nu1 = np.exp(0.5 * (lo + hi)); u = _fit_tau(hs, g, mu, nu1, tau)
    return u, np.trapezoid(mu * u, hs), nu1


def schedule(hs, u, tau):
    from scipy.interpolate import interp1d
    t = np.concatenate([[0.], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(hs))]); t = t / t[-1] * tau
    return interp1d(t, hs, bounds_error=False, fill_value=(hs[0], hs[-1]))

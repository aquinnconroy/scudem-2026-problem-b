import os
import numpy as np
import matplotlib.pyplot as plt

# pair states (A's mode, B's mode); same 9 boxes as the diagram (peace)
# plus the feud copy: same boxes with the tip arrows off, so only 5 are reachable (suffix f)
STATES = ['(S,S)', '(S,Ea)', '(Ea,S)', '(Ea,Ea)', '(T,Ea)', '(Ea,T)', '(T,S)', '(S,T)', '(Es,Es)',
          '(S,S)f', '(S,Ea)f', '(Ea,S)f', '(Ea,Ea)f', '(Es,Es)f']

# chosen values, matching ScudemIdeating.py where it has them
PARAMS = {
    'A': 100**2,     # forest area [L^2]
    'n': 13,         # number of good patches
    'w': 3,          # patch width (gaussian sigma) [L]
    'amp': 10,       # peak density of a fresh patch [food/L^2]
    'm': 0.5,        # threshold density a monkey notices [food/L^2]
    'v': 0.5,        # travel speed [L/T]
    'r': 2,          # vision radius [L]
    'c': 0.03,       # eat rate [1/T]; slow, so a patch outlasts a trip (the sim uses 1)
    'kappa': 0.01,   # rate a patch vanishes on its own: rot, fall, competitors [1/T] (about mu)
    'd': 10,         # pair separation [L]
    'C': 10,         # channel capacity [bits/T]
    'b': 10,         # bits per tip
    'sigma0': 0.05,  # tip blur per unit distance [radians]
    'R_c': 30,       # contact range [L]
    'pA': 1.0,       # accuracy of tips A sends (1 - p_lie)
    'pB': 1.0,       # accuracy of tips B sends
    'phi': 1/1500,   # forgiveness rate: a feud lasts 1/phi on average [1/T]
}


def derived(p:dict) -> dict:
    '''parameters plus the derived rates from the notebook'''
    q = dict(p)
    q['lam'] = 2*p['r']*p['v']*p['n'] / p['A']            # discovery rate while searching [1/T]
    q['mu'] = p['c'] / np.log(p['amp']/p['m'])            # patch depletion rate, one eater [1/T]
    q['F'] = 2*np.pi*p['amp']*p['w']**2                   # food in a fresh patch [food]
    q['e'] = q['mu']*q['F']                               # intake rate while eating [food/T]
    q['cd'] = 1.0 if p['d'] <= p['R_c'] else 0.0          # contact factor c(d)
    q['o'] = 1/p['n'] + (1 - 1/p['n'])*np.exp(-p['d']/(p['r'] + p['w']))   # overlap o(d)
    q['a_left'] = max(p['A']/2**p['b'], np.pi*(p['sigma0']*p['d'])**2)    # area left after a tip [L^2]
    q['tau'] = p['d']/p['v'] + q['a_left']/(2*p['r']*p['v'])                # mean trip time [T]
    q['nu'] = p['C'] / p['b']                             # tip rate [1/T]
    return q


def deriv(P:np.ndarray, q:dict) -> np.ndarray:
    '''the 14 equations from the diagram: dP/dt = flow in - flow out
    a false tip (arrive, nothing there, partner still eating) starts a feud: no tips until forgiven'''
    SS, SE, ES, EE, TE, ET, TS, ST, Es, SSf, SEf, ESf, EEf, Esf = P
    lam, o, tau, phi = q['lam'], q['o'], q['tau'], q['phi']
    gone = q['mu'] + q['kappa']             # a patch with one eater ends: eaten out or vanished
    gone2 = 2*q['mu'] + q['kappa']          # a shared patch ends
    tip, pA, pB = q['nu']*q['cd'], q['pA'], q['pB']
    return np.array([
        # (S,S): a find in shared ground (prob o) is seen by both and shared
        gone*SE + gone*ES + gone2*Es + ST/tau + TS/tau + phi*SSf - lam*(2 - o)*SS,
        # (S,Ea)  (false tips from (T,Ea) now go to the feud copy)
        lam*(1 - o)*SS + gone*EE + phi*SEf - (lam + tip + gone)*SE,
        # (Ea,S)
        lam*(1 - o)*SS + gone*EE + phi*ESf - (lam + tip + gone)*ES,
        # (Ea,Ea)
        lam*(1 - o)*SE + lam*(1 - o)*ES + phi*EEf - 2*gone*EE,
        # (T,Ea)
        tip*SE + lam*TS - (1/tau + gone)*TE,
        # (Ea,T)
        tip*ES + lam*ST - (1/tau + gone)*ET,
        # (T,S)
        gone*TE - (1/tau + lam)*TS,
        # (S,T)
        gone*ET - (1/tau + lam)*ST,
        # (Es,Es)
        lam*o*SS + lam*o*SE + lam*o*ES + pB/tau*TE + pA/tau*ET + phi*Esf - gone2*Es,
        # feud copy: the silent model, plus false tips coming in and forgiveness (phi) going out
        # (S,S)f
        gone*SEf + gone*ESf + gone2*Esf - (lam*(2 - o) + phi)*SSf,
        # (S,Ea)f: A just found B's tip was false
        lam*(1 - o)*SSf + gone*EEf + (1 - pB)/tau*TE - (lam + gone + phi)*SEf,
        # (Ea,S)f
        lam*(1 - o)*SSf + gone*EEf + (1 - pA)/tau*ET - (lam + gone + phi)*ESf,
        # (Ea,Ea)f
        lam*(1 - o)*SEf + lam*(1 - o)*ESf - (2*gone + phi)*EEf,
        # (Es,Es)f
        lam*o*SSf + lam*o*SEf + lam*o*ESf - (gone2 + phi)*Esf,
    ])


def rk4(q:dict, P0:np.ndarray, T:float, dt:float):
    '''integrate dP/dt = deriv(P) with classic RK4; returns times and P at each time'''
    f = lambda P: deriv(P, q)
    steps = int(T / dt)
    t = np.linspace(0, steps*dt, steps + 1)
    out = np.zeros((steps + 1, len(P0)))
    out[0] = P = P0
    for k in range(steps):
        k1 = f(P)
        k2 = f(P + dt/2*k1)
        k3 = f(P + dt/2*k2)
        k4 = f(P + dt*k3)
        P = P + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        out[k + 1] = P
    return t, out


def steady_state(q:dict) -> np.ndarray:
    '''
    set every dP/dt = 0: 14 linear equations in the 14 P's, written M @ P = 0
    they sum to zero, so one is redundant; swap it for sum(P) = 1 and solve
    '''
    # column j of M is deriv of the j-th unit vector (the equations are linear)
    M = np.column_stack([deriv(np.eye(len(STATES))[j], q) for j in range(len(STATES))])
    M[-1] = 1
    rhs = np.zeros(len(STATES))
    rhs[-1] = 1
    return np.linalg.solve(M, rhs)


def intake(P:np.ndarray, q:dict):
    '''food per time for A and B: e x P(that monkey is eating)'''
    SS, SE, ES, EE, TE, ET, TS, ST, Es, SSf, SEf, ESf, EEf, Esf = P
    return (q['e']*(ES + EE + ET + Es + ESf + EEf + Esf),
            q['e']*(SE + EE + TE + Es + SEf + EEf + Esf))


def feud_fraction(P:np.ndarray) -> float:
    '''fraction of time the pair is feuding'''
    return P[9:].sum()


def run(p:dict) -> tuple:
    '''steady-state intake for A and B'''
    q = derived(p)
    return intake(steady_state(q), q)


if __name__ == '__main__':
    os.makedirs('img', exist_ok=True)
    q = derived(PARAMS)
    print(f"lam = {q['lam']:.4g}  mu = {q['mu']:.4g}  kappa = {q['kappa']:.4g}  tau = {q['tau']:.4g}"
          f"  nu = {q['nu']:.4g}  o = {q['o']:.4g}  e = {q['e']:.4g}")

    # time evolution from both searching; check it lands on the steady state
    P0 = np.zeros(len(STATES))
    P0[0] = 1
    t, P = rk4(q, P0, T=3000, dt=0.5)
    Pss = steady_state(q)
    print('total probability stays 1:', np.allclose(P.sum(axis=1), 1))
    print('RK4 end matches steady state:', np.allclose(P[-1], Pss, atol=1e-6))
    print('\nsteady state:')
    for s, x in zip(STATES, Pss):
        print(f'  {s:8s} {x:.4f}')
    talk = run(PARAMS)
    silent = run(PARAMS | {'C': 0})
    print(f'\nintake A, B talking = {talk[0]:.4f}, {talk[1]:.4f}   silent = {silent[0]:.4f}')

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, s in enumerate(STATES):
        ax.plot(t, P[:, i], label=s)
    ax.set_xlabel('t')
    ax.set_ylabel('P')
    ax.legend(fontsize=7)
    fig.savefig('img/ctmc_time.png', dpi=150)
    plt.close(fig)

    # one-parameter sweeps: A's intake talking and silent at the same parameters
    sweeps = [('d', np.linspace(0.5, 50, 200)),
              ('b', np.arange(1, 25)),
              ('kappa', np.linspace(0, 2, 100)*q['mu'])]
    for name, xs in sweeps:
        talking = np.array([run(PARAMS | {name: x})[0] for x in xs])
        quiet = np.array([run(PARAMS | {name: x, 'C': 0})[0] for x in xs])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, talking, label='talking')
        ax.plot(xs, quiet, label='silent', ls='--')
        ax.set_xlabel(name)
        ax.set_ylabel("A's intake [food/T]")
        ax.legend()
        fig.savefig(f'img/ctmc_sweep_{name}.png', dpi=150)
        plt.close(fig)
        best = int(np.argmax(talking))
        print(f'{name:6s} best = {xs[best]:.3g}, talking/silent there = {talking[best]/quiet[best]:.4f}')

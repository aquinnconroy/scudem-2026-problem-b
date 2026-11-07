'''
analysis of the 9-state CTMC in ctmc.py (2026-11-07)
answers the problem's questions: distance vs accuracy, detail, deception, which factors matter
run: uv run --no-project --with numpy --with matplotlib python analysis.py
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from ctmc import PARAMS, STATES, derived, deriv, steady_state, intake, run, feud_fraction

NS = len(STATES)
NO_PUNISH = {'phi': 1e6}   # feuds end instantly = the old 9-state model

os.makedirs('img', exist_ok=True)
rng = np.random.default_rng(0)


def pair(p:dict) -> float:
    '''pair intake (A + B) at steady state'''
    a, b = run(p)
    return a + b


def gain(p:dict) -> float:
    '''talking / silent pair intake at the same parameters'''
    return pair(p) / pair(p | {'C': 0})


def generator(q:dict) -> np.ndarray:
    '''Q[i, j] = rate from state i to state j (row convention), read off deriv()'''
    M = np.column_stack([deriv(np.eye(NS)[j], q) for j in range(NS)])   # dP/dt = M P
    return M.T


def gillespie(q:dict, T:float) -> np.ndarray:
    '''simulate the jump process itself; return fraction of time in each state'''
    Q = generator(q)
    s, t, occ = 0, 0.0, np.zeros(NS)
    while t < T:
        rates = np.clip(Q[s], 0, None)
        rates[s] = 0
        total = rates.sum()
        wait = rng.exponential(1/total)
        occ[s] += min(wait, T - t)
        t += wait
        s = rng.choice(NS, p=rates/total)
    return occ / occ.sum()


q = derived(PARAMS)
print('--- 1. check: steady state vs a stochastic run of the same chain (p = 0.9 so feuds happen)')
qc = derived(PARAMS | {'pA': 0.9, 'pB': 0.9})
Psim = gillespie(qc, T=2e7)
for s, a, b in zip(STATES, steady_state(qc), Psim):
    print(f'  {s:8s} solve {a:.4f}   gillespie {b:.4f}')
Pss = steady_state(q)

# how much flow goes through the questionable (T,S) -lam-> (T,Ea) arrow
flux_stale = q['lam']*Pss[6]
flux_share = q['pB']/q['tau']*Pss[4]
print(f'  (T,S)->(T,Ea) flow / tip-share flow = {flux_stale/flux_share:.3f}')

print('\n--- 2. distance: d* and the overlap there')
ds = np.linspace(0.5, 40, 400)
talk = np.array([pair(PARAMS | {'d': d}) for d in ds])
quiet = np.array([pair(PARAMS | {'d': d, 'C': 0}) for d in ds])
k = int(np.argmax(talk))
dstar = ds[k]
o = derived(PARAMS | {'d': dstar})['o']
print(f'  d* = {dstar:.1f}, pair intake {talk[k]:.3f} vs silent {quiet[k]:.3f} (x{talk[k]/quiet[k]:.3f})')
print(f'  overlap o(d*) = {o:.3f}; as intersection/union o/(2-o) = {o/(2-o):.3f} (paper w* = 1/3)')
print(f'  silent best is at the far end: d = {ds[np.argmax(quiet)]:.1f}')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ds, talk, label='talking')
ax.plot(ds, quiet, '--', label='silent')
ax.axvline(PARAMS['R_c'], color='gray', lw=0.8)
ax.text(PARAMS['R_c'] + 0.5, ax.get_ylim()[0], 'contact range', fontsize=8, color='gray', va='bottom')
ax.set_xlabel('separation d [L]')
ax.set_ylabel('pair intake [food/T]')
ax.legend()
fig.tight_layout()
fig.savefig('img/an_d.png', dpi=150)
plt.close(fig)

# when does the pair want to sit at the edge of contact? sweep speed v
print('  d* as speed changes (travel cost):')
for v in [0.25, 0.5, 1, 2, 4, 8]:
    best = ds[int(np.argmax([pair(PARAMS | {'d': d, 'v': v}) for d in ds]))]
    print(f'    v = {v:<5} d* = {best:.1f}')

print('\n--- 3. distance x accuracy (both monkeys tell the truth with prob p)')
ps = np.linspace(0, 1, 41)
ds2 = np.linspace(0.5, 40, 80)
G = np.array([[gain(PARAMS | {'d': d, 'pA': p, 'pB': p}) for d in ds2] for p in ps])
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.pcolormesh(ds2, ps, G, cmap='RdBu', vmin=2 - G.max(), vmax=G.max(), shading='auto')
ax.contour(ds2, ps, G, levels=[1], colors='k', linewidths=1)
ax.set_xlabel('separation d [L]')
ax.set_ylabel('tip accuracy p')
fig.colorbar(im, label='talking / silent pair intake')
fig.tight_layout()
fig.savefig('img/an_d_by_p.png', dpi=150)
plt.close(fig)
# break-even accuracy: lowest p where talking still beats silence, at a few d
for d in [2, 5, 10, 20, 29]:
    g = np.array([gain(PARAMS | {'d': d, 'pA': p, 'pB': p}) for p in ps])
    pstar = ps[np.argmax(g >= 1)] if (g >= 1).any() else np.nan
    print(f'  d = {d:<3} break-even accuracy p* = {pstar:.3f}   gain at p=1: {g[-1]:.3f}')
# best d for each accuracy
for p in [1.0, 0.9, 0.8, 0.7]:
    best = ds2[int(np.argmax([pair(PARAMS | {'d': d, 'pA': p, 'pB': p}) for d in ds2]))]
    print(f'  p = {p}: best d = {best:.1f}')

print('\n--- 4. deception: B lies with prob 1 - pB, A always honest (with feuds, 1/phi = 1500)')
pbs = np.linspace(0, 1, 21)
IA = np.array([run(PARAMS | {'pB': p})[0] for p in pbs])
IB = np.array([run(PARAMS | {'pB': p})[1] for p in pbs])
IB0 = np.array([run(PARAMS | {'pB': p} | NO_PUNISH)[1] for p in pbs])
print(f'  no punishment: B at pB=0 {IB0[0]:.3f} vs pB=1 {IB0[-1]:.3f} -> lying pays')
print(f'  pB = 1: A {IA[-1]:.3f}, B {IB[-1]:.3f}')
print(f'  pB = 0: A {IA[0]:.3f}, B {IB[0]:.3f}')
print(f'  B gains by lying: {IB[0] > IB[-1]}; best pB for B = {pbs[np.argmax(IB)]:.2f}')
silentA = run(PARAMS | {'C': 0})[0]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(pbs, IA, label='A (honest, listens)')
ax.plot(pbs, IB, label='B (sends with accuracy pB)')
ax.plot(pbs, IB0, ':', color='C1', label='B, no punishment')
ax.plot(pbs, IA + IB, label='pair')
ax.axhline(silentA, color='gray', ls=':', label='each, silent')
ax.set_xlabel("B's tip accuracy pB")
ax.set_ylabel('intake [food/T]')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('img/an_lying.png', dpi=150)
plt.close(fig)

print('\n--- 5. level of communication: detail b and channel capacity C')
for d in [10, dstar]:
    bs = np.arange(1, 31)
    g = np.array([gain(PARAMS | {'d': d, 'b': b}) for b in bs])
    bfloor = np.log2(PARAMS['A'] / (np.pi*(PARAMS['sigma0']*d)**2))
    print(f'  d = {d:.1f}: best b = {bs[np.argmax(g)]}, blur floor log2(A/pi(s0 d)^2) = {bfloor:.1f};'
          f' gain b=1 {g[0]:.3f}, best {g.max():.3f}, b=30 {g[-1]:.3f}')
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
print('  gain vs C (b=10):', ' '.join(f'{C}:{gain(PARAMS | {"C": C}):.3f}' for C in Cs))
bs = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(6, 4))
for d in [5, 10, 20]:
    ax.plot(bs, [gain(PARAMS | {'d': d, 'b': b}) for b in bs], label=f'd = {d}')
ax.axhline(1, color='gray', lw=0.8)
ax.set_xlabel('bits per tip b')
ax.set_ylabel('talking / silent pair intake')
ax.legend()
fig.tight_layout()
fig.savefig('img/an_b.png', dpi=150)
plt.close(fig)

print('\n--- 6. the timescale: trip time vs patch lifetime')
# talking pays when a tip arrives before the patch is gone: group tau*(mu + kappa)
rows = []
for c in np.geomspace(0.003, 3, 25):
    for kr in [0.25, 0.5, 1, 2]:
        p = PARAMS | {'c': c}
        mu = derived(p)['mu']
        p['kappa'] = kr*mu
        qq = derived(p)
        rows.append((qq['tau']*(qq['mu'] + qq['kappa']), kr, gain(p)))
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(6, 4))
for kr in [0.25, 0.5, 1, 2]:
    r = rows[rows[:, 1] == kr]
    ax.semilogx(r[:, 0], r[:, 2], 'o-', ms=3, label=f'kappa/mu = {kr}')
ax.axhline(1, color='gray', lw=0.8)
ax.set_xlabel('trip time / patch lifetime   tau (mu + kappa)')
ax.set_ylabel('talking / silent pair intake')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('img/an_timescale.png', dpi=150)
plt.close(fig)
for kr in [0.25, 1, 2]:
    r = rows[rows[:, 1] == kr]
    cross = r[np.argmax(r[:, 2] < 1), 0] if (r[:, 2] < 1).any() else np.nan
    print(f'  kappa/mu = {kr}: talking stops paying near tau(mu+kappa) = {cross:.2f}; max gain {r[:, 2].max():.3f}')
print(f'  baseline tau(mu+kappa) = {q["tau"]*(q["mu"] + q["kappa"]):.2f}')

print('\n--- 7. which factors matter: elasticities at the baseline (d = d*)')
base = PARAMS | {'d': dstar, 'pA': 0.95, 'pB': 0.95}   # accuracy below 1 so it has room both ways
names = ['d', 'pB', 'b', 'C', 'v', 'c', 'kappa', 'r', 'w', 'n', 'A', 'sigma0']
h = 0.01
out = []
for nm in names:
    x = base[nm]
    up, dn = base | {nm: x*(1 + h)}, base | {nm: x*(1 - h)}
    if nm == 'pB':       # accuracy is symmetric in the pair
        up['pA'], dn['pA'] = up['pB'], dn['pB']
    if nm == 'c':        # keep kappa = mu when the eat rate changes
        up['kappa'] = base['kappa']*(1 + h)
        dn['kappa'] = base['kappa']*(1 - h)
    e_int = (np.log(pair(up)) - np.log(pair(dn))) / (2*h)
    e_gain = (np.log(gain(up)) - np.log(gain(dn))) / (2*h)
    out.append((nm, e_int, e_gain))
print(f'  {"param":7s} {"intake":>8s} {"talk gain":>10s}   (% change per 1% change)')
for nm, a, b in sorted(out, key=lambda t: -abs(t[2])):
    print(f'  {nm:7s} {a:8.3f} {b:10.3f}')
fig, ax = plt.subplots(figsize=(6, 4))
srt = sorted(out, key=lambda t: abs(t[2]))
ax.barh([t[0] for t in srt], [t[2] for t in srt])
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('elasticity of talking/silent gain')
fig.tight_layout()
fig.savefig('img/an_elasticity.png', dpi=150)
plt.close(fig)

print('\n--- 8. punishment: how long must a grudge last, and what does it cost honest pairs?')
Ls = np.array([30, 100, 300, 600, 1000, 1500, 2500, 4000, 7000])
fine = np.linspace(0, 1, 41)
bestp, cost = [], {0.95: [], 0.9: [], 0.8: []}
talk_benefit = pair(PARAMS) - pair(PARAMS | {'C': 0})
for L in Ls:
    IBs = [run(PARAMS | {'pB': x, 'phi': 1/L})[1] for x in fine]
    bestp.append(fine[int(np.argmax(IBs))])
    for p in cost:
        honest = PARAMS | {'pA': p, 'pB': p}
        cost[p].append((pair(honest | NO_PUNISH) - pair(honest | {'phi': 1/L})) / talk_benefit)
for k, L in enumerate(Ls):
    print(f'  grudge {L:<5} B best pB = {bestp[k]:.2f}   honest-error cost (share of talking benefit):'
          + ''.join(f'  p={p}: {cost[p][k]:.0%}' for p in cost))
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.semilogx(Ls, bestp, 'k-o', ms=3, label="liar's best accuracy")
ax1.set_xlabel('grudge length 1/phi [T]')
ax1.set_ylabel("B's best accuracy pB (A honest)")
ax2 = ax1.twinx()
for p in cost:
    ax2.semilogx(Ls, cost[p], '--', label=f'cost, honest errors p = {p}')
ax2.set_ylabel('share of talking benefit lost')
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc='center left')
fig.tight_layout()
fig.savefig('img/an_punish.png', dpi=150)
plt.close(fig)

print('\n--- 8b. patch size: talking gain and overlap at the best d, vs kappa/mu')
krs = np.array([0.1, 0.3, 1, 2, 4, 8, 15, 40])
mu = q['mu']
dsr = np.linspace(0.2, 29.9, 300)
G, W, D = [], [], []
for kr in krs:
    p = PARAMS | {'kappa': kr*mu}
    I = np.array([pair(p | {'d': d}) for d in dsr]); k = int(np.argmax(I))
    o = derived(p | {'d': dsr[k]})['o']
    G.append(I[k] / pair(p | {'d': dsr[k], 'C': 0})); W.append(o/(2 - o)); D.append(dsr[k])
    print(f'  kappa/mu = {kr:<4} d* = {dsr[k]:4.1f}  gain {G[-1]:.3f}  w = {W[-1]:.3f}')
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.semilogx(krs, G, 'C0-o', ms=3)
ax1.axhline(1, color='gray', lw=0.8)
ax1.set_xlabel('patch size vs one eater: kappa / mu')
ax1.set_ylabel('talking / silent pair intake at best d', color='C0')
ax2 = ax1.twinx()
ax2.semilogx(krs, W, 'C1--o', ms=3)
ax2.axhline(1/3, color='C1', lw=0.6, ls=':')
ax2.text(krs[0], 1/3, "  paper's w* = 1/3", color='C1', fontsize=7, va='bottom')
ax2.set_ylabel('overlap at best d, intersection/union', color='C1')
ax2.set_ylim(0.05, 0.37)
fig.tight_layout()
fig.savefig('img/an_patchsize.png', dpi=150)
plt.close(fig)

print('\n--- 9. what sets d*: the overlap length l = r + w')
for w in [1, 3, 5, 8, 12, 20]:
    l = PARAMS['r'] + w
    best = ds[int(np.argmax([pair(PARAMS | {'d': d, 'w': w}) for d in ds]))]
    print(f'  l = {l:<3} d* = {best:.1f}  d*/l = {best/l:.2f}')

print('\n--- 10. factor ranges: swing in talking/silent gain over a plausible range (d = d*)')
base = PARAMS | {'d': dstar}
ranges = {'pB': (0.5, 1.0), 'd': (2, 29), 'b': (6, 20), 'C': (0.1, 10), 'kappa': (0.25*q['mu'], 2*q['mu']),
          'v': (0.25, 2), 'n': (6, 26), 'sigma0': (0.01, 0.2)}
for nm, (lo, hi) in ranges.items():
    def g(x):
        p = base | {nm: x}
        if nm == 'pB':
            p['pA'] = x
        return gain(p)
    print(f'  {nm:7s} {lo:<7.3g}-> {hi:<7.3g} gain {g(lo):.3f} -> {g(hi):.3f}')

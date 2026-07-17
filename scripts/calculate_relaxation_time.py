from sdcc.treatment import HoldStep, VRMStep
from sdcc.simulation import parallelized_mono_dispersion
from sdcc.analysis import blocking_temperature
from sdcc.particles import load_particle
import numpy as np


def logprt_calc(particle, T):
    init_hold = HoldStep(0, T, 40, np.array([1, 0, 0]), hold_steps=3, hold_time=1)
    demag = HoldStep(1 + 1e-12, T, 0, np.array([1, 0, 0]), hold_steps=2, hold_time=100)
    steps = [init_hold, demag]
    n_steps = len(particle.get_params(20)["min_e"])
    start_p = np.full(n_steps, 1 / n_steps)
    vs, ps = parallelized_mono_dispersion(
        start_p,
        particle.info_dict["Minimum Size"],
        steps,
        particle,
        n_dirs=30,
        ctx="spawn",
        eq=[True, False],
    )
    m_rem = np.linalg.norm(vs[1][-1]) / np.linalg.norm(vs[0][-1])
    return m_rem


def find_Tb(particle):
    uniques = np.unique(particle.get_params(20)["bar_e"])
    uniques = uniques[~np.isinf(uniques)]
    Ts = []
    m_rems = []

    for unique in uniques:
        where = np.where(particle.get_params(20)["bar_e"] == unique)
        i = where[0][0]
        j = where[1][0]
        est_Tb = blocking_temperature(
            particle, particle.info_dict["Minimum Size"], i, j
        )
        T = est_Tb
        m_rem = logprt_calc(particle, T)
        Ts.append(T)
        m_rems.append(m_rem)

    if min(Ts) < 20.5 and max(m_rems) < 1 / np.e**2:
        return 20.0

    # Bisection method
    stop_crit = False
    m_rems_array = np.array(m_rems)
    T = np.array(Ts)[
        (m_rems_array - 1 / np.e**2) ** 2 == np.amin((m_rems_array - 1 / np.e**2) ** 2)
    ][0]
    if max(m_rems) < 1 / np.e**2:
        diff = -10
    elif min(m_rems) > 1 / np.e**2:
        diff = 10
    else:
        pass
    while ((m_rem - 1 / np.e**2) ** 2 > 1e-4) and (stop_crit == False):
        m_rems_array = np.array(m_rems)[np.argsort(Ts)]
        Ts_array = np.sort(Ts)
        i = np.where(Ts_array == T)[0][0]
        if max(m_rems) < 1 / np.e**2:
            pass
        elif min(m_rems) > 1 / np.e**2:
            pass
        elif m_rems_array[i] < 1 / np.e**2:
            diff = (Ts_array[i - 1] - Ts_array[i]) / 2
        elif m_rems_array[i] > 1 / np.e**2:
            diff = (Ts_array[i + 1] - Ts_array[i]) / 2

        if np.abs(diff) < 1:
            T += diff
            stop_crit = True
        elif T + diff < particle.T_min or T + diff > particle.T_max:
            diff /= 2
            print(diff)
        else:
            T += diff
            m_rem = logprt_calc(particle, T)
            print(T, m_rem)
            Ts.append(T)
            m_rems.append(m_rem)
    return np.round(T, 0)


def get_tr(particle):
    init_hold = HoldStep(0, 20, 40, np.array([1, 0, 0]), hold_steps=2, hold_time=1)
    demag = VRMStep(
        1 + 1e-12, 20, 0, np.array([1, 0, 0]), hold_time=1e18, hold_steps=100
    )
    steps = [init_hold, demag]
    n_steps = len(particle.get_params(20)["min_e"])
    start_p = np.full(n_steps, 1 / n_steps)
    vs, ps = parallelized_mono_dispersion(
        start_p,
        particle.info_dict["Minimum Size"],
        steps,
        particle,
        n_dirs=30,
        eq=[True, False],
        ctx="spawn",
    )
    m_rem = np.linalg.norm(vs[1], axis=1) / np.linalg.norm(vs[0][-1])
    tr = np.interp(1 / np.e**2, np.flip(m_rem), np.log(np.flip(demag.ts - (1 + 1e-12))))
    return np.exp(tr)


if __name__ == "__main__":
    particle = load_particle(
        "../../smelt_files/Ellipsoid_TM00_PRO_1.44_OBL_1.00_180nm_hard.smelt"
    )
    t = get_tr(particle)
    T = find_Tb(particle)
    print(t, T)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def thellier_to_xy(results_file, dist_file):
    # Read Thellier file
    results = pd.read_csv(results_file)
    # Read particle distribution file
    dist = pd.read_csv(dist_file)

    # Get raw data summing over experiments
    for i in range(len(dist)):
        row = dist.loc[i]
        p_filter = (
            (results["PRO"] == row["PRO"])
            & (results["OBL"] == row["OBL"])
            & (results["size"] == row["size"])
        )
        if i == 0:
            l = len(results.loc[p_filter])
            moments = np.zeros((l, 3))

        particle_result = results.loc[p_filter, "Mx":"Mz"].values * row["particle_n"]
        moments += particle_result

    # Calculate NRM, moments and temperatures
    particle_df = results.loc[p_filter]
    NRMs = moments[particle_df["step type"] == "Z"]
    pTRMs = moments[particle_df["step type"] == "I"]
    pTRMs = np.append([NRMs[0]], pTRMs, axis=0)
    pTRM_checks = moments[particle_df["step type"] == "P"]
    Ts = particle_df.temperature.unique()

    # Calculate whether infield or zerofield first steps
    NRM_df = particle_df[particle_df["step type"] == "Z"]
    pTRM_df = particle_df[particle_df["step type"] == "I"]
    steptypes = []
    for i in range(len(NRM_df))[1:]:
        if NRM_df.index[i] < pTRM_df.index[i - 1]:
            steptypes.append(0)
        else:
            steptypes.append(1)
    steptypes = np.append(1, steptypes)

    # Calculate pTRM checks by vector subtraction.
    check_df = particle_df.loc[particle_df["step type"] == "P"]
    baselines = []
    pTRM_ys = []
    js = []
    for i in range(len(check_df)):
        heat_T = check_df.iloc[i]["temperature"]
        baseline_T = check_df.iloc[i]["check temperature"]
        baseline = moments[
            (particle_df["step type"] == "Z")
            & (particle_df["temperature"] == baseline_T)
        ][0]
        pTRM_ys.append(
            moments[
                (particle_df["step type"] == "Z")
                & (particle_df["temperature"] == heat_T)
            ][0]
        )
        baselines.append(baseline)
        j = np.where(Ts == baseline_T)[0][0]
        js.append(j)

    baselines = np.array(baselines)
    pTRM_ys = np.array(pTRM_ys)

    # Take vector norms and normalize
    NRM_norm = np.linalg.norm(NRMs, axis=1)
    pTRM_norm = np.linalg.norm(pTRMs - NRMs, axis=1)
    pTRM_check_norm = np.linalg.norm(pTRM_checks - baselines, axis=1)
    pTRM_norm_y = np.linalg.norm(pTRM_ys, axis=1)
    pTRM_norm /= NRM_norm[0]
    pTRM_check_norm /= NRM_norm[0]
    pTRM_norm_y /= NRM_norm[0]
    NRM_norm /= NRM_norm[0]
    return (
        Ts,
        js,
        NRMs,
        pTRMs,
        NRM_norm,
        pTRM_norm,
        pTRM_check_norm,
        pTRM_norm_y,
        steptypes,
    )


def plot_arai(Ts, js, NRM_norm, pTRM_norm, pTRM_check_norm, pTRM_norm_y, steptypes, ax):
    ax.plot(pTRM_norm, NRM_norm, "k")
    ax.plot(
        pTRM_norm[steptypes == 0],
        NRM_norm[steptypes == 0],
        "o",
        markeredgecolor="k",
        markerfacecolor="r",
        markersize=10,
    )
    ax.plot(
        pTRM_norm[steptypes == 1],
        NRM_norm[steptypes == 1],
        "o",
        markeredgecolor="k",
        markerfacecolor="b",
        markersize=10,
    )
    ax.plot([0, 1], [1, 0], "g")
    for i, j in enumerate(js):
        ax.plot(
            [pTRM_norm[j], pTRM_check_norm[i], pTRM_check_norm[i]],
            [NRM_norm[j], NRM_norm[j], pTRM_norm_y[i]],
            "grey",
            lw=1,
        )
    ax.plot(
        pTRM_check_norm,
        pTRM_norm_y,
        "^",
        markerfacecolor="None",
        markeredgecolor="k",
        markersize=15,
    )

    for i in range(len(NRM_norm)):
        ax.text(pTRM_norm[i] + 0.01, NRM_norm[i] + 0.01, Ts[i])
    # thel_b,k = calc_b_and_k(NRM_norm,pTRM_norm)
    # ax.text(np.abs(1/thel_b),1,f'Arai plot slope: {thel_b:1.3f}\n Curvature: {k:1.3f}',ha='right',va='top');

    ax.set_xlabel("pTRM/NRM0")
    ax.set_ylabel("NRM/NRM0")
    return None


def frac_plot(results_file, dist_file, sd_lim, ax):
    # Read Thellier file
    results = pd.read_csv(results_file)
    # Read particle distribution file
    dist = pd.read_csv(dist_file)

    # Get raw data summing over experiments
    NRMs = np.zeros((len(dist), 3))
    for i in range(len(dist)):
        row = dist.loc[i]
        p_filter = (
            (results["PRO"] == row["PRO"])
            & (results["OBL"] == row["OBL"])
            & (results["size"] == row["size"])
        )
        particle_result = results.loc[p_filter, "Mx":"Mz"].values * row["particle_n"]
        NRMs[i] = np.linalg.norm(particle_result[0])
    NRMs = np.linalg.norm(NRMs, axis=1)
    NRMs /= np.sum(NRMs)

    ax.bar(
        dist["size"][:sd_lim],
        NRMs[:sd_lim] / 10,
        width=8,
        color="skyblue",
        label="SD Moment",
    )
    ax.bar(
        dist["size"][sd_lim:],
        NRMs[sd_lim:] / 10,
        width=8,
        color="tomato",
        label="SV Moment",
    )
    ax.bar(
        dist["size"],
        dist.particle_n,
        edgecolor="k",
        facecolor="None",
        width=8,
        label="Particle Number",
    )
    ax.legend()
    ax.set_xlabel("Size (nm)")
    ax.set_ylabel("Fraction")
    return None


def zijd_plot(NRMs, ax):
    ax.axvline(0, color="grey", lw=1)
    ax.axhline(0, color="grey", lw=1)
    ax.plot(NRMs[:, 0], NRMs[:, 2], "k")
    ax.plot(NRMs[:, 0], NRMs[:, 1], "k")
    ax.plot(NRMs[:, 0], NRMs[:, 2], "s", markerfacecolor="None", markeredgecolor="k")
    ax.plot(NRMs[:, 0], NRMs[:, 1], "ko")
    ax.axis("equal")
    ax.set_ylabel("y/z (Am$^2$)")
    ax.set_xlabel("x (Am$^2$)")
    return None


def dmag_plot(Ts, pTRM_norm, NRM_norm, ax):
    ax.plot(Ts, pTRM_norm, color="r", marker="o")
    ax.plot(Ts, NRM_norm, color="b", marker="s")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Relative Magnetization")
    return None


data_file = "data/TM30_P1.csv"  # Can be changed for other distributions

dist_file = (
    "data/TM30_Particle_Distribution.csv"  # Can be changed for other distributions
)

sd_limit = 8  # 5 for TM00

Ts, js, NRMs, pTRMs, NRM_norm, pTRM_norm, pTRM_check_norm, pTRM_norm_y, steptypes = (
    thellier_to_xy(data_file, dist_file)
)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

frac_plot(data_file, dist_file, sd_limit, ax[0, 0])

dmag_plot(Ts, pTRM_norm, NRM_norm, ax[0, 1])

plot_arai(
    Ts, js, NRM_norm, pTRM_norm, pTRM_check_norm, pTRM_norm_y, steptypes, ax[1, 0]
)

zijd_plot(NRMs, ax[1, 1])

plt.show()

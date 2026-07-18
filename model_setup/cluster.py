import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from matplotlib.colors import BoundaryNorm
import os
from itertools import combinations, permutations, product

import numpy as np


def nearest_111(Mag_x, Mag_y, Mag_z, alpha, theta, phi):
    """
    Finds the angle between a magnetic moment vector and the nearest <1 1 1> direction.

    Parameters
    ----------
    Mag_x, Mag_y, Mag_z: floats
    Components of the magnetic moment vector

    alpha,theta,phi: floats
    Euler angles by which magnetocrystalline anisotropy was rotated.

    Returns
    -------
    angle: float
    Angle (in radians) to the nearest direction.
    """
    special_111s = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ]
    )
    special_111s = special_111s / np.sqrt(3)
    vector_dir = np.array([Mag_x, Mag_y, Mag_z])
    vector_dir /= np.linalg.norm(vector_dir)

    rot = Rotation.from_euler(seq="XYZ", angles=np.array([alpha, theta, phi]))
    cosangles = []
    for d in special_111s:
        mat = rot.as_matrix()
        d = np.matmul(mat, d)
        cosangles.append(np.dot(d, vector_dir))
    return np.arccos(max(cosangles))


def nearest_110(Mag_x, Mag_y, Mag_z, alpha, theta, phi):
    """
    Finds the angle between a magnetic moment vector and the nearest <1 1 0> direction.

    Parameters
    ----------
    Mag_x, Mag_y, Mag_z: floats
    Components of the magnetic moment vector

    alpha,theta,phi: floats
    Euler angles by which magnetocrystalline anisotropy was rotated.

    Returns
    -------
    angle: float
    Angle (in radians) to the nearest direction.
    """
    special_110s = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
            [-1, -1, 0],
            [-1, 0, -1],
            [0, -1, -1],
        ]
    )
    special_110s = special_110s / np.sqrt(2)
    vector_dir = np.array([Mag_x, Mag_y, Mag_z])
    vector_dir /= np.linalg.norm(vector_dir)
    rot = Rotation.from_euler(seq="XYZ", angles=np.array([alpha, theta, phi]))
    cosangles = []
    for d in special_110s:
        mat = rot.as_matrix()
        d = np.matmul(mat, d)
        cosangles.append(np.dot(d, vector_dir))
    return np.arccos(max(cosangles))


def nearest_100(Mag_x, Mag_y, Mag_z, alpha, theta, phi):
    """
    Finds the angle between a magnetic moment vector and the nearest <1 0 0> direction.

    Parameters
    ----------
    Mag_x, Mag_y, Mag_z: floats
    Components of the magnetic moment vector

    alpha,theta,phi: floats
    Euler angles by which magnetocrystalline anisotropy was rotated.

    Returns
    -------
    angle: float
    Angle (in radians) to the nearest direction.
    """
    special_100s = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    vector_dir = np.array([Mag_x, Mag_y, Mag_z])
    vector_dir /= np.linalg.norm(vector_dir)
    rot = Rotation.from_euler(seq="XYZ", angles=np.array([alpha, theta, phi]))
    cosangles = []
    for d in special_100s:
        mat = rot.as_matrix()
        d = np.matmul(mat, d)
        cosangles.append(np.dot(d, vector_dir))
    return np.arccos(max(cosangles))


def calculate_anisotropies(TMx):
    """
    Calculates which is the hard, intermediate and easy axis
    for a given titanomagnetite composition.

    Parameters
    ----------
    TMx: float
    Titanomagnetite composition (%)

    Returns
    -------
    sorted_axes: list of strings
    List of magnetocrystalline axes in the order:
    [easy, intermediate, hard]
    """
    TMx /= 100
    Tc = 3.7237e02 * TMx**3 - 6.9152e02 * TMx**2 - 4.1385e02 * TMx**1 + 5.8000e02
    Tnorm = 20 / Tc
    K1 = (
        1e4
        * (-3.5725e01 * TMx**3 + 5.0920e01 * TMx**2 - 1.5257e01 * TMx**1 - 1.3579e00)
        * (1 - Tnorm) ** (-6.3643e00 * TMx**2 + 2.3779e00 * TMx**1 + 3.0318e00)
    )
    K2 = (
        1e4
        * (
            1.5308e02 * TMx**4
            - 2.2600e01 * TMx**3
            - 4.9734e01 * TMx**2
            + 1.5822e01 * TMx**1
            - 5.5522e-01
        )
        * (1 - Tnorm) ** 7.2652e00
    )

    oneoneone = K1 / 3 + K2 / 27
    oneonezero = K1 / 4
    onezerozero = 0

    axes_names = np.array(["1 1 1", "1 1 0", "1 0 0"])
    axes_values = np.array([oneoneone, oneonezero, onezerozero])
    sorted_axes = axes_names[np.argsort(axes_values)]
    return sorted_axes


def dir_to_euler(x, x_prime):
    """
    Finds a set of Euler angles that rotate one vector to another.
    Please note that this is non-unique, multiple sets OBL_filter
    euler angles can produce the same result.

    Parameters
    ----------
    x: string
    Vector to be rotated from, string of floats separated by " "

    x_prime: string
    Vector to be rotated to, same format.

    Returns
    -------
    angles: array of floats
    Euler angles for rotation
    """
    a = np.array(x.split(" ")).astype(float)
    b = np.array(x_prime.split(" ")).astype(float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    theta = np.arccos(np.dot(a, b))
    v = np.cross(a, b)
    euler_vector = v / np.linalg.norm(v) * theta
    rot = Rotation.from_rotvec(euler_vector)
    angles = rot.as_euler(seq="XYZ", degrees=False)
    angles[np.isnan(angles)] = 0
    return angles


def trim_dataframe(infile, TM, PRO, OBL, size, temp, alignment):
    """
    Trims an existing dataframe to include summary statistics about LEM states
    for a single particle.

    Inputs
    ------
    infile: string
    File path of csv file containing data

    TM: float
    Titanomagnetite composition (%)

    PRO: float
    Prolateness (easy/intermediate axis ratio) of particle

    OBL: float
    Oblateness (easy/intermediate axis ratio) of particle

    size: float
    Size of particle (ESVD, nm)

    temp: float
    Temperature LEM states were calculated at.

    alignment: str
    Either "easy","intermediate" or "hard". Magnetocrystalline axis that aligns
    with particle major axis.
    """

    # Use anisotropy information to calculate the euler angles to check for.
    anis = calculate_anisotropies(TM)
    material = "TM " + str(TM).zfill(2)
    if alignment.lower() == "easy":
        anis = anis[0]
    elif alignment.lower() == "hard":
        anis = anis[-1]
    elif alignment.lower() == "intermediate":
        anis = anis[1]
    else:
        raise ValueError("anis must be one of: easy,intermediate,hard")

    alpha, theta, phi = np.round(dir_to_euler(anis, "1 0 0"), 8)

    # Read in file
    outfile = pd.read_csv(infile, index_col=0)

    # Filter file for particle
    PRO_filter = outfile.Prolateness == PRO
    OBL_filter = outfile.Oblateness == OBL
    mat_filter = outfile.Material == material
    size_filter = outfile.Size == size * 1e-3
    temp_filter = outfile.Temperature == temp
    alpha_filter = np.round(outfile.Alpha, 8) == alpha
    theta_filter = np.round(outfile.Theta, 8) == theta
    phi_filter = np.round(outfile.Phi, 8) == phi

    full_filter = (
        PRO_filter
        & OBL_filter
        & mat_filter
        & size_filter
        & temp_filter
        & alpha_filter
        & theta_filter
        & phi_filter
    )
    outfile = outfile[full_filter]
    print(len(outfile))

    # Add additional collumns
    outfile["Mag"] = np.linalg.norm(outfile.loc[:, "Mag x":"Mag z"], axis=1)

    s111s = []
    s110s = []
    s100s = []
    indices = []

    # Calculate nearest easy,int, hard axis to LEM state
    for index, row in outfile.iterrows():
        s111s.append(
            nearest_111(
                row["Mag x"],
                row["Mag y"],
                row["Mag z"],
                row["Alpha"],
                row["Theta"],
                row["Phi"],
            )
        )
        s110s.append(
            nearest_110(
                row["Mag x"],
                row["Mag y"],
                row["Mag z"],
                row["Alpha"],
                row["Theta"],
                row["Phi"],
            )
        )
        s100s.append(
            nearest_100(
                row["Mag x"],
                row["Mag y"],
                row["Mag z"],
                row["Alpha"],
                row["Theta"],
                row["Phi"],
            )
        )
        indices.append(index)

    outfile["s111"] = s111s
    outfile["s110"] = s110s
    outfile["s100"] = s100s

    outfile["v111"] = s111s
    outfile["v110"] = s110s
    outfile["v100"] = s100s

    return outfile


def make_features(cluster_df):
    """
    Creates an input array for clustering analysis from a dataframe.

    Inputs
    ------
    cluster_df: pandas dataframe
    Dataframe containing LEM summary statistics for a particle.

    Returns
    -------
    cluster_on: numpy array
    Array of summary statistics to be clustered
    """

    # Settings for scaling results
    volume = 4 / 3 * np.pi * (cluster_df["Size"] * 1e-6 / 2) ** 3
    hmin = 3.22e-7
    hmax = 5.28e-3
    magvolume = 4 / 3 * np.pi * (cluster_df["Size"] / 2) ** 3

    # Scale energy
    tot_energy = cluster_df["Total energy"] / volume
    percs = np.percentile(tot_energy, (1, 99))
    percs = percs[1] - percs[0]
    std = max(percs, 100)
    tot_energy = (tot_energy - np.mean(tot_energy)) / std

    # Scale helicity
    hrel = np.abs(cluster_df["Relative helicity"])
    hrel = (hrel - hmin) / (hmax - hmin)

    # Sum absolute energy values to divide by total.
    abstot = (
        np.abs(cluster_df["Anisotropy energy"])
        + np.abs(cluster_df["Exchange energy"])
        + np.abs(cluster_df["Demag energy"])
    )

    cluster_on = np.array(
        [
            cluster_df["Mag"] / magvolume,
            hrel,
            cluster_df["v111"],
            cluster_df["v110"],
            cluster_df["v100"],
            cluster_df["Anisotropy energy"] / abstot,
            cluster_df["Exchange energy"] / abstot,
            cluster_df["Demag energy"] / abstot,
            tot_energy,
        ]
    ).T
    return cluster_on


def cluster_structures(cluster_df, eps=0.1):
    """
    Clusters a set of LEM states into different domain structures
    (domain states without directional information).

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe containing LEM summary statistics for a particle

    eps: float
    eps parameter for DBSCAN clustering (leave at 0.1)

    Returns
    -------
    labels_: array of ints
    Array of labels of the same length of the dataframe, each label
    int corresponds to a cluster. 0 means no cluster.
    """
    cluster_on = make_features(cluster_df)
    db = DBSCAN(eps=eps).fit(cluster_on)
    return db.labels_


def plot_clusters(cluster_df, labels, xcol="Relative helicity", ycol="Mag", pca=False):
    """
    Plots results of clustering algorithm.

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe containing LEM summary statistics for a particle

    labels: array of ints
    Array of labels of the same length of the dataframe, each label
    int corresponds to a cluster. 0 means no cluster.

    xcol: str
    Name of dataframe column to display on x axis.

    ycol: str
    Name of dataframe column to display on y axis

    pca: bool
    If True, ignore xcol and ycol and display a pca of the data.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()

    # Set up axes
    if pca:
        pca = PCA(2)
        cluster_on = make_features(cluster_df)
        x, y = pca.fit_transform(cluster_on).T
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    else:
        x = cluster_df[xcol]
        y = cluster_df[ycol]
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    # Plot axes
    bounds = np.arange(np.min(labels) - 0.5, np.max(labels) + 1.5)
    Norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)

    s = ax.scatter(x, y, c=labels, cmap="tab10", norm=Norm)
    c = fig.colorbar(s, ax=ax, ticks=np.unique(labels), label="Cluster Number")

    return None


def cubic_rotations(rot, reflections=True):
    """
    Calculates a set of matrices that preserve the symmetry
    of the cubic magnetocrystalline anisotropy function when
    applied to a set of vector directions. e.g. a rotation
    of 90 degrees about the <1 0 0> axis or 120 degrees about
    the <1 1 1> axis.

    Parameters
    ----------
    rot: 3x3 matrix
    Rotation matrix applied to the magnetocrystalline anisotropy
    function. All symmetry matrices are applied to this matrix.

    reflections: bool
    If True, allow reflection as well as rotation matrices.

    Returns
    -------
    rot_mats: array of 3x3 matrices
    Set of symmetry preserving transformations. Note that if
    reflections = True, this included non-rotation matrices,
    despite the name.
    """

    # Generating function for cubic symmetry using permutations.
    I = np.eye(3)
    perms = np.array(list(permutations(I)))

    # Get matrices with -1 values in them.
    products = np.array(list(product([1, -1], repeat=3)))
    rot_mats = []

    # Generate matrices
    for per in perms:
        for prod in products:
            if reflections:
                rot_mats.append(rot @ (per * prod) @ np.linalg.inv(rot))
            else:
                if np.linalg.det(per * prod) >= 0:
                    rot_mats.append(rot @ (per * prod) @ np.linalg.inv(rot))

    rot_mats = np.array(rot_mats)
    return rot_mats


def ellipsoid_rotations(rot_mats, PRO, OBL):
    """
    Calculates a set of matrix transformations that preserve the symmetry
    of the demagnetization factor for an ellispoid.

    Parameters
    ----------
    rot_mats: array of 3x3 matrices
    Set of possible matrices provided by magnetocrystalline anisotropy

    PRO: float
    Prolateness (major/intermediate) of ellipsoid.

    OBL: float
    Oblateness (intermediate/minor) of ellipsoid.

    Returns
    -------
    rot_mats: array of 3x3 matrices
    Set of symmetry preserving transformations. Note that if
    reflections = True, this included non-rotation matrices,
    despite the name.
    """

    passed = []
    for rot_mat in rot_mats:
        # Calculate axis of rotation for rotation vector
        rot_vec = Rotation.from_matrix(rot_mat).as_rotvec()
        rot_dist = np.linalg.norm(rot_vec)
        rot_vec /= rot_dist

        # If a sphere, any transformation preserves symmetry.
        if (PRO == 1) and (OBL == 1):
            return rot_mats

        # If the rotation is about the z axis (short axis), then we preserve
        # symmetry in an oblate particle.
        elif PRO == 1:
            if np.isclose(np.abs(rot_mat[2, 2]), 1.0):
                passed.append(rot_mat)

        # If the rotation is about the x axis (long axis), then we preserve
        # symmetry in a prolate particle.
        elif OBL == 1:
            if np.isclose(np.abs(rot_mat[0, 0]), 1.0):
                passed.append(rot_mat)

        # If there is no rotation, or there is a rotation by 180 degrees about
        # x, y or z, then we preserve symmetry in all cases.
        if (np.isclose(rot_dist, np.pi) and np.isclose(np.amax(rot_vec), 1)) or (
            np.isclose(rot_dist, 0)
        ):
            passed.append(rot_mat)
    return np.unique(passed, axis=0)


def get_alt_dirs(base_dir, rot_mats, min_angle=2):
    """
    Given a vector direction, calculate other
    existing directions in a particle given a set of
    symmetry matrices for that particle.

    Parameters
    ----------
    base_dir: length 3 array
    Vector direction to generate from.

    rot_mats: array of 3x3 matrices
    Matrices that preserve symmetry for the particle.

    min_angle: float
    Tolerance for considering two directions "the same".
    Necessary when base_dir has some uncertainty due to
    being generated from a numerical model (MERRILL).
    """
    alt_dirs = []
    for R in rot_mats:
        alt_dirs.append(R @ base_dir)

    alt_dirs = np.array(alt_dirs)
    dot_prods = np.inner(alt_dirs, alt_dirs)
    same = dot_prods >= np.cos(np.radians(min_angle))
    groups = []
    for i in range(len(alt_dirs)):
        exists = False
        for group in groups:
            if i in group:
                exists = True
        if not exists:
            new_group = np.where(same[i] == True)[0]
            groups.append(new_group)
    alt_dirs_new = []
    for group in groups:
        alt_dirs_new.append(np.mean(alt_dirs[group], axis=0))
    alt_dirs_new = np.array(alt_dirs_new)
    alt_dirs_new /= np.linalg.norm(alt_dirs_new, axis=1, keepdims=True)
    return alt_dirs_new


def dfs(graph, node, visited):
    """
    Depth first search algorithm (recursive).

    Parameters
    ----------
    graph: n x n array of bools.
    Connectivity between states in graph.

    node: node to analyze

    visited: length n array
    array of visited nodes in connectivity graph

    Returns
    -------
    None (recursive function).
    """
    visited[node] = True
    for neighbor in range(len(graph)):
        if graph[node][neighbor] and not visited[neighbor]:
            dfs(graph, neighbor, visited)
    return None


def is_connected(graph):
    """
    Checks if a graph is fully connected or has isolated points

    Parameters
    ----------
    graph: nxn array of bools
    Connectivity between states

    Returns
    -------
    visited_all: bool
    Whether the graph is fully connected or not.
    """
    n = len(graph)
    visited = [False] * n
    dfs(graph, 0, visited)
    return all(visited)


def get_LEM_pairs(base_dir, rot_mats, min_angle=2, eps=0.01):
    """
    Calculates the pairs of states to calculate energy barriers
    between in the same LEM structure. Uses the symmetry structure
    of the particle to determine whether more than one pair of states
    is necessary to explore everything. This allows for particles with
    a single LEM state but multiple energy barriers to have all barrier
    calculations done (e.g. slightly elongate SD).

    Parameters
    ----------
    base_dir: length 3 array
    "base" moment vector associated with state.

    rot_mats: array of 3x3 matrices
    Symmetry preserving transformation matrices for a particle

    min_angle: float
    Tolerance (in degress) within which two states are considered
    to be "the same"

    eps: float
    eps value for clustering of states into different directions.

    Returns
    -------
    pair list: array
    List of directions for pairs of states
    """

    # Calculate possible directions for a given state.
    alts = get_alt_dirs(base_dir, rot_mats, min_angle)
    alts = np.array(alts)

    # Calculate distances between state pairs
    dot_prods = np.inner(alts, alts)

    # Calculate unique sets of distances to check for different
    # possible transitions between states.
    unique_prods = np.unique(dot_prods)

    # Clear transitions to the same state which have dot product 1
    unique_prods = unique_prods[~np.isclose(unique_prods, 1)]

    # Cluster the distance levels as there may be some variability.
    dbscan = DBSCAN(eps=eps, min_samples=1)
    cluster = dbscan.fit_predict(unique_prods.reshape(-1, 1))
    uniquer_prods = np.array(
        [np.amin(unique_prods[cluster == c]) for c in np.unique(cluster)]
    )

    # Sort by distance.
    uniquer_prods = np.sort(uniquer_prods)

    # Find the connectivity between states if we look at only the shortest transition.
    connectivity = (dot_prods >= uniquer_prods[-1]) & (~np.isclose(dot_prods, 1))

    # Are all states connected?
    if not is_connected(connectivity):
        # If not, loop through combinations of 2 states, then go to 3, etc...
        for i in range(2, len(uniquer_prods) + 1):
            # Find combination of states
            combos = np.array(list(combinations(uniquer_prods, i)))

            # Sort combinations by the total distances between states
            # Find the minimally distance state transitions that connect all states
            combosum = np.sum(combos, axis=1)
            idxs = np.argsort(combosum)
            combos = combos[idxs[::-1]]

            # Calculate the new connectivity for each combo.
            for c in combos:
                prods = []
                clusts = []
                for prod in c:
                    clust = cluster[unique_prods == prod][0]
                    prod_list = unique_prods[cluster == clust]
                    for p in prod_list:
                        prods.append(p)
                        clusts.append(clust)
                connectivity = np.isin(dot_prods, prods)

                # If all states are connected, stop checking
                if is_connected(connectivity):
                    break
            if is_connected(connectivity):
                break

    # If we start connected, then make our clusters and products for each cluster.
    else:
        clusts = cluster[unique_prods >= uniquer_prods[-1]]
        prods = unique_prods[cluster == clusts[0]]

    # Create lists of paired states
    clusts = np.array(clusts)
    prods = np.array(prods)
    pair_list = []
    for clust in np.unique(clusts):
        prod_subset = prods[clusts == clust]
        subgraph = np.isin(dot_prods, prod_subset)
        pairs = []
        for i in range(len(subgraph)):
            for j in range(i, len(subgraph)):
                if subgraph[i, j]:
                    pairs.append([alts[i], alts[j]])
        pairs = np.array(pairs)
        pair_list.append(pairs)
    return pair_list


def get_barrier_endpoints(cluster_df, labels, min_angle=2):
    """
    Calculate a set of LEM state pairs for calculating energy
    barriers in MERRILL.

    This has several parts:
    1. "Subcluster" initial domain structures by direction to
    obtain distinct domain states.

    2. Prune "bad" domain structures containing no domain states

    3. Find possible combinations of domain structures and assess
    which are the best examples of these structures to cluster
    between.

    4. Find possible combinations of domain states within a given
    structure and assess which are the best states to cluster between.

    5. Produce keys and pairs for states used for barrier calculation.

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe containing LEM summary statistics for a particle

    labels: array of ints
    Array of labels of the same length of the dataframe, each label
    int corresponds to a cluster. 0 means no cluster.

    min_angle: float
    Min tolerance within which two domain states are considered to be
    "the same".

    Returns
    -------
    repr_pairs: n x 2 array
    List of pairs of state IDs to calculate energy barriers between.

    repr_keys: length n dictionary
    Pairs state IDs to an alphanumerical description of the state.
    """

    # Get direction-based subclusters
    (
        cluster_df,
        reprs,
        reprs_pos,
        reprs_neg,
        min_energies_neg,
        min_energies_pos,
        mean_dirs,
        mean_dirs_pos,
        mean_dirs_neg,
        signs,
        isboths,
        idxs,
        min_energies,
    ) = subcluster_clusters(cluster_df, labels)

    # Remove bad structures
    bad_structures = prune_bad_structures(
        reprs_pos,
        reprs_neg,
        min_energies_neg,
        min_energies_pos,
        mean_dirs,
        mean_dirs_pos,
        mean_dirs_neg,
        signs,
        isboths,
        idxs,
        min_energies,
    )

    # Calculate pairs of structures
    chosen_reprs = find_structure_pairs(
        reprs_pos,
        reprs_neg,
        mean_dirs_pos,
        mean_dirs_neg,
        min_energies_pos,
        min_energies_neg,
        min_energies,
        signs,
        idxs,
    )

    # Calculate pairs of states
    chosen_reprs_r1, chosen_reprs_r2, chosen_reprs_m1, chosen_reprs_m2 = (
        subcluster_pairs(
            cluster_df,
            reprs,
            chosen_reprs,
            reprs_pos,
            reprs_neg,
            mean_dirs_pos,
            mean_dirs_neg,
            min_energies_pos,
            min_energies_neg,
            min_energies,
            isboths,
            min_angle=min_angle,
        )
    )

    # Make key-pair outputs for function
    repr_pairs, repr_key = make_keys_and_pairs(
        cluster_df,
        chosen_reprs,
        chosen_reprs_r1,
        chosen_reprs_r2,
        chosen_reprs_m1,
        chosen_reprs_m2,
    )
    return (np.array(repr_pairs), repr_key)


def get_cluster_group_idxs(cluster_df, i, min_angle=2):
    cluster_array = cluster_df[cluster_df.cluster == i]
    PRO = cluster_df.Prolateness.iloc[0]
    OBL = cluster_df.Oblateness.iloc[0]

    min_dir = cluster_array.loc[
        cluster_array["Total energy"] == np.amin(cluster_array["Total energy"]),
        "Mag x":"Mag z",
    ].values[0]
    min_dir /= np.linalg.norm(min_dir)
    angles = cluster_array.loc[:, "Alpha":"Phi"].values[0]
    rot = Rotation.from_euler(seq="XYZ", angles=angles).as_matrix()
    # print(rot)
    rot_mats = cubic_rotations(rot, reflections=False)
    rot_mats = ellipsoid_rotations(rot_mats, PRO, OBL)
    pairs = get_LEM_pairs(min_dir, rot_mats, min_angle=min_angle)
    # print(pairs)
    subcs = cluster_array["subcluster"].unique()
    subcs = subcs[~np.isnan(subcs)]
    mean_mags = []
    for c in subcs:
        subc = cluster_array[cluster_array["subcluster"] == c]

        subMag = subc.loc[:, "Mag x":"Mag z"].values
        subMag /= np.linalg.norm(subMag, axis=1, keepdims=True)
        subE = subc.loc[:, "Total energy"].values
        subE = np.exp((np.amin(subE) - subE) / (1.380649e-23 * 293))
        mean_mag = np.sum(subMag.T * subE, axis=1)
        mean_mag /= np.linalg.norm(mean_mag)
        mean_mags.append(mean_mag)
    mean_mags = np.array(mean_mags)
    idx_list = []
    for unique_bar in pairs:
        idxs = []
        for pair in unique_bar:
            dists = np.inner(pair, mean_mags)
            where = np.where(dists >= np.cos(np.radians(min_angle)))

            if len(where[1]) == 2:
                idxs.append(where[1])
            elif len(where[1]) == 1:
                loc = where[0][0]
                if loc == 0:
                    idxs.append(np.append(np.nan, where[1]))
                else:
                    idxs.append(np.append(where[1], np.nan))
            elif len(where[1]) == 0:
                idxs.append(np.array([np.nan, np.nan]))
        idxs = np.array(idxs)
        idx_list.append(idxs)
    return idx_list


def subcluster_clusters(cluster_df, labels):
    """
    "Subclusters" initial domain structures by direction to
    obtain distinct domain states.

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe containing LEM summary statistics for a particle

    labels: array of ints
    Array of labels of the same length of the dataframe, each label
    int corresponds to a cluster. 0 means no cluster.

    Returns
    -------
    cluster_df: pandas dataframe
    Dataframe updated with clusters and subclusters

    reprs: list of list of ints
    reprs[i] returns a list of direction-based states in cluster_df
    for particles with domain structure i

    reprs_pos: list of list of ints
    same as reprs, but for only states with positive directions

    reprs_neg: list of list of ints
    same as reprs, but for only states with negative directions.

    min_energies_neg: list of list of floats
    same as reprs_neg, but with total energy of the LEM state

    min_energies_pos: list of list of floats
    same as reprs_pos, but with total energy of the LEM state

    mean_dirs: list of list of length 3 arrays
    same as reprs, but with the mean LEM net magnetization direction

    mean_dirs_pos: list of list of length 3 arrays
    same as reprs_pos, but with the mean LEM net magnetization direction

    mean_dirs_neg: list of list of length 3 arrays
    same as reprs_neg, but with the mean LEM net magnetization direction

    signs: list of list of ints
    provides the sign of the helicity for each element in reprs

    isboths: list of list of bools
    says whether reprs[i] contains both positive and negative helicity states

    idxs: list of list of ints
    assigns indices to each subcluster in the structure

    min_energies:
    same as reprs, but with total energy of the LEM state
    """
    cluster_df["subcluster"] = np.nan
    cluster_df["cluster"] = labels
    cluster_df["subcluster"] = np.nan
    cluster_df["cluster"] = labels
    reprs = []
    mean_dirs = []
    signs = []
    isboths = []
    idxs = []
    min_energies = []
    reprs_pos = []
    reprs_neg = []
    mean_dirs_pos = []
    mean_dirs_neg = []
    min_energies_pos = []
    min_energies_neg = []
    for i in np.unique(labels[labels != (-1)]):
        structurecluster = cluster_df[labels == i]
        mags = structurecluster.loc[:, "Mag x":"Mag z"].values.T / np.linalg.norm(
            structurecluster.loc[:, "Mag x":"Mag z"], axis=1
        )
        mags = mags.T
        subclusters = DBSCAN(min_samples=2).fit(mags)
        js = subclusters.labels_
        repr = []
        mean_dir = []
        signl = []
        isboth = []
        idx = []
        min_en = []
        mean_dir_pos = []
        mean_dir_neg = []
        min_en_pos = []
        min_en_neg = []
        repr_pos = []
        repr_neg = []

        for j in np.unique(js[js != -1]):
            indices = structurecluster[js == j].index
            cluster_df.loc[indices, "subcluster"] = j
            meandir = np.mean(mags[js == j], axis=0)
            meandir /= np.linalg.norm(meandir)
            sign = np.sign(structurecluster[js == j]["Relative helicity"].values)
            dots = np.dot(mags[js == j], meandir)
            dots_new = np.copy(dots)
            dots[dots > 0] *= sign[dots > 0]
            energies = structurecluster[js == j]["Total energy"]

            min_energy = energies == np.amin(energies)
            min_dir = mags[js == j][min_energy][0]
            repr.append(indices[min_energy][0])
            signl.append(sign[min_energy][0])
            isboth.append(np.abs(np.sum(sign)) < len(sign))
            mean_dir.append(min_dir)
            idx.append(j)
            min_en.append(np.amin(energies))

            try:
                energies_pos = energies[sign == 1]
                min_pos = energies_pos == np.amin(energies_pos)
                mean_dir_pos.append(mags[js == j][sign == 1][min_pos][0])
                repr_pos.append(indices[sign == 1][min_pos][0])
                min_en_pos.append(np.amin(energies_pos))

            except:
                mean_dir_pos.append(np.array([np.nan, np.nan, np.nan]))
                repr_pos.append(np.nan)
                min_en_pos.append(np.nan)

            try:
                energies_neg = energies[sign == -1]
                min_neg = energies_neg == np.amin(energies_neg)
                mean_dir_neg.append(mags[js == j][sign == -1][min_neg][0])
                repr_neg.append(indices[sign == -1][min_neg][0])
                min_en_neg.append(np.amin(energies_neg))
            except:
                mean_dir_neg.append(np.array([np.nan, np.nan, np.nan]))
                repr_neg.append(np.nan)
                min_en_neg.append(np.nan)

        reprs.append(repr)
        reprs_pos.append(repr_pos)
        reprs_neg.append(repr_neg)
        min_energies_neg.append(min_en_neg)
        min_energies_pos.append(min_en_pos)
        mean_dirs.append(mean_dir)
        mean_dirs_pos.append(mean_dir_pos)
        mean_dirs_neg.append(mean_dir_neg)
        signs.append(signl)
        isboths.append(isboth)
        idxs.append(idx)
        min_energies.append(min_en)

    return (
        cluster_df,
        reprs,
        reprs_pos,
        reprs_neg,
        min_energies_neg,
        min_energies_pos,
        mean_dirs,
        mean_dirs_pos,
        mean_dirs_neg,
        signs,
        isboths,
        idxs,
        min_energies,
    )


def prune_bad_structures(
    reprs_pos,
    reprs_neg,
    min_energies_neg,
    min_energies_pos,
    mean_dirs,
    mean_dirs_pos,
    mean_dirs_neg,
    signs,
    isboths,
    idxs,
    min_energies,
):
    """
    Removes "bad" structures with no subclusters from the set of
    existing structures

    Parameters
    ----------
    reprs_pos: list of list of ints
    reprs[i] returns a list of states in cluster_df for particles
    with domain structure i and positive helicity

    reprs_neg: list of list of ints
    same as reprs_pos, but for only states with negative directions.

    min_energies_neg: list of list of floats
    same as reprs_neg, but with total energy of the LEM state

    min_energies_pos: list of list of floats
    same as reprs_pos, but with total energy of the LEM state

    mean_dirs: list of list of length 3 arrays
    same as reprs_pos, but with the mean LEM net magnetization
    direction and both helicities

    mean_dirs_pos: list of list of length 3 arrays
    same as reprs_pos, but with the mean LEM net magnetization direction

    mean_dirs_neg: list of list of length 3 arrays
    same as reprs_neg, but with the mean LEM net magnetization direction

    signs: list of list of ints
    provides the sign of the helicity for each element in reprs

    isboths: list of list of bools
    says whether reprs[i] contains both positive and negative helicity states

    idxs: list of list of ints
    assigns indices to each subcluster in the structure

    min_energies: list of list of floats
    same as reprs_pos, but with total energy of the LEM state and both helicities

    Returns
    -------
    bad_structures: array of ints
    lists removed structures
    """
    bad_structures = []
    for i in range(len(idxs)):
        if len(mean_dirs_pos[i]) == 0 and len(mean_dirs_neg[i]) == 0:
            print(f"Warning: had to delete structure {i}")
            bad_structures.append(i)
        else:
            pass
    bad_structures = np.array(bad_structures)
    for i in bad_structures:
        del mean_dirs_pos[i]
        del mean_dirs_neg[i]
        del idxs[i]
        del min_energies_pos[i]
        del min_energies_neg[i]
        del signs[i]
        del isboths[i]
        del min_energies[i]
        del reprs_pos[i]
        del reprs_neg[i]
        del mean_dirs[i]
        bad_structures -= 1
    return bad_structures


def find_structure_pairs(
    reprs_pos,
    reprs_neg,
    mean_dirs_pos,
    mean_dirs_neg,
    min_energies_pos,
    min_energies_neg,
    min_energies,
    signs,
    idxs,
):
    """
    Finds pairs of states in structures to minimize between for MERRILL
    energy barrier calculations

    Parameters
    -----------
    reprs_pos: list of list of ints
    same as reprs, but for only states with positive directions

    reprs_neg: list of list of ints
    same as reprs, but for only states with negative directions.

    mean_dirs_pos: list of list of length 3 arrays
    same as reprs_pos, but with the mean LEM net magnetization direction

    mean_dirs_neg: list of list of length 3 arrays
    same as reprs_neg, but with the mean LEM net magnetization direction

    min_energies_neg: list of list of floats
    same as reprs_neg, but with total energy of the LEM state

    min_energies_pos: list of list of floats
    same as reprs_pos, but with total energy of the LEM state

    min_energies:
    same as reprs, but with total energy of the LEM state

    signs: list of list of ints
    provides the sign of the helicity for each element in reprs

    isboths: list of list of bools
    says whether reprs[i] contains both positive and negative helicity states

    idxs: list of list of ints
    assigns indices to each subcluster in the structure

    Returns
    -------
    chosen_reprs: list of ints
    indices of "best examples" of domain structures (those which are lowest
    energy and closest to one another)
    """
    combos = []
    for c in product(*idxs):
        combos.append(list(c))

    sign_combos = []
    for c in product(*signs):
        sign_combos.append(list(c))

    combos = np.array(combos)
    # s_max = np.abs(np.sum(sign_combos,axis=1))
    # print(sign_combos)
    # sign_combos = np.array(sign_combos)[s_max==len(combos[0])]
    # combos = combos[s_max==len(combos[0])]
    # print(min_energies_pos,mean_dirs_pos)
    mean_combos_pos = np.array(
        [
            [mean_dirs_pos[j][k] for j, k in enumerate(combos[i])]
            for i in range(len(combos))
        ]
    )
    mean_combos_neg = np.array(
        [
            [mean_dirs_neg[j][k] for j, k in enumerate(combos[i])]
            for i in range(len(combos))
        ]
    )
    if len(mean_combos_pos) > 0:
        scores_pos = np.linalg.norm(np.sum(mean_combos_pos, axis=1), axis=1)
        scores2_pos = np.sum(
            [
                [
                    np.exp(
                        (np.amin(min_energies[j]) - min_energies_pos[j][k])
                        / (1.380649e-23 * 293.15)
                    )
                    for j, k in enumerate(combos[i])
                ]
                for i in range(len(combos))
            ],
            axis=1,
        )
    else:
        scores_pos = []
        scores2_pos = []
    if len(mean_combos_neg) > 0:
        scores_neg = np.linalg.norm(np.sum(mean_combos_neg, axis=1), axis=1)
        scores2_neg = np.sum(
            [
                [
                    np.exp(
                        (np.amin(min_energies[j]) - min_energies_neg[j][k])
                        / (1.380649e-23 * 293.15)
                    )
                    for j, k in enumerate(combos[i])
                ]
                for i in range(len(combos))
            ],
            axis=1,
        )
    else:
        scores_neg = []
        scores2_neg = []

    scores = np.append(scores_pos, scores_neg)
    scores2 = np.append(scores2_pos, scores2_neg)
    score_cluster = DBSCAN(min_samples=2, eps=0.1).fit(
        np.nan_to_num(scores.reshape(-1, 1), nan=0)
    )
    score_clusters = score_cluster.labels_
    unique_clusters = np.unique(score_clusters)
    cluster_means = np.array(
        [np.mean(scores[score_clusters == label]) for label in unique_clusters]
    )
    biggest_mean = unique_clusters[cluster_means == np.nanmax(cluster_means)][0]
    filtered_scores2 = scores2[score_clusters == biggest_mean]

    chosen_idx = np.array(list(range(len(combos) * 2)))[
        scores2 == np.nanmax(filtered_scores2)
    ][0]
    if chosen_idx >= len(combos):
        chosen_idx -= len(combos)
        repr_sign = reprs_neg
    else:
        repr_sign = reprs_pos
    chosen_idx = combos[chosen_idx]
    chosen_reprs = [repr_sign[i][j] for i, j in enumerate(chosen_idx)]

    return chosen_reprs


def subcluster_pairs(
    cluster_df,
    reprs,
    chosen_reprs,
    reprs_pos,
    reprs_neg,
    mean_dirs_pos,
    mean_dirs_neg,
    min_energies_pos,
    min_energies_neg,
    min_energies,
    isboths,
    min_angle=2,
):
    """
    Finds pairs of states in structures to minimize between for MERRILL
    energy barrier calculations

    returns "r" states (those that can be achieved via rotation) and "m"
    states, those that can be achieved via mirroring.

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe updated with clusters and subclusters

    reprs: list of list of ints
    reprs[i] returns a list of direction-based states in cluster_df
    for particles with domain structure i

    chosen_reprs: list of ints:
    set of reprs chosen as the "ideal" base state

    reprs_pos: list of list of ints
    same as reprs, but for only states with positive directions

    reprs_neg: list of list of ints
    same as reprs, but for only states with negative directions.

    mean_dirs_pos: list of list of length 3 arrays
    same as reprs_pos, but with the LEM net magnetization direction

    mean_dirs_neg: list of list of length 3 arrays
    same as reprs_neg, but with the LEM net magnetization direction

    min_energies_pos: list of list of floats
    same as reprs_pos, but with total energy of the LEM state

    min_energies_neg: list of list of floats
    same as reprs_neg, but with total energy of the LEM state

    min_energies: list of list of ints
    same as reprs, but with total energy of the LEM state

    isboths: list of list of bools
    says whether reprs[i] contains both positive and negative helicity states

    min_angle: float
    Angle (in degress) within which two states are considered "the same"

    Returns
    -------
    chosen_reprs_r1: list of ints
    Indices of "chosen" states at the start of barriers where one reaches
    another via rotation

    chosen_reprs_r2: list of ints
    Indices of "chosen" states at the end of barriers where one reaches another
    via rotation

    chosen_reprs_m1: list of ints
    Indices of "chosen" states at the start of barriers where one reaches
    another via mirroring

    chosen_reprs_m2: list of ints
    Indices of "chosen" states at the end of barriers where one reaches another
    via mirroring
    """
    chosen_reprs_r1 = []
    chosen_reprs_r2 = []
    chosen_reprs_m1 = []
    chosen_reprs_m2 = []
    for j in range(len(chosen_reprs)):
        print(j)
        group_idxs = get_cluster_group_idxs(cluster_df, j, min_angle=min_angle)
        group_r1s = []
        group_r2s = []
        for group in group_idxs:
            scores = []
            combos = []
            signs_new = []
            try:
                gfilter = ~np.any(np.isnan(group), axis=1)
                gfilter = (gfilter) & (group[:, 0] != group[:, 1])
                group = group[gfilter]
                for new_idxs in group:
                    new_idxs = new_idxs.astype(int)
                    combos.append(new_idxs)
                    combos.append(new_idxs)
                    # print(len(mean_dirs_pos[j]),new_idxs[0],new_idxs[1])
                    score = np.dot(
                        mean_dirs_pos[j][new_idxs[0]], mean_dirs_pos[j][new_idxs[1]]
                    ) / np.exp(
                        (
                            min_energies_pos[j][new_idxs[0]]
                            + min_energies_pos[j][new_idxs[1]]
                            - 2 * np.nanmin(min_energies[j])
                        )
                        / (1.380649e-23 * 293.15)
                    )
                    scores.append(score)
                    score = np.dot(
                        mean_dirs_neg[j][new_idxs[0]], mean_dirs_neg[j][new_idxs[1]]
                    ) / np.exp(
                        (
                            min_energies_neg[j][new_idxs[0]]
                            + min_energies_neg[j][new_idxs[1]]
                            - 2 * np.nanmin(min_energies[j])
                        )
                        / (1.380649e-23 * 293.15)
                    )
                    scores.append(score)
                    signs_new.append(1)
                    signs_new.append(-1)
                combos = np.array(combos)
                scores = np.array(scores)
                signs_new = np.array(signs_new)
                rot_idx = np.array(list(range(len(combos))))[
                    scores == np.nanmax(scores)
                ][0]

                if rot_idx % 2 == 0:
                    repr_sign = reprs_pos
                else:
                    repr_sign = reprs_neg
                rot = combos[rot_idx]
                if repr_sign[j][rot[1]] == chosen_reprs[j]:
                    rot = np.flip(rot)
                group_r1s.append(repr_sign[j][rot[0]])
                group_r2s.append(repr_sign[j][rot[1]])
            except:
                print(j)
                group_r1s.append(np.nan)
                group_r2s.append(np.nan)
        chosen_reprs_r1.append(group_r1s)
        chosen_reprs_r2.append(group_r2s)
        # print(chosen_reprs_r1)
        # print(chosen_reprs_r1)
        try:
            min_both_energy = np.array(min_energies[j])[
                np.array(isboths[j])
            ] == np.amin(np.array(min_energies[j])[np.array(isboths[j])])
            unmirror = np.array(reprs[j])[np.array(isboths[j])][min_both_energy][0]
            cluster, subcluster = cluster_df.loc[
                unmirror, ["cluster", "subcluster"]
            ].values
            sign = np.sign(cluster_df.loc[chosen_reprs[j], "Relative helicity"])
            unwind = cluster_df[
                (cluster_df.cluster == cluster)
                & (cluster_df.subcluster == subcluster)
                & (np.sign(cluster_df["Relative helicity"]) == -sign)
            ]
            wind = cluster_df[
                (cluster_df.cluster == cluster)
                & (cluster_df.subcluster == subcluster)
                & (np.sign(cluster_df["Relative helicity"]) == sign)
            ]
            unwind = unwind[unwind["Total energy"] == np.amin(unwind["Total energy"])]
            wind = wind[wind["Total energy"] == np.amin(wind["Total energy"])]
            mirror = wind.index[0]
            unmirror = unwind.index[0]
            chosen_reprs_m1.append(mirror)
            chosen_reprs_m2.append(unmirror)
        except:
            chosen_reprs_m1.append(np.nan)
            chosen_reprs_m2.append(np.nan)

    return (chosen_reprs_r1, chosen_reprs_r2, chosen_reprs_m1, chosen_reprs_m2)


def make_keys_and_pairs(
    cluster_df,
    chosen_reprs,
    chosen_reprs_r1,
    chosen_reprs_r2,
    chosen_reprs_m1,
    chosen_reprs_m2,
):
    """
    Produces output files that can be used with cluster_df to choose pairs of
    states to calculate energy values between.

    Parameters
    ----------
    cluster_df: pandas dataframe
    Dataframe updated with clusters and subclusters

    chosen_reprs: list of ints:
    set of reprs chosen as the "ideal" base state

    chosen_reprs_r1: list of ints
    Indices of "chosen" states at the start of barriers where one reaches
    another via rotation

    chosen_reprs_r2: list of ints
    Indices of "chosen" states at the end of barriers where one reaches another
    via rotation

    chosen_reprs_m1: list of ints
    Indices of "chosen" states at the start of barriers where one reaches
    another via mirroring

    chosen_reprs_m2: list of ints
    Indices of "chosen" states at the end of barriers where one reaches another
    via mirroring
    """

    chosen_reprs = [
        cluster_df.loc[chosen_repr, "Unique ID"] for chosen_repr in chosen_reprs
    ]

    chosen_reprs_r1 = [
        [
            (
                cluster_df.loc[chosen_repr_r1, "Unique ID"]
                if type(chosen_repr_r1) != float
                else "-"
            )
            for chosen_repr_r1 in gp_r1
        ]
        for gp_r1 in chosen_reprs_r1
    ]
    chosen_reprs_r2 = [
        [
            (
                cluster_df.loc[chosen_repr_r2, "Unique ID"]
                if type(chosen_repr_r2) != float
                else "-"
            )
            for chosen_repr_r2 in gp_r2
        ]
        for gp_r2 in chosen_reprs_r2
    ]
    chosen_reprs_m1 = [
        (
            cluster_df.loc[chosen_repr_m1, "Unique ID"]
            if type(chosen_repr_m1) != float
            else "-"
        )
        for chosen_repr_m1 in chosen_reprs_m1
    ]
    chosen_reprs_m2 = [
        (
            cluster_df.loc[chosen_repr_m2, "Unique ID"]
            if type(chosen_repr_m2) != float
            else "-"
        )
        for chosen_repr_m2 in chosen_reprs_m2
    ]
    repr_pairs = []
    repr_key = {}
    for new_line in combinations(chosen_reprs, 2):
        repr_pairs.append(new_line)

    for i in range(len(chosen_reprs)):
        if chosen_reprs_m1[i] != "-":
            new_line = [chosen_reprs_m1[i], chosen_reprs_m2[i]]
            repr_pairs.append(new_line)

        if chosen_reprs_m1[i] != "-":
            repr_key[str(chosen_reprs_m1[i])] = "s" + str(i) + "m1"

        for k in range(len(chosen_reprs_r1[i])):
            new_line = [chosen_reprs_r1[i][k], chosen_reprs_r2[i][k]]
            repr_pairs.append(new_line)
            repr_key[str(chosen_reprs_r1[i][k])] = "s" + str(i) + f"r{2 * k + 1}"

        if chosen_reprs_m2[i] != "-":
            repr_key[str(chosen_reprs_m2[i])] = "s" + str(i) + "m2"

        for k in range(len(chosen_reprs_r1[i])):
            repr_key[str(chosen_reprs_r2[i][k])] = "s" + str(i) + f"r{2 * k + 2}"

        repr_key[str(chosen_reprs[i])] = "s" + str(i)
    return (repr_pairs, repr_key)

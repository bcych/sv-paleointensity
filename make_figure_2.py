import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import pandas as pd

# Make TMx,Tb data
TMxs = np.arange(0, 0.65, 0.05)
Tcs = (3.7237e02 * TMxs**3 - 6.9152e02 * TMxs**2 - 4.1385e02 * TMxs**1 + 5.8000e02,)

dataframe = pd.read_csv("data/Tb_tr.csv", sep="\s+")

spheres = dataframe[dataframe.Prolateness == 1]
relax_times = []
Tbs_array = []
for TMx in TMxs:
    cslice = spheres[spheres.Tmx == int(TMx * 100)]
    relax_times.append(cslice.tr.values)
    Tbs_array.append(cslice.Tb.values)
relax_times = np.array(relax_times)
Tbs_array = np.array(Tbs_array)

bounds = np.array([1, 86400, 3.154e7, 3.154e10, 3.154e13, 1.7e16, 1.438e17])
norm = colors.BoundaryNorm(np.log10(bounds), ncolors=256, extend="both")


plt.pcolor(
    np.arange(0, 65, 5),
    np.arange(40, 250, 10),
    np.log10(relax_times.T),
    cmap="plasma",
    norm=norm,
)
cbar = plt.colorbar(extend="both", label="Relaxation time (s)", spacing="proportional")
cbar.set_ticklabels(
    ["1s", "1 day", "1yr", "1kyr", "1 Myr", "Phanerozoic", "Age of Earth"]
)
plt.xlabel("TMx (%)")
plt.ylabel("Size (nm)")
plt.title("a)", loc="left")

plt.show()


plt.figure()


plt.pcolor(
    np.arange(0, 65, 5), np.arange(40, 250, 10), np.array(Tbs_array).T, cmap="inferno"
)
cbar = plt.colorbar(extend="min", label="Blocking Temperature $^\circ$C")

plt.xlabel("TMx (%)")
plt.ylabel("Size (nm)")
plt.title("b)", loc="left")
plt.yticks(np.arange(40, 260, 20))
plt.show()

plt.figure()
plt.pcolor(
    np.arange(0, 65, 5),
    np.arange(40, 250, 10),
    ((np.array(Tbs_array) - 20) / (np.array(Tcs) - 20).T).T,
    cmap="inferno",
    vmin=0,
    vmax=1,
)
cbar = plt.colorbar(extend="min", label="Relative Blocking Temperature")
cbar.ax.set_yticklabels(["$T_\mathrm{room}$", 0.2, 0.4, 0.6, 0.8, "$T_\mathrm{c}$"])
plt.xlabel("TMx (%)")
plt.ylabel("Size (nm)")
plt.title("c)", loc="left")
plt.yticks(np.arange(40, 260, 20))
plt.show()

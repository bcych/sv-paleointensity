import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmocean.cm import haline

dataframe = pd.read_csv("data/Tb_tr.csv", sep="\s+")
elongate = dataframe[dataframe.Prolateness == 1.44]
markers = ["s", "p", "x", "h", "+"]
labels = ["0%", "15%", "30%", "45%", "60%"]
i = 0
for composition in [0, 15, 30, 45, 60]:
    cslice = elongate[elongate.Tmx == composition]
    plt.plot(
        cslice.Size,
        cslice.Tb,
        marker=markers[i],
        color=haline(0.9 * (i / 4)),
        label=labels[i],
    )
    i += 1

plt.xlim(25, 250)
plt.ylim(0, 600)

plt.xlabel("Size (nm)")
plt.ylabel("Blocking Temperature (°C)")
plt.xlabel("Size (nm)")
plt.legend(title="TMx (%)", fontsize=8)
plt.show()

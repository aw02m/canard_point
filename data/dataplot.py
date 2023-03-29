import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from matplotlib.patches import ArrowStyle

# データ読み取り用
def setdata(filename, xcol, ycol):
    x_list = []
    y_list = []
    fd = open(filename, "rt")  # specify appropriate data file here
    for line in fd:
        data = line[:-1].split(" ")
        x_list.append(float(data[xcol]))
        y_list.append(float(data[ycol]))
    return x_list, y_list


# 版組パラメタ
params = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    "legend.fontsize": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "serif",
    "grid.color": "k",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
}
plt.rcParams.update(params)

# 図の大きさと比
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
# ax.set_aspect('equal', 'datalim')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# 軸回り
x_limit = [-0.01, 0.001]
y_limit = [0, 500]
x_tick = 0.005
y_tick = 100
ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
plt.xticks(np.arange(x_limit[0], x_limit[1] + x_tick, x_tick), fontsize=16)
plt.yticks(np.arange(y_limit[0], y_limit[1] + y_tick, y_tick), fontsize=16)
ax.set_xlabel(r"$\lambda \longrightarrow$", fontsize=16)
ax.set_ylabel(r"$\frac{1}{\epsilon} \longrightarrow $", fontsize=16)

# 格子
ax.grid(c="gainsboro", zorder=2)

# データ
(x_list, y_list) = setdata("bifdata", 1, 0)
y_list = [1 / i for i in y_list]
ax.plot(x_list, y_list, color="BLACK", linewidth=2)
(x_list, y_list) = setdata("canard_bif_data", 1, 0)
y_list = [1 / i for i in y_list]
ax.plot(x_list, y_list, color="BLACK", linewidth=2)
(x_list, y_list) = setdata("canard_bif_data", 2, 0)
y_list = [1 / i for i in y_list]
ax.plot(x_list, y_list, color="darkgray", linewidth=2)
(x_list, y_list) = setdata("canard_bif_data", 3, 0)
y_list = [1 / i for i in y_list]
ax.plot(x_list, y_list, color="darkgray", linewidth=2)

# 出力
fig.subplots_adjust(left=0.175, right=0.94, bottom=0.135, top=0.975)
pdf = PdfPages("snapshot.pdf")
pdf.savefig(dpi=300)
pdf.close()

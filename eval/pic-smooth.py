import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# raw data
x = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
y = np.array([0.0171, 0.9654, 0.9705, 0.9741, 0.9959])

# monotone smooth interpolation
x_smooth = np.linspace(x.min(), x.max(), 300)
interp = PchipInterpolator(x, y)
y_smooth = interp(x_smooth)

save_dir = "outputs/eval"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(4.8, 3.8))

# smooth curve
plt.plot(x_smooth, y_smooth, linewidth=2)

# raw points
plt.scatter(x, y, s=45, zorder=3)

plt.xlabel(r"Control value $s$")
plt.ylabel("Average post probability")
plt.xticks(x)
plt.ylim(0, 1.05)
plt.xlim(-0.02, 1.02)
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()

png_path = os.path.join(save_dir, "fig_5_2_post_prob_curve_smooth.png")
pdf_path = os.path.join(save_dir, "fig_5_2_post_prob_curve_smooth.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Saved to:")
print(png_path)
print(pdf_path)
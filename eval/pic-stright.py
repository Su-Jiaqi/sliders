import os
import matplotlib.pyplot as plt

# data
s_vals = [0.00, 0.25, 0.50, 0.75, 1.00]
post_probs = [0.0171, 0.9654, 0.9705, 0.9741, 0.9959]

save_dir = "/home/sjq/concept_sliders/outputs/eval"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(4.8, 3.8))
plt.plot(s_vals, post_probs, marker='o', linewidth=2)

plt.xlabel(r"Control value $s$")
plt.ylabel("Average post probability")
plt.ylim(0, 1.05)
plt.xticks(s_vals)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

png_path = os.path.join(save_dir, "fig_5_2_post_prob_curve.png")
pdf_path = os.path.join(save_dir, "fig_5_2_post_prob_curve.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print("Saved to:")
print(png_path)
print(pdf_path)
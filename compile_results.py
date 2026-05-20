import json
import csv
import os

results = [
    # variant, layers, sine_layer, training_psnr, training_epoch
    ("ablation_depth_4",       4,  "L2",   30.88, 40),
    ("ablation_depth_6",       6,  "L3",   31.06, 70),
    ("ablation_depth_8_verify",8,  "L4",   30.99, 50),
    ("ablation_depth_10",      10, "L5",   31.06, 70),
    ("ablation_depth_12",      12, "L6",   30.86, 40),
    ("ablation_sine_none",     8,  "none", 30.89, 40),
    ("ablation_sine_L2",       8,  "L2",   30.89, 40),
    ("ablation_sine_L6",       8,  "L6",   30.91, 80),
    ("ablation_sine_all",      8,  "all",  30.97, 80),
]

rows = []
for variant, layers, sine, train_psnr, train_epoch in results:
    path = f"results/{variant}/results.json"
    with open(path) as f:
        r = json.load(f)
    rows.append({
        "variant":        variant,
        "mlp_layers":     layers,
        "sine_layer":     sine,
        "train_psnr_db":  train_psnr,
        "best_epoch":     train_epoch,
        "psnr_mean":      round(r["psnr_mean"], 4),
        "psnr_std":       round(r["psnr_std"],  4),
        "ssim_mean":      round(r["ssim_mean"], 4),
        "ssim_std":       round(r["ssim_std"],  4),
        "rmse_mean":      round(r["rmse_mean"], 4),
        "rmse_std":       round(r["rmse_std"],  4),
    })

# Add baseline
with open("results/baseline_r3/results.json") as f:
    b = json.load(f)
rows.append({
    "variant":        "baseline_200ep",
    "mlp_layers":     8,
    "sine_layer":     "L4",
    "train_psnr_db":  None,
    "best_epoch":     200,
    "psnr_mean":      round(b["psnr_mean"], 4),
    "psnr_std":       round(b["psnr_std"],  4),
    "ssim_mean":      round(b["ssim_mean"], 4),
    "ssim_std":       round(b["ssim_std"],  4),
    "rmse_mean":      round(b["rmse_mean"], 4),
    "rmse_std":       round(b["rmse_std"],  4),
})

out = "results/ablation_results_all.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out}")
for r in rows:
    print(f"{r['variant']:30s} | train={r['train_psnr_db']} | "
          f"PSNR={r['psnr_mean']}±{r['psnr_std']} | "
          f"SSIM={r['ssim_mean']}±{r['ssim_std']} | "
          f"RMSE={r['rmse_mean']}±{r['rmse_std']}")

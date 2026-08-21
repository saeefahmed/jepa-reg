# Stage 4 control: lambda=1.0 WITHOUT lambda warmup, PAWS, 3 seeds.
# Distinguishes "JEPA-Reg is insensitive to lambda" from "lambda warmup removes lambda sensitivity".
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("_setup_extracted.py", encoding="utf-8").read())

LAMBDA_WARMUP = False   # the control condition
print(f"LAMBDA_WARMUP = {LAMBDA_WARMUP}", flush=True)

bb = "bert-base-uncased"; cfg = BACKBONE_CONFIGS[bb]
tok, col, ds, ldr = make_loaders(bb)
f1s = []
for seed in SEEDS:
    r = run_experiment(ldr["paws_train"], ldr["paws_val"], ldr["paws_test"],
                       "jepa", bb, seed, lr=cfg["lr_paws"], aux_lambda=1.0,
                       epochs=cfg["epochs_paws"], grad_accum=cfg["grad_accum"],
                       tag=f"PAWS/lam1.0-nowarmup/s{seed}")
    f1s.append(r["test"]["f1"])
    print(f"CONTROL seed {seed}: F1 = {r['test']['f1']:.4f}", flush=True)

print(f"CONTROL RESULT lam=1.0 no-warmup: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}", flush=True)
with open("results/lambda1_nowarmup_control.json", "w") as f:
    json.dump({"f1s": f1s, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))}, f, indent=2)
print("STAGE 4 COMPLETE", flush=True)

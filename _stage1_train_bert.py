# Stage 1: full BERT-base run (equivalent of notebook Cell 21).
# MRPC + low-resource sweep + QQP + PAWS, 3 seeds. Crash-safe via per-dataset checkpoints.
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("_setup_extracted.py", encoding="utf-8").read())

all_results = {}
ckpt_bert = "results/checkpoint_bert.json"
resume = {}
partial = "results/checkpoint_bert-base-uncased_partial.json"
if os.path.exists(ckpt_bert):
    with open(ckpt_bert) as f:
        all_results.update(json.load(f))
    print("BERT checkpoint found - already complete")
else:
    if os.path.exists(partial):
        with open(partial) as f:
            resume = json.load(f)
        print("Resuming from partial checkpoint:", [k for k in resume])
    all_results["bert-base-uncased"] = run_backbone("bert-base-uncased", resume=resume)
    os.makedirs("results", exist_ok=True)
    with open(ckpt_bert, "w") as f:
        json.dump({"bert-base-uncased": all_results["bert-base-uncased"]}, f, indent=2, default=float)
    print("BERT checkpoint saved ->", ckpt_bert)

print("STAGE 1 COMPLETE")

# Stage 3: R-5 (positive-only vs all-pairs mask ablation + collapse check)
#          R-7 (full lambda ablation with std, incl. lambda=1.0) — BERT, MRPC+PAWS
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("_setup_extracted.py", encoding="utf-8").read())

with open("results/checkpoint_bert.json") as f:
    all_results = json.load(f)

nb = json.load(open("JEPA_final_optionB_fixed.ipynb", encoding="utf-8"))
def cell_src(i): return "".join(nb["cells"][i]["source"])

print("========== R-5 mask ablation ==========", flush=True)
exec(cell_src(49))   # R-5

print("========== R-7 lambda ablation ==========", flush=True)
exec(cell_src(51))   # R-7

print("STAGE 3 COMPLETE", flush=True)

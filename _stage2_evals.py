# Stage 2: evaluations + reviewer cells. Auto-generated.
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("_setup_extracted.py", encoding="utf-8").read())

# ---- restore all_results: fresh BERT (raw) + March RoBERTa (adapted) ----
with open("results/checkpoint_bert.json") as f:
    all_results = json.load(f)
_packed = json.load(open("all_results_final.json"))
_rb = {}
for _ds in ("mrpc","qqp","paws"):
    _rb[_ds] = {mt: [{"test": {"f1": v, "accuracy": 0.0}} for v in d["f1s"]]
                for mt, d in _packed["roberta-base"][_ds].items()
                if isinstance(d, dict) and "f1s" in d}
_rb["best_lam_mrpc"] = _rb["best_lam_qqp"] = _rb["best_lam_paws"] = 0.1
all_results["roberta-base"] = _rb
print("all_results restored:", list(all_results))

# ---- restore BEST_MODELS from disk (cell 25 logic) ----


# ═══════════════════════════════════════════════════════════
# §9b. Restore BEST_MODELS from disk (safe after kernel restart)
# ═══════════════════════════════════════════════════════════
import os as _os

_model_dir = 'results/models'
_loaded = []
if _os.path.isdir(_model_dir):
    for _fname in sorted(_os.listdir(_model_dir)):
        if _fname.endswith('.pt'):
            _key = _fname[:-3]
            if _key not in BEST_MODELS:
                try:
                    BEST_MODELS[_key] = torch.load(
                        f'{_model_dir}/{_fname}', map_location='cpu',
                        weights_only=True)
                    _loaded.append(_key)
                except Exception as _e:
                    print(f'  [WARN] could not load {_fname}: {_e}')

if _loaded:
    print(f'✅ Restored {len(_loaded)} models into BEST_MODELS from disk:')
    for _k in _loaded: print(f'   {_k}')
else:
    print(f'BEST_MODELS already has {len(BEST_MODELS)} entries — nothing to restore')
    if not BEST_MODELS:
        print('  ⚠ WARNING: No saved models found in results/models/')
        print('  Re-run Cells 21+23 to retrain, or place .pt files in results/models/')


class HANSDataset(torch.utils.data.Dataset):  # explicit — avoids HF Dataset inheritance
    def __init__(self, hf_ds):
        self.hf_ds = list(hf_ds)  # plain list — avoids HF batch-index bug
    def __len__(self): return len(self.hf_ds)
    def __getitem__(self, idx):
        ex = self.hf_ds[idx]
        # entailment(0) -> 1 (paraphrase), non-entailment(1) -> 0
        label = 1 if ex['label'] == 0 else 0
        return {'s1': ex['premise'], 's2': ex['hypothesis'],
                'labels': label, 'heuristic': ex['heuristic']}

def hans_collate(batch, tokenizer, max_len=MAX_LEN):
    s1 = [b['s1'] for b in batch]
    s2 = [b['s2'] for b in batch]
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
    heuristics = [b['heuristic'] for b in batch]
    kw = dict(padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    pair = tokenizer(s1, s2, **kw)
    e1   = tokenizer(s1, **kw)
    e2   = tokenizer(s2, **kw)
    return {'pair_input_ids':      pair['input_ids'],
            'pair_attention_mask': pair['attention_mask'],
            's1_input_ids':        e1['input_ids'],
            's1_attention_mask':   e1['attention_mask'],
            's2_input_ids':        e2['input_ids'],
            's2_attention_mask':   e2['attention_mask'],
            'labels': labels, 'heuristics': heuristics}

@torch.no_grad()
def evaluate_hans(bert_name, model_type, state_dict):
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
    collate_fn = lambda b: hans_collate(b, tok)
    ldr = DataLoader(HANSDataset(hans_raw[HANS_SPLIT]), 64,
                     shuffle=False, collate_fn=collate_fn, num_workers=0)
    if model_type == 'baseline':
        m = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2)
        m.load_state_dict(state_dict)
    elif model_type == 'jepa':
        m = JepaBertPair(bert_name)
        m.load_state_dict(state_dict, strict=False)
    elif model_type == 'simcse':
        m = SimCSEBertPair(bert_name)
        m.load_state_dict(state_dict)
    m = m.to(device); m.eval()
    all_preds=[]; all_labels=[]; all_heur=[]
    for batch in ldr:
        heur = batch.pop('heuristics')
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast(enabled=USE_AMP):
            if model_type == 'baseline':
                logits = m(input_ids=batch['pair_input_ids'],
                           attention_mask=batch['pair_attention_mask']).logits
            else:
                fwd = {k: batch[k] for k in JEPA_KEYS if k in batch}
                if model_type == 'jepa': fwd['jepa_lambda'] = 0.0
                logits = m(**fwd)['logits']
        all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
        all_labels.extend(batch['labels'].cpu().tolist())
        all_heur.extend(heur)
    del m; torch.cuda.empty_cache()
    overall_f1  = f1_score(all_labels, all_preds, zero_division=0)
    overall_acc = accuracy_score(all_labels, all_preds)
    heur_names  = sorted(set(all_heur))
    per_heur = {}
    for h in heur_names:
        idx = [i for i, hh in enumerate(all_heur) if hh == h]
        per_heur[h] = {
            'f1':  f1_score( [all_labels[i] for i in idx],
                             [all_preds[i]  for i in idx], zero_division=0),
            'acc': accuracy_score([all_labels[i] for i in idx],
                                  [all_preds[i]  for i in idx]),
            'n':   len(idx)
        }
    return {'overall_f1': overall_f1, 'overall_acc': overall_acc,
            'per_heuristic': per_heur}

# Run HANS evaluation
hans_results = {}
BNAME = {'bert-base-uncased': 'BERT-base', 'roberta-base': 'RoBERTa-base'}

if hans_raw is None:
    print('HANS dataset not available -- skipping')
else:
    print('Running HANS zero-shot evaluation (MRPC-trained models)...')
    _missing_keys = []
    for bb in BACKBONES:
        if bb not in all_results: continue
        bname = bb.split('/')[-1]
        hans_results[bname] = {}
        for mt in ('baseline', 'simcse', 'jepa'):
            key = f'{bname}_mrpc_{mt}'
            if key not in BEST_MODELS:
                _missing_keys.append(key); continue
            res = evaluate_hans(bb, mt, BEST_MODELS[key])
            hans_results[bname][mt] = res
            print(f'  {BNAME[bb]:<20} {mt:<12}'
                  f' f1={res["overall_f1"]:.4f}  acc={res["overall_acc"]:.4f}')
        if hans_results[bname]:
            print()
    if _missing_keys:
        print(f'\n⚠ Skipped {len(_missing_keys)} models not in BEST_MODELS:')
        for k in _missing_keys: print(f'    {k}')
        print('  → Re-run training cells so .pt files are saved, then run Cell 25 to restore')
    print(f'\nHANS done. Backbones evaluated: {[b for b in hans_results if hans_results[b]]}')
    bname0 = 'bert-base-uncased'
    if hans_results.get(bname0):
        hns = sorted(set().union(
            *[set(v['per_heuristic']) for v in hans_results[bname0].values()]))
        header = f'  {"Heuristic":<25}'
        for mt in ('baseline','simcse','jepa'): header += f'  {mt:>12}'
        print(header); print('  ' + '-' * 65)
        for h in hns:
            row = f'  {h:<25}'
            for mt in ('baseline','simcse','jepa'):
                acc = (hans_results[bname0].get(mt,{})
                       .get('per_heuristic',{}).get(h,{}).get('acc',0))
                row += f'  {acc:>12.4f}'
            print(row)
    print('\nHANS done. Key: lexical_overlap -- JEPA should outperform baseline here')


import re
from collections import Counter

def jaccard_overlap(s1, s2):
    """Word-level Jaccard overlap between two sentences."""
    def tok(s): return set(re.findall(r'\w+', s.lower()))
    a, b = tok(s1), tok(s2)
    if not a and not b: return 0.0
    return len(a & b) / len(a | b)

def word_order_divergence(s1, s2):
    """Fraction of shared words that appear in different order.
    High divergence + high overlap = PAWS-style adversarial pair."""
    def tok(s): return re.findall(r'\w+', s.lower())
    t1, t2 = tok(s1), tok(s2)
    shared = set(t1) & set(t2)
    if not shared: return 0.0
    pos1 = {w: i for i, w in enumerate(t1) if w in shared}
    pos2 = {w: i for i, w in enumerate(t2) if w in shared}
    mismatches = sum(1 for w in shared if abs(pos1[w] - pos2[w]) > 1)
    return mismatches / len(shared)

@torch.no_grad()
def get_predictions_with_overlap(bert_name, model_type, state_dict, dataset_name="paws"):
    """Run model on test set, returning predictions + overlap scores per example."""
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
    col = make_collator(tok)

    # Get raw examples for overlap computation
    if dataset_name == "paws":
        raw_te = paws_te
        s1_key, s2_key, lbl_key = "sentence1", "sentence2", "label"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name} (only 'paws' supported)")

    ds_obj = SimpleDataset(raw_te, s1_key, s2_key, lbl_key)
    ldr = DataLoader(ds_obj, 64, shuffle=False, collate_fn=col, num_workers=0)

    if model_type == "baseline":
        m = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2)
        m.load_state_dict(state_dict)
    elif model_type == "jepa":
        m = JepaBertPair(bert_name); m.load_state_dict(state_dict, strict=False)
    elif model_type == "simcse":
        m = SimCSEBertPair(bert_name); m.load_state_dict(state_dict)
    m = m.to(device); m.eval()

    preds_all=[]; labels_all=[]
    for batch in ldr:
        batch = _move(batch)
        with autocast(enabled=USE_AMP):
            if model_type=="baseline":
                out=m(input_ids=batch["pair_input_ids"],attention_mask=batch["pair_attention_mask"])
                logits=out.logits
            else:
                fwd={k:batch[k] for k in JEPA_KEYS if k in batch}
                if model_type=="jepa": fwd["jepa_lambda"]=0.0
                out=m(**fwd); logits=out["logits"]
        preds_all.extend(torch.argmax(logits,1).cpu().tolist())
        labels_all.extend(batch["labels"].cpu().tolist())
    del m; torch.cuda.empty_cache()

    # Compute per-example overlap
    overlaps=[]; orders=[]
    for ex in raw_te:
        overlaps.append(jaccard_overlap(ex[s1_key], ex[s2_key]))
        orders.append(word_order_divergence(ex[s1_key], ex[s2_key]))

    return {"preds": preds_all, "labels": labels_all,
            "overlaps": overlaps[:len(preds_all)],
            "order_div": orders[:len(preds_all)]}

# ── Compute overlap-binned accuracy for PAWS ────────────────────
# Use whichever backbone has PAWS models saved — prefer BERT, fall back to RoBERTa
_pref_order = ["bert-base-uncased", "roberta-base"]
bb0 = next((bb for bb in _pref_order
             if any(f'{bb.split("/")[-1]}_paws_{mt}' in BEST_MODELS
                    for mt in ("baseline","simcse","jepa"))), None)

overlap_data = {}
if bb0 is None:
    print("⚠ No PAWS models found in BEST_MODELS — skipping overlap analysis")
    print("  Run training cells first, then Cell 25 to restore models from disk")
else:
    bname0 = bb0.split("/")[-1]
    print(f"Computing lexical overlap analysis on PAWS test set ({bname0})...")
    for mt in ("baseline", "simcse", "jepa"):
        key = f"{bname0}_paws_{mt}"
        if key in BEST_MODELS:
            overlap_data[mt] = get_predictions_with_overlap(bb0, mt, BEST_MODELS[key], "paws")
            print(f"  {mt}: done ({len(overlap_data[mt]['preds'])} examples)")
        else:
            print(f"  ⚠ {mt}: no saved model for {bname0}")

# Bin by overlap quartile and compute accuracy per bin
print("\n── Accuracy by Overlap Bin (PAWS test) ──")
print("(High overlap + label=0 → hard cases — JEPA should maintain accuracy here)")
bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
bin_labels = ["0–25%", "25–50%", "50–75%", "75–100%"]
print(f"{'Overlap bin':<14}", end="")
for mt in ("baseline","simcse","jepa"): print(f"  {mt:>12}", end="")
print("    N")
print("-"*60)
for (lo,hi), bl in zip(bins, bin_labels):
    print(f"  {bl:<12}", end="")
    ns=[]
    for mt in ("baseline","simcse","jepa"):
        if mt not in overlap_data: print(f"  {'N/A':>12}", end=""); continue
        d=overlap_data[mt]
        idx=[i for i,ov in enumerate(d["overlaps"]) if lo<=ov<hi]
        if not idx: print(f"  {'—':>12}", end=""); ns.append(0); continue
        p=[d["preds"][i] for i in idx]; l=[d["labels"][i] for i in idx]
        acc=accuracy_score(l,p)
        print(f"  {acc:>12.4f}", end="")
        ns.append(len(idx))
    print(f"    {max(ns) if ns else 0}")

print("\n✅ Overlap analysis complete")
print("Paper narrative: at 75-100% overlap, JEPA should show smallest accuracy drop")


# ═══ R-1 (Reviewer 4, item 1): PAWS configuration + validity diagnostics ═══
# Documents the PAWS subset, class balance, majority-class baseline, and per-model
# confusion matrix + per-class precision/recall on the PAWS test set.
# NOTE FOR PAPER: PAWS = "labeled_final". Models ARE fine-tuned on 20k PAWS train
# pairs (see run_backbone) — the paper text must stop calling PAWS "zero-shot".
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

_labels_te = [int(x["label"]) for x in paws_te]
_n = len(_labels_te); _pos = sum(_labels_te)
print(f"PAWS subset: labeled_final | test n={_n} pos={_pos} ({_pos/_n:.1%}) neg={_n-_pos} ({1-_pos/_n:.1%})")
_maj_acc = max(_pos, _n-_pos)/_n
print(f"Majority-class baseline (predict non-paraphrase): acc={_maj_acc:.4f}  F1(pos)=0.0000")
_p = _pos/_n
print(f"Always-paraphrase baseline: acc={_p:.4f}  F1(pos)={2*_p/(1+_p):.4f}")

for _bb in BACKBONES:
    _bn = _bb.split("/")[-1]
    for _mt in ("baseline","simcse","jepa"):
        _key = f"{_bn}_paws_{_mt}"
        if _key not in BEST_MODELS:
            print(f"  [skip] {_key} not in BEST_MODELS"); continue
        _d = get_predictions_with_overlap(_bb, _mt, BEST_MODELS[_key], "paws")
        _cm = confusion_matrix(_d["labels"], _d["preds"])
        _pr, _rc, _f1, _sup = precision_recall_fscore_support(_d["labels"], _d["preds"], zero_division=0)
        print(f"\n{_bn} / {_mt}  (seed {SEEDS[0]} best-val model)")
        print(f"  confusion matrix [rows=gold 0,1; cols=pred 0,1]:\n{_cm}")
        for _c in (0,1):
            print(f"  class {_c}: precision={_pr[_c]:.4f} recall={_rc[_c]:.4f} f1={_f1[_c]:.4f} support={_sup[_c]}")
print("\n✅ R-1 done — put subset name, class balance, majority row, and confusion matrices in the paper (Sec 4.1 + appendix)")


# ═══ R-2 (Reviewer 4, item 2): HANS accuracy split by entailed / non-entailed subcases ═══
# McCoy et al. (2019) style reporting. The +9.16 constituent F1 claim stands only if
# the gain survives the per-label split (accuracy on gold=entailment AND gold=non-entailment).
@torch.no_grad()
def evaluate_hans_subcase(bert_name, model_type, state_dict):
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
    ex_list = list(hans_raw[HANS_SPLIT])
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(ex_list)
        def __getitem__(self, i):
            ex = ex_list[i]
            return {"s1": ex["premise"], "s2": ex["hypothesis"],
                    "labels": 1 if ex["label"] == 0 else 0,   # entailment -> paraphrase(1)
                    "heuristic": ex["heuristic"], "subcase": ex["subcase"]}
    def _col(batch):
        out = hans_collate([{k: b[k] for k in ("s1","s2","labels","heuristic")} for b in batch], tok)
        out["subcases"] = [b["subcase"] for b in batch]
        return out
    ldr = DataLoader(_DS(), 64, shuffle=False, collate_fn=_col, num_workers=0)
    if model_type == "baseline":
        m = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2); m.load_state_dict(state_dict)
    elif model_type == "jepa":
        m = JepaBertPair(bert_name); m.load_state_dict(state_dict, strict=False)
    else:
        m = SimCSEBertPair(bert_name); m.load_state_dict(state_dict)
    m = m.to(device); m.eval()
    P,L,H,S = [],[],[],[]
    for batch in ldr:
        heur = batch.pop("heuristics"); subs = batch.pop("subcases")
        batch = _move(batch)
        with autocast(enabled=USE_AMP):
            if model_type == "baseline":
                logits = m(input_ids=batch["pair_input_ids"], attention_mask=batch["pair_attention_mask"]).logits
            else:
                fwd = {k: batch[k] for k in JEPA_KEYS if k in batch}
                if model_type == "jepa": fwd["jepa_lambda"] = 0.0
                logits = m(**fwd)["logits"]
        P += torch.argmax(logits,1).cpu().tolist(); L += batch["labels"].cpu().tolist(); H += heur; S += subs
    del m; torch.cuda.empty_cache()
    res = {}
    for h in sorted(set(H)):
        for gold in (1,0):  # 1 = entailed, 0 = non-entailed
            idx = [i for i in range(len(P)) if H[i]==h and L[i]==gold]
            res[(h, "entailed" if gold==1 else "non-entailed")] = (
                sum(P[i]==L[i] for i in idx)/len(idx), len(idx))
    per_sub = {}
    for h,s in sorted(set(zip(H,S))):
        idx = [i for i in range(len(P)) if H[i]==h and S[i]==s]
        per_sub[(h,s)] = (sum(P[i]==L[i] for i in idx)/len(idx), len(idx))
    return res, per_sub

hans_subcase = {}
for _bb in BACKBONES:
    _bn = _bb.split("/")[-1]
    for _mt in ("baseline","simcse","jepa"):
        _key = f"{_bn}_mrpc_{_mt}"
        if _key not in BEST_MODELS: print(f"[skip] {_key}"); continue
        r, ps = evaluate_hans_subcase(_bb, _mt, BEST_MODELS[_key])
        hans_subcase[(_bn,_mt)] = {"per_label": {f"{k[0]}/{k[1]}": v for k,v in r.items()},
                                   "per_subcase": {f"{k[0]}/{k[1]}": v for k,v in ps.items()}}
        print(f"\n{_bn} / {_mt}")
        for k,(acc,n) in r.items():
            print(f"  {k[0]:<17} {k[1]:<13} acc={acc:.4f} (n={n})")
import json as _json
with open("results/hans_subcase.json","w") as _f:
    _json.dump({f"{a}|{b}": v for (a,b),v in hans_subcase.items()}, _f, indent=2, default=float)
print("\n✅ R-2 done → results/hans_subcase.json")
print("PAPER RULE: if JEPA-Reg's constituent gain comes only from higher entailed-accuracy")
print("while non-entailed accuracy drops (prediction-bias shift), SOFTEN mechanistic claim (1).")


# ═══ R-3 (Reviewer 4, item 3): Figure 5 fix — bin sizes, per-bin class balance, quantile bins ═══
# FINDING (dataset-level, no model needed): in PAWS labeled_final TEST, the fixed bins
# 0–25% and 25–50% Jaccard are EMPTY (n=0) and 50–75% has only n=84 (8.3% positive);
# 7916/8000 pairs fall in 75–100%. The old Figure 5 plotted empty bins as 0.00.
# Fix: use overlap QUARTILES of the empirical distribution and annotate n + class balance.
import numpy as _np
_ovs  = _np.array([jaccard_overlap(x["sentence1"], x["sentence2"]) for x in paws_te])
_lbls = _np.array([int(x["label"]) for x in paws_te])
_qs   = _np.quantile(_ovs, [0.25, 0.5, 0.75])
_edges = [_ovs.min()-1e-9] + list(_qs) + [_ovs.max()+1e-9]
print("PAWS test Jaccard overlap: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f"
      % (_ovs.min(), _qs[0], _qs[1], _qs[2], _ovs.max()))
print(f"\n{'quartile bin':<22}{'n':>6}{'%pos':>8}", end="")
for _mt in ("baseline","simcse","jepa"): print(f"{_mt+' acc':>14}", end="")
print(f"{'jepa acc|pos':>14}{'jepa acc|neg':>14}")
for _b in range(4):
    _in = (_ovs >= _edges[_b]) & (_ovs < _edges[_b+1])
    _row = f"[{_edges[_b]:.3f},{_edges[_b+1]:.3f})"
    print(f"{_row:<22}{int(_in.sum()):>6}{_lbls[_in].mean():>8.1%}", end="")
    for _mt in ("baseline","simcse","jepa"):
        if _mt in overlap_data:
            _d = overlap_data[_mt]
            _pr = _np.array(_d["preds"]); _gl = _np.array(_d["labels"])
            _acc = (_pr[_in]==_gl[_in]).mean() if _in.sum() else float("nan")
            print(f"{_acc:>14.4f}", end="")
        else:
            print(f"{'N/A':>14}", end="")
    if "jepa" in overlap_data:
        _pr = _np.array(overlap_data["jepa"]["preds"]); _gl = _np.array(overlap_data["jepa"]["labels"])
        for _cls in (1,0):
            _m2 = _in & (_gl==_cls)
            print(f"{(_pr[_m2]==_gl[_m2]).mean() if _m2.sum() else float('nan'):>14.4f}", end="")
    print()
print("\n✅ R-3 done — redraw Figure 5 from this table (per-class accuracy separates dose-response from class-prior shift)")


# ═══ R-4 (Reviewer 4 item 4; Reviewer 6): honest statistics at n=3 ═══
# Per-seed values, Cohen's d, paired t-test, and EXACT sign-flip permutation test.
# KEY FACT: with n=3 paired seeds the exact two-sided permutation p-value can never be
# below 0.25. Bootstrap "p<0.001" at n=3 is not supportable — remove it from the paper.
# Also: report "8.8x std reduction" once; do NOT additionally report 77x variance (it is 8.8^2).
from scipy.stats import ttest_rel as _ttest
import itertools as _it, numpy as _np

def exact_signflip_p(a, b):
    diff = _np.array(a) - _np.array(b); obs = abs(diff.mean()); cnt = tot = 0
    for signs in _it.product([1,-1], repeat=len(diff)):
        tot += 1
        if abs((diff*_np.array(signs)).mean()) >= obs - 1e-12: cnt += 1
    return cnt/tot

print(f"{'setting':<18}{'per-seed baseline':<28}{'per-seed JEPA-Reg':<28}{'d':>7}{'t-p':>8}{'exact-p':>9}")
for _bb in BACKBONES:
    if _bb not in all_results: continue
    _bn = _bb.split("/")[-1]
    for _ds,_mt in [("mrpc","jepa"),("paws","jepa"),("qqp","hybrid")]:
        _r = all_results[_bb].get(_ds,{})
        if not _r.get("baseline") or not _r.get(_mt): continue
        _b = [x["test"]["f1"] for x in _r["baseline"]]; _j = [x["test"]["f1"] for x in _r[_mt]]
        _d = cohen_d(_j,_b); _tp = paired_ttest_p(_j,_b); _ep = exact_signflip_p(_j,_b)
        print(f"{_bn[:7]+' '+_ds.upper():<18}"
              f"{str([round(v*100,2) for v in _b]):<28}{str([round(v*100,2) for v in _j]):<28}"
              f"{_d:>7.2f}{_tp:>8.3f}{_ep:>9.3f}")
print("\n✅ R-4 done — use per-seed values + d + t-test p in the paper; drop bootstrap p<0.001 and the 77x claim")


# ═══ R-8 (Reviewer 4, item 6; Reviewer 6): training/inference overhead measurement ═══
# Measures wall-clock per optimizer step, peak GPU memory, and parameter counts for
# baseline vs JEPA-Reg, plus inference cost (classification path only — identical arch).
import time as _time
_bb = "bert-base-uncased"; _cfg = BACKBONE_CONFIGS[_bb]
_tok,_col,_ds,_ldr = make_loaders(_bb)
_batches = []
for _i,_b in enumerate(_ldr["mrpc_train"]):
    _batches.append(_move(_b))
    if _i >= 29: break

def _bench_train(model_type, n_warm=5):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _m,_opt = build_model(model_type, _bb, aux_lambda=0.1, lr=2e-5)
    _sc = GradScaler(enabled=USE_AMP)
    _sch = get_linear_schedule_with_warmup(_opt, 5, 1000)
    _n_par = sum(p.numel() for p in _m.parameters())
    _n_trn = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    _times = []
    for _i,_b in enumerate(_batches):
        _t0 = _time.perf_counter()
        with autocast(enabled=USE_AMP):
            if model_type == "baseline":
                out = _m(input_ids=_b["pair_input_ids"], attention_mask=_b["pair_attention_mask"], labels=_b["labels"])
                loss = out.loss
            else:
                fwd = {k:_b[k] for k in JEPA_KEYS if k in _b}; fwd["jepa_lambda"]=0.1
                loss = _m(**fwd)["loss"]
        _sc.scale(loss).backward(); _sc.step(_opt); _sc.update(); _sch.step(); _opt.zero_grad(set_to_none=True)
        if model_type == "jepa": _m.update_ema()
        torch.cuda.synchronize()
        if _i >= n_warm: _times.append(_time.perf_counter()-_t0)
    _mem = torch.cuda.max_memory_allocated()/1e9
    del _m,_opt; torch.cuda.empty_cache()
    return {"ms_per_step": 1000*float(np.mean(_times)), "peak_gb": _mem,
            "params_total_M": _n_par/1e6, "params_trainable_M": _n_trn/1e6}

@torch.no_grad()
def _bench_infer(model_type):
    torch.cuda.empty_cache()
    _m,_ = build_model(model_type, _bb, aux_lambda=0.1, lr=2e-5); _m.eval()
    _times=[]
    for _i,_b in enumerate(_batches):
        _t0=_time.perf_counter()
        with autocast(enabled=USE_AMP):
            if model_type=="baseline":
                _m(input_ids=_b["pair_input_ids"], attention_mask=_b["pair_attention_mask"])
            else:  # classification path only — what deployment uses
                _m.classifier(_m._pool(_m.bert, _b["pair_input_ids"], _b["pair_attention_mask"]))
        torch.cuda.synchronize()
        if _i>=5: _times.append(_time.perf_counter()-_t0)
    del _m; torch.cuda.empty_cache()
    return 1000*float(np.mean(_times))

overhead = {}
for _mt in ("baseline","jepa"):
    overhead[_mt] = _bench_train(_mt); overhead[_mt]["infer_ms_per_batch"] = _bench_infer(_mt)
    print(f"{_mt}: {overhead[_mt]}")
_r = overhead
print(f"\nTraining overhead: {_r['jepa']['ms_per_step']/_r['baseline']['ms_per_step']:.2f}x time, "
      f"{_r['jepa']['peak_gb']/_r['baseline']['peak_gb']:.2f}x peak memory")
print(f"Inference overhead: {_r['jepa']['infer_ms_per_batch']/_r['baseline']['infer_ms_per_batch']:.2f}x "
      f"(classification path only; target encoder & heads dropped at deployment)")
print("Analytic: JEPA-Reg = 3 grad forward passes (pair,s1,s2) + 2 no-grad target passes vs 1 for baseline.")
import json as _json
with open("results/overhead.json","w") as _f: _json.dump(overhead, _f, indent=2)
print("✅ R-8 done → results/overhead.json (fill Sec. 'computationally light' claims with these numbers)")



# ---- consolidated fresh-results pack ----
def pack(runs):
    f1s=[r["test"]["f1"] for r in runs]; accs=[r["test"].get("accuracy",0) for r in runs]
    import numpy as np
    return {"f1_mean":float(np.mean(f1s)),"f1_std":float(np.std(f1s)),
            "acc_mean":float(np.mean(accs)),"acc_std":float(np.std(accs)),
            "f1s":[float(x) for x in f1s]}
save={"bert-base-uncased":{},"hans":hans_results}
for ds in ("mrpc","qqp","paws"):
    save["bert-base-uncased"][ds]={mt:pack(r) for mt,r in all_results["bert-base-uncased"][ds].items()}
save["bert-base-uncased"]["lr_mrpc"]={k:{mt:{"f1_mean":float(np.mean(v)),"f1_std":float(np.std(v))} for mt,v in fd.items()} for k,fd in all_results["bert-base-uncased"]["lr_mrpc"].items()}
save["bert-base-uncased"]["best_lambdas"]={"mrpc":all_results["bert-base-uncased"]["best_lam_mrpc"],"qqp":all_results["bert-base-uncased"]["best_lam_qqp"],"paws":all_results["bert-base-uncased"]["best_lam_paws"]}
with open("results/all_results_fresh_v2.json","w") as f: json.dump(save,f,indent=2,default=float)
print("saved results/all_results_fresh_v2.json")
print("NOW RUNNING R-6 (pure JEPA on QQP)...")


# ═══ R-6 (Reviewer 2; Reviewer 4, item 7): pure JEPA-Reg on QQP ═══
# Table 4's QQP row currently shows only the Hybrid variant. Run pure JEPA-Reg so the
# main table can report JEPA-Reg itself (Hybrid moves to the analysis section).
_bb = "bert-base-uncased"; _cfg = BACKBONE_CONFIGS[_bb]
_tok,_col,_ds,_ldr = make_loaders(_bb)
_lam = all_results[_bb].get("best_lam_qqp", 0.1)
qqp_pure_jepa = []
for _seed in SEEDS:
    _r = run_experiment(_ldr["qqp_train"], _ldr["qqp_val"], _ldr["qqp_test"],
                        "jepa", _bb, _seed, lr=_cfg["lr_qqp"], aux_lambda=_lam,
                        epochs=_cfg["epochs_qqp"], grad_accum=_cfg["grad_accum"],
                        tag=f"QQP/pure-jepa/s{_seed}")
    qqp_pure_jepa.append(_r["test"]["f1"])
    print(f"  seed {_seed}: QQP pure JEPA-Reg F1 = {_r['test']['f1']:.4f}")
_b = [x["test"]["f1"] for x in all_results[_bb]["qqp"]["baseline"]]
print(f"\nQQP pure JEPA-Reg: {np.mean(qqp_pure_jepa):.4f} ± {np.std(qqp_pure_jepa):.4f}"
      f"  (baseline {np.mean(_b):.4f} ± {np.std(_b):.4f}, d={cohen_d(qqp_pure_jepa,_b):+.2f})")
import json as _json
with open("results/qqp_pure_jepa.json","w") as _f:
    _json.dump({"f1s": qqp_pure_jepa, "baseline_f1s": _b}, _f, indent=2)
print("✅ R-6 done → results/qqp_pure_jepa.json (this is the number Table 4 must show for QQP)")



print("STAGE 2 COMPLETE")
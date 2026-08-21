# Smoke test: one JEPA training step + one baseline step on real MRPC data.
# Fails fast if any library API changed. Runs in ~2-3 min (mostly model download).
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("_setup_extracted.py", encoding="utf-8").read())

print("\n--- SMOKE TEST ---")
bb = "bert-base-uncased"
tok, col, ds, ldr = make_loaders(bb)
batch = next(iter(ldr["mrpc_train"]))
batch = _move(batch)

for mt in ("baseline", "jepa", "simcse", "hybrid"):
    m, opt = build_model(mt, bb, aux_lambda=0.1, lr=2e-5)
    sc = GradScaler(enabled=USE_AMP)
    with autocast(enabled=USE_AMP):
        if mt == "baseline":
            out = m(input_ids=batch["pair_input_ids"], attention_mask=batch["pair_attention_mask"], labels=batch["labels"])
            loss = F.cross_entropy(out.logits, batch["labels"], label_smoothing=LABEL_SMOOTHING)
        else:
            fwd = {k: batch[k] for k in JEPA_KEYS if k in batch}
            if mt in ("jepa", "hybrid"): fwd["jepa_lambda"] = 0.1
            loss = m(**fwd)["loss"]
    sc.scale(loss).backward()
    sc.step(opt); sc.update()
    if mt in ("jepa", "hybrid"): m.update_ema()
    print(f"  {mt:<10} one step OK, loss={float(loss):.4f}")
    del m, opt
    torch.cuda.empty_cache()

print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("SMOKE TEST PASSED")

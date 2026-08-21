# AUTO-EXTRACTED from JEPA_final_optionB_fixed.ipynb — setup sections
# ======== notebook cell 4 ========
# ═══════════════════════════════════════════════════════════
#  JEPA-Reg — Option B: Adversarial Lexical Overlap Robustness
#  Target: ACL/EMNLP Findings  or  *SEM / RepL4NLP Workshop
# ═══════════════════════════════════════════════════════════

PAPER_TITLE = (
    "JEPA-Reg: Predictive Representation Regularization Improves Robustness "
    "to Adversarial Lexical Overlap in Paraphrase Detection"
)

PAPER_CLAIM = (
    "JEPA-style EMA predictive auxiliary loss regularizes the projection space "
    "of BERT, creating an inductive bias toward structurally-grounded semantic representations. "
    "This improves robustness on datasets where lexical overlap is a misleading surface cue "
    "(PAWS, HANS), while requiring no negative pair sampling. "
    "Stop-gradient is applied to the TARGET encoder only (consistent with I-JEPA design)."
)

BACKBONE_CONFIGS = {
    "bert-base-uncased": {
        "batch_size":  32,
        "lr_mrpc":     3e-5,
        "lr_qqp":      1e-5,
        "lr_paws":     2e-5,
        "epochs_mrpc": 3,
        "epochs_qqp":  3,
        "epochs_paws": 3,
        "grad_accum":  1,
    },
    "roberta-base": {
        "batch_size":  8,   # reduced from 16 to prevent OOM on PAWS/JEPA
        "lr_mrpc":     2e-5,
        "lr_qqp":      1e-5,
        "lr_paws":     2e-5,
        "epochs_mrpc": 3,
        "epochs_qqp":  3,
        "epochs_paws": 3,
        "grad_accum":  1,
    },
}

BACKBONES        = ["bert-base-uncased", "roberta-base"]
MAX_LEN          = 128
SEEDS            = [42, 43, 44]   # n=3 seeds — Cohen's d is primary metric
EMA_DECAY        = 0.999
PROJ_DIM         = 256
WARMUP_RATIO     = 0.20
LABEL_SMOOTHING  = 0.05
JEPA_LABEL_SMOOTHING = 0.0
TUNE_FRAC        = 0.10
TUNE_EPOCHS      = 1
LAMBDA_CANDS     = [0.05, 0.1, 0.3, 0.5]
LAMBDA_WARMUP    = True
N_BOOTSTRAP      = 2_000
USE_AMP          = True
HYBRID_JEPA_W    = 0.7
HYBRID_SIMCSE_W  = 0.3
QQP_TRAIN_MAX    = 20_000
QQP_VAL_MAX      = 5_000
PAWS_TRAIN_MAX   = 20_000
PAWS_VAL_MAX     = 5_000
LR_FRACTIONS     = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]

TSNE_SAMPLE      = 500
TSNE_PERPLEXITY  = 30
TSNE_ITER        = 1_000

import os, time, copy, json, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, ttest_rel
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
import transformers as _transformers
_transformers.logging.set_verbosity_error()  # suppress MISSING/UNEXPECTED load warnings
from datasets import load_dataset

# Make CUDA errors report at the correct line instead of a random later op
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)
BEST_MODELS = {}

print(f"Device        : {device}")
print(f"Paper framing : Option B — Adversarial Lexical Overlap Robustness")
print(f"Seeds         : {SEEDS}  (n={len(SEEDS)}, df={len(SEEDS)-1})")
print(f"Datasets      : MRPC, QQP, PAWS, HANS (zero-shot)")
if torch.cuda.is_available():
    print(f"Free VRAM     : {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

USE_LLRD   = True
LLRD_DECAY = 0.9
JEPA_POOL  = "mean"

# ======== notebook cell 6 ========
def make_collator(tokenizer, max_len=MAX_LEN):
    def collate(batch):
        s1=[b.get("s1") or b.get("sentence1") for b in batch]
        s2=[b.get("s2") or b.get("sentence2") for b in batch]
        labels=torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        def enc(a, b=None):
            kw=dict(padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            return tokenizer(a, b, **kw) if b else tokenizer(a, **kw)
        pair=enc(s1,s2); e1=enc(s1); e2=enc(s2)
        return {"pair_input_ids":   pair["input_ids"],
                "pair_attention_mask": pair["attention_mask"],
                "s1_input_ids":    e1["input_ids"],
                "s1_attention_mask": e1["attention_mask"],
                "s2_input_ids":    e2["input_ids"],
                "s2_attention_mask": e2["attention_mask"],
                "labels": labels}
    return collate
print("✅ Collator ready  (num_workers=0 — no pickle error)")

# ======== notebook cell 8 ========
# ── JEPA (mean pooling, EMA=0.999) ───────────────────────────────
#
# ARCHITECTURE CLARIFICATION (FIX P1):
# stop_gradient is applied ONLY to target_bert / target_proj.
# The online self.bert receives full gradients from the JEPA aux loss:
#   loss = cls_loss + λ * jepa_loss
#   jepa_loss = f(proj(bert(s1)), proj(bert(s2)), target_proj(target_bert(...)))
# → gradient flows: jepa_loss → predictor → proj → bert (online encoder)
# This is the correct I-JEPA / SimSiam design: target is frozen EMA,
# online encoder is fully trainable. Cite: Assran et al. 2023, Chen & He 2021.
#
class JepaBertPair(nn.Module):
    def __init__(self, bert_name, num_labels=2, proj_dim=PROJ_DIM,
                 ema_decay=EMA_DECAY, jepa_lambda=1.0,
                 label_smoothing=JEPA_LABEL_SMOOTHING):
        super().__init__()
        self.ema_decay=ema_decay; self.jepa_lambda=jepa_lambda
        self.label_smoothing=label_smoothing
        self.bert=AutoModel.from_pretrained(bert_name, add_pooling_layer=False)
        hidden=self.bert.config.hidden_size
        self.classifier=nn.Linear(hidden, num_labels)
        self.proj=nn.Sequential(
            nn.Linear(hidden,hidden), nn.GELU(), nn.Linear(hidden,proj_dim))
        self.predictor=nn.Sequential(
            nn.Linear(proj_dim,proj_dim), nn.GELU(), nn.Linear(proj_dim,proj_dim))
        # TARGET encoder: EMA copy — parameters frozen, updated via update_ema()
        self.target_bert=copy.deepcopy(self.bert)
        self.target_proj=copy.deepcopy(self.proj)
        for p in list(self.target_bert.parameters())+list(self.target_proj.parameters()):
            p.requires_grad_(False)  # stop_gradient on TARGET only

    @torch.no_grad()
    def update_ema(self):
        d=self.ema_decay
        for o,t in zip(self.bert.parameters(),self.target_bert.parameters()):
            t.data.mul_(d).add_(o.data,alpha=1-d)
        for o,t in zip(self.proj.parameters(),self.target_proj.parameters()):
            t.data.mul_(d).add_(o.data,alpha=1-d)

    def _pool(self,enc,iids,amask):
        out=enc(input_ids=iids,attention_mask=amask).last_hidden_state
        if JEPA_POOL=="mean":
            mask=amask.unsqueeze(-1).float()
            return (out*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        return out[:,0]  # CLS fallback

    def forward(self,pair_input_ids,pair_attention_mask,
                s1_input_ids,s1_attention_mask,
                s2_input_ids,s2_attention_mask,
                labels=None,jepa_lambda=None,**kw):
        lam=jepa_lambda if jepa_lambda is not None else self.jepa_lambda

        # Classification head (shared BERT backbone)
        logits=self.classifier(self._pool(self.bert,pair_input_ids,pair_attention_mask))
        cls_loss=F.cross_entropy(logits,labels,label_smoothing=self.label_smoothing) \
                  if labels is not None else None

        # JEPA auxiliary loss:
        # Gradient flows: jepa_loss → predictor → proj → self.bert (ONLINE, trainable)
        # target_bert/target_proj: torch.no_grad() — EMA-only update via update_ema()
        s1=self._pool(self.bert,s1_input_ids,s1_attention_mask)   # grad-enabled
        s2=self._pool(self.bert,s2_input_ids,s2_attention_mask)   # grad-enabled
        z1,z2=self.proj(s1),self.proj(s2)
        with torch.no_grad():
            # Target encoder: stop_gradient (TARGET ONLY)
            t1=F.normalize(self.target_proj(
                self._pool(self.target_bert,s1_input_ids,s1_attention_mask)),dim=-1)
            t2=F.normalize(self.target_proj(
                self._pool(self.target_bert,s2_input_ids,s2_attention_mask)),dim=-1)
        p12=F.normalize(self.predictor(z1),dim=-1)
        p21=F.normalize(self.predictor(z2),dim=-1)
        sym=((1-(p12*t2).sum(-1))+(1-(p21*t1).sum(-1)))/2.0

        if labels is not None:
            # THEORETICAL MOTIVATION (FIX P5):
            # Applying JEPA loss ONLY to paraphrase pairs (label=1) creates a direct
            # inductive bias: the encoder must map semantically equivalent sentences
            # to mutually predictable representations in projection space.
            # Excluding non-paraphrase pairs is correct: structurally different sentences
            # should NOT be predictive of each other's latent state.
            # This mirrors predictive coding theory (Rao & Ballard 1999):
            # the model learns to predict the latent state of a semantically related input.
            mask=(labels==1).float()
            jepa_loss=(sym*mask).sum()/mask.sum().clamp(min=1)
        else:
            jepa_loss=sym.mean()

        loss=(cls_loss+lam*jepa_loss) if cls_loss is not None else None
        return {"loss":loss,"logits":logits,"cls_loss":cls_loss,"jepa_loss":jepa_loss}

print("✅ JepaBertPair — online BERT receives full JEPA gradient (stop_grad on TARGET only)")

# ======== notebook cell 9 ========
# ── HybridJepaSimCSE (for QQP) ──────────────────────────────────
class HybridJepaSimCSE(nn.Module):
    def __init__(self,bert_name,num_labels=2,proj_dim=PROJ_DIM,
                 ema_decay=EMA_DECAY,jepa_lambda=1.0,
                 jepa_w=HYBRID_JEPA_W,simcse_w=HYBRID_SIMCSE_W,
                 label_smoothing=LABEL_SMOOTHING,temp=0.05):
        super().__init__()
        self.jepa_lambda=jepa_lambda; self.jepa_w=jepa_w
        self.simcse_w=simcse_w; self.label_smoothing=label_smoothing
        self.ema_decay=ema_decay; self.temp=temp
        self.bert=AutoModel.from_pretrained(bert_name, add_pooling_layer=False)
        hidden=self.bert.config.hidden_size
        self.classifier=nn.Linear(hidden,num_labels)
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,proj_dim))
        self.predictor=nn.Sequential(nn.Linear(proj_dim,proj_dim),nn.GELU(),nn.Linear(proj_dim,proj_dim))
        self.target_bert=copy.deepcopy(self.bert)
        self.target_proj=copy.deepcopy(self.proj)
        for p in list(self.target_bert.parameters())+list(self.target_proj.parameters()):
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self):
        d=self.ema_decay
        for o,t in zip(self.bert.parameters(),self.target_bert.parameters()):
            t.data.mul_(d).add_(o.data,alpha=1-d)
        for o,t in zip(self.proj.parameters(),self.target_proj.parameters()):
            t.data.mul_(d).add_(o.data,alpha=1-d)

    def _cls(self,enc,iids,amask):
        return enc(input_ids=iids,attention_mask=amask).last_hidden_state[:,0]

    def forward(self,pair_input_ids,pair_attention_mask,
                s1_input_ids,s1_attention_mask,
                s2_input_ids,s2_attention_mask,
                labels=None,jepa_lambda=None,**kw):
        lam=jepa_lambda if jepa_lambda is not None else self.jepa_lambda
        logits=self.classifier(self._cls(self.bert,pair_input_ids,pair_attention_mask))
        cls_loss=F.cross_entropy(logits,labels,label_smoothing=self.label_smoothing)                  if labels is not None else None
        # JEPA loss flows back through proj → BERT backbone (true regularization)
        s1=self._cls(self.bert,s1_input_ids,s1_attention_mask)
        s2=self._cls(self.bert,s2_input_ids,s2_attention_mask)
        z1,z2=self.proj(s1),self.proj(s2)
        with torch.no_grad():
            t1=F.normalize(self.target_proj(self._cls(self.target_bert,s1_input_ids,s1_attention_mask)),dim=-1)
            t2=F.normalize(self.target_proj(self._cls(self.target_bert,s2_input_ids,s2_attention_mask)),dim=-1)
        p12=F.normalize(self.predictor(z1),dim=-1); p21=F.normalize(self.predictor(z2),dim=-1)
        sym=((1-(p12*t2).sum(-1))+(1-(p21*t1).sum(-1)))/2.0
        if labels is not None:
            mask=(labels==1).float()
            jepa_loss=(sym*mask).sum()/mask.sum().clamp(min=1)
        else:
            jepa_loss=sym.mean()
        c1=F.normalize(self._cls(self.bert,s1_input_ids,s1_attention_mask),dim=-1)
        c2=F.normalize(self._cls(self.bert,s2_input_ids,s2_attention_mask),dim=-1)
        B=c1.size(0)
        simcse_loss=F.cross_entropy(torch.mm(c1,c2.T)/self.temp,torch.arange(B,device=c1.device))
        aux=self.jepa_w*lam*jepa_loss+self.simcse_w*simcse_loss
        loss=(cls_loss+aux) if cls_loss is not None else None
        return {"loss":loss,"logits":logits,"cls_loss":cls_loss,
                "jepa_loss":jepa_loss,"simcse_loss":simcse_loss}

# ── SimCSE ────────────────────────────────────────────────────────
class SimCSEBertPair(nn.Module):
    def __init__(self,bert_name,num_labels=2,temp=0.05,lam=0.1,
                 label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.temp=temp; self.lam=lam; self.ls=label_smoothing
        self.bert=AutoModel.from_pretrained(bert_name, add_pooling_layer=False)
        self.classifier=nn.Linear(self.bert.config.hidden_size,num_labels)
    def _cls(self,iids,amask):
        return self.bert(input_ids=iids,attention_mask=amask).last_hidden_state[:,0]
    def forward(self,pair_input_ids,pair_attention_mask,s1_input_ids,s1_attention_mask,
                s2_input_ids=None,s2_attention_mask=None,labels=None,**kw):
        logits=self.classifier(self._cls(pair_input_ids,pair_attention_mask))
        cls_loss=F.cross_entropy(logits,labels,label_smoothing=self.ls) if labels is not None else None
        z1=F.normalize(self._cls(s1_input_ids,s1_attention_mask),dim=-1)
        z2=F.normalize(self._cls(s2_input_ids,s2_attention_mask),dim=-1)
        B=z1.size(0)
        cl=F.cross_entropy(torch.mm(z1,z2.T)/self.temp,torch.arange(B,device=z1.device))
        loss=(cls_loss+self.lam*cl) if cls_loss is not None else None
        return {"loss":loss,"logits":logits,"cls_loss":cls_loss,"simcse_loss":cl}

print("✅ HybridJepaSimCSE + SimCSEBertPair ready (detach removed, s2 bug fixed)")

# ======== notebook cell 11 ========
JEPA_KEYS=["pair_input_ids","pair_attention_mask",
           "s1_input_ids","s1_attention_mask",
           "s2_input_ids","s2_attention_mask","labels"]

def _move(b):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}

def make_llrd_params(model, base_lr, decay=LLRD_DECAY):
    """Layer-wise LR decay: top layers get base_lr, lower layers get base_lr*decay^n."""
    if not USE_LLRD:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "embeddings" in name: depth = 0
        elif ".layer." in name:
            try: depth = int(name.split(".layer.")[1].split(".")[0]) + 1
            except: depth = 0
        else: depth = 13  # classifier / projection head — highest LR
        groups.setdefault(depth, []).append(param)
    max_depth = max(groups.keys()) if groups else 13
    return [{"params": ps, "lr": base_lr * (decay ** (max_depth - d))} for d, ps in groups.items()]

def build_model(model_type, bert_name, aux_lambda=1.0, lr=2e-5):
    if   model_type=="baseline": m=AutoModelForSequenceClassification.from_pretrained(bert_name,num_labels=2).to(device); m=m.float() if "deberta" in bert_name else m
    elif model_type=="jepa":     m=JepaBertPair(bert_name,jepa_lambda=aux_lambda).to(device); m=m.float() if "deberta" in bert_name else m; m.bert.encoder.gradient_checkpointing = True if "roberta" in bert_name else False
    elif model_type=="simcse":   m=SimCSEBertPair(bert_name).to(device); m=m.float() if "deberta" in bert_name else m
    elif model_type=="hybrid":   m=HybridJepaSimCSE(bert_name,jepa_lambda=aux_lambda).to(device); m=m.float() if "deberta" in bert_name else m; m.bert.encoder.gradient_checkpointing = True if "roberta" in bert_name else False
    pg  = make_llrd_params(m, lr)
    opt = AdamW(pg, lr=lr, weight_decay=0.01)
    return m, opt

def set_lr(opt,lr):
    for g in opt.param_groups: g["lr"]=lr

def train_epoch(model,loader,optimizer,scheduler,scaler,
                model_type="baseline",aux_lambda=1.0,grad_accum=1):
    model.train()
    total_loss=0.; total_cls=0.; total_aux=0.
    preds_all=[]; labels_all=[]; n=0
    n_steps=len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step,batch in enumerate(loader):
        batch=_move(batch)
        labels=batch["labels"].clamp(0, 1)  # guard against invalid PAWS labels
        # Lambda warmup: ramp 0→λ over first epoch to prevent collapse
        eff_lam = aux_lambda * min(1.0,(step+1)/max(1,n_steps)) if LAMBDA_WARMUP else aux_lambda

        with autocast(enabled=USE_AMP):
            if model_type=="baseline":
                out=model(input_ids=batch["pair_input_ids"],
                          attention_mask=batch["pair_attention_mask"],labels=labels)
                loss=F.cross_entropy(out.logits,labels,label_smoothing=LABEL_SMOOTHING)  # baseline only
                logits=out.logits; cls_v=float(loss.detach().cpu()); aux_v=0.
            else:
                fwd={k:batch[k] for k in JEPA_KEYS if k in batch}
                if model_type in ("jepa","hybrid"): fwd["jepa_lambda"]=eff_lam
                out=model(**fwd); loss,logits=out["loss"],out["logits"]
                cls_v=float(out.get("cls_loss",loss).detach().cpu())
                if   model_type=="jepa":    aux_v=float(out["jepa_loss"].detach().cpu())
                elif model_type=="simcse":  aux_v=float(out["simcse_loss"].detach().cpu())
                elif model_type=="hybrid":  aux_v=float((out["jepa_loss"]+out["simcse_loss"]).detach().cpu()/2)
            loss=loss/grad_accum

        scaler.scale(loss).backward()
        if (step+1)%grad_accum==0 or (step+1)==n_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if model_type in ("jepa","hybrid"): model.update_ema()

        total_loss+=float(loss.detach().cpu())*grad_accum
        total_cls+=cls_v; total_aux+=aux_v; n+=1
        preds_all.extend(torch.argmax(logits,1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    nb=max(1,n)
    return total_loss/nb,accuracy_score(labels_all,preds_all),total_cls/nb,total_aux/nb

@torch.no_grad()
def evaluate_model(model,loader,model_type="baseline"):
    model.eval(); total_loss=0.; preds_all=[]; labels_all=[]
    for batch in loader:
        batch=_move(batch)
        labels=batch["labels"].clamp(0, 1)
        with autocast(enabled=USE_AMP):
            if model_type=="baseline":
                out=model(input_ids=batch["pair_input_ids"],attention_mask=batch["pair_attention_mask"],labels=labels)
                loss,logits=out.loss,out.logits
            else:
                fwd={k:batch[k] for k in JEPA_KEYS if k in batch}
                if model_type in ("jepa","hybrid"): fwd["jepa_lambda"]=0.0
                out=model(**fwd); loss=out.get("cls_loss",out["loss"]); logits=out["logits"]
        total_loss+=float(loss.detach().cpu())
        preds_all.extend(torch.argmax(logits,1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return {"loss":total_loss/max(1,len(loader)),
            "f1":f1_score(labels_all,preds_all,zero_division=0),
            "accuracy":accuracy_score(labels_all,preds_all),
            "preds":preds_all,"labels":labels_all}

def run_experiment(train_loader,val_loader,test_loader,
                   model_type,bert_name,seed,lr=2e-5,
                   aux_lambda=1.0,epochs=3,grad_accum=1,
                   tag="",save_key=None):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler=GradScaler(enabled=USE_AMP)
    model,opt=build_model(model_type,bert_name,aux_lambda,lr=lr)
    eff=max(1,len(train_loader)*epochs//grad_accum)
    sched=get_linear_schedule_with_warmup(opt,max(1,int(eff*WARMUP_RATIO)),eff)
    best_val_f1=0.; best_state=None
    history={"cls_loss":[],"aux_loss":[],"total_loss":[],"val_f1":[]}
    for ep in range(1,epochs+1):
        t0=time.time()
        tr_loss,tr_acc,cls_l,aux_l=train_epoch(model,train_loader,opt,sched,scaler,model_type,aux_lambda,grad_accum)
        val=evaluate_model(model,val_loader,model_type)
        history["total_loss"].append(tr_loss); history["cls_loss"].append(cls_l)
        history["aux_loss"].append(aux_l);     history["val_f1"].append(val["f1"])
        print(f"  [{tag} ep{ep}/{epochs}] loss={tr_loss:.4f} acc={tr_acc:.3f} aux={aux_l:.4f} | val_f1={val['f1']:.4f} | {time.time()-t0:.0f}s")
        if val["f1"]>best_val_f1: best_val_f1=val["f1"]; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test=evaluate_model(model,test_loader,model_type)
    if save_key and seed==SEEDS[0]:
        BEST_MODELS[save_key]=copy.deepcopy(best_state)
        # Also persist to disk so kernel restarts don't lose it
        import os as _os; _os.makedirs('results/models', exist_ok=True)
        torch.save(best_state, f'results/models/{save_key}.pt')
    del model; torch.cuda.empty_cache()
    return {"val":val,"test":test,"history":history}

print("✅ Training utilities ready")
print(f"   Lambda warmup: {LAMBDA_WARMUP}  EMA: {EMA_DECAY}  Warmup: {WARMUP_RATIO}")

# ======== notebook cell 13 ========
def bootstrap_p(a,b,n=N_BOOTSTRAP,seed=0):
    rng=np.random.default_rng(seed); a,b=np.array(a),np.array(b)
    obs=np.mean(a)-np.mean(b); diff=a-b
    count=sum(abs(rng.choice(diff,len(diff),replace=True).mean()-diff.mean())>=abs(obs) for _ in range(n))
    return count/n

def ci95(scores,n=N_BOOTSTRAP,seed=0):
    rng=np.random.default_rng(seed); arr=np.array(scores)
    boot=[np.mean(rng.choice(arr,len(arr),replace=True)) for _ in range(n)]
    return np.percentile(boot,[2.5,97.5])

def stars(p):
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

# FIX P3: Cohen's d effect size — more informative than p-value at small n.
# d = 0.2 small, 0.5 medium, 0.8 large, >1.2 very large.
# At n=3, p-values are almost always non-significant even for d=2.0.
# Reporting d allows reviewers to assess practical significance.
def cohen_d(a, b):
    """Cohen's d effect size between two groups a and b."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / max(pooled_std, 1e-9))

# Paired t-test (complement to bootstrap; requires n>=2)
def paired_ttest_p(a, b):
    """Two-sided paired t-test p-value."""
    if len(a) < 2: return 1.0
    _, p = ttest_rel(a, b)
    return float(p)

def effect_label(d):
    ad = abs(d)
    if   ad >= 1.2: return "very large"
    elif ad >= 0.8: return "large"
    elif ad >= 0.5: return "medium"
    elif ad >= 0.2: return "small"
    else:           return "negligible"

print(f"✅ Stats: bootstrap ({N_BOOTSTRAP:,} samples) + Cohen's d + paired t-test")

# ======== notebook cell 15 ========
def select_lambda(train_ds,collator,bert_name,lr=2e-5,
                  grad_accum=1,seed=0,model_type="jepa"):
    n_tune=max(32,int(len(train_ds)*TUNE_FRAC))
    n_small=len(train_ds)-n_tune
    g=torch.Generator().manual_seed(seed)
    small_ds,tune_ds=random_split(train_ds,[n_small,n_tune],generator=g)
    bs=BACKBONE_CONFIGS[bert_name]["batch_size"]
    small_ldr=DataLoader(small_ds,bs,shuffle=True,collate_fn=collator,num_workers=0)
    tune_ldr =DataLoader(tune_ds,64,shuffle=False,collate_fn=collator,num_workers=0)
    best_lam,best_f1=LAMBDA_CANDS[0],-1.
    print(f"  λ sweep ({n_small} train, {n_tune} tune, TUNE_EPOCHS={TUNE_EPOCHS}, model={model_type}):")
    for lam in LAMBDA_CANDS:
        torch.manual_seed(seed); np.random.seed(seed)
        sc=GradScaler(enabled=USE_AMP); m,opt=build_model(model_type,bert_name,lam); set_lr(opt,lr)
        eff=max(1,len(small_ldr)*TUNE_EPOCHS//grad_accum)
        sched=get_linear_schedule_with_warmup(opt,max(1,int(eff*WARMUP_RATIO)),eff)
        for _ in range(TUNE_EPOCHS):
            train_epoch(m,small_ldr,opt,sched,sc,model_type,lam,grad_accum)
        r=evaluate_model(m,tune_ldr,model_type)
        print(f"    λ={lam:<5} tune_f1={r['f1']:.4f}")
        if r["f1"]>best_f1: best_f1,best_lam=r["f1"],lam
        del m; torch.cuda.empty_cache()
    print(f"  → Best λ={best_lam}  (tune_f1={best_f1:.4f})")
    return best_lam
print("✅ Lambda selector  (4 candidates, TUNE_EPOCHS=1)")

# ======== notebook cell 17 ========
class SimpleDataset(Dataset):
    def __init__(self,hf_ds,s1="sentence1",s2="sentence2",lbl="label"):
        self.data=list(hf_ds); self.s1=s1; self.s2=s2; self.lbl=lbl  # plain list — avoids HF batch-index bug
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        ex=self.data[idx]
        lbl=int(ex[self.lbl])
        assert 0 <= lbl <= 1, f'Invalid label {lbl} in {self.lbl} at idx {idx}'
        return {"s1":ex[self.s1],"s2":ex[self.s2],"labels":lbl}

print("Loading all datasets (one-time)...")
mrpc_raw = load_dataset("nyu-mll/glue", "mrpc")
qqp_raw  = load_dataset("nyu-mll/glue", "qqp")
paws_raw = load_dataset("google-research-datasets/paws", "labeled_final")

# Load HANS directly from GitHub raw TSV.
# The HuggingFace jhu-cogsci/hans repo uses a legacy loading script (hans.py)
# which is no longer supported by the datasets library (v3+).
# We load the original authors' TSV files directly instead.
HANS_EVAL_URL  = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
HANS_TRAIN_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"

def _load_hans_tsv(url):
    import urllib.request, csv, io
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    label_map = {"entailment": 0, "non-entailment": 1}
    return [{
        "premise":   row["sentence1"],
        "hypothesis":row["sentence2"],
        "label":     label_map.get(row["gold_label"], -1),
        "heuristic": row["heuristic"],
        "subcase":   row["subcase"],
    } for row in reader]

try:
    from datasets import Dataset as HFDataset, DatasetDict  # renamed — avoids shadowing torch Dataset
    hans_raw = DatasetDict({
        "validation": HFDataset.from_list(_load_hans_tsv(HANS_EVAL_URL)),
        "train":      HFDataset.from_list(_load_hans_tsv(HANS_TRAIN_URL)),
    })
    HANS_SPLIT = "validation"
    print(f"HANS loaded from GitHub: {len(hans_raw[HANS_SPLIT]):,} eval examples")
    print(f"  heuristics: {sorted(set(hans_raw[HANS_SPLIT]['heuristic']))}")
except Exception as e:
    print(f"WARNING: Could not load HANS ({e}) -- evaluation will be skipped.")
    hans_raw = None
    HANS_SPLIT = None

stsb_raw = load_dataset("nyu-mll/glue", "stsb")   # diagnostic only

# QQP splits
qqp_tr_hf = qqp_raw["train"].shuffle(seed=42).select(range(QQP_TRAIN_MAX))
qqp_va_hf = qqp_raw["validation"].shuffle(seed=42).select(range(QQP_VAL_MAX))
n_tst     = max(1, int(len(qqp_tr_hf)*0.10))
qqp_tr2   = qqp_tr_hf.select(range(len(qqp_tr_hf)-n_tst))
qqp_te    = qqp_tr_hf.select(range(len(qqp_tr_hf)-n_tst, len(qqp_tr_hf)))

# PAWS splits
# Filter PAWS: drop rows with empty sentences OR invalid labels (-1, 2, etc.)
# Invalid labels cause cudaErrorIllegalAddress in cross_entropy CUDA kernel
_paws_ok = lambda x: bool(x["sentence1"]) and bool(x["sentence2"]) and x["label"] in (0, 1)
paws_tr = paws_raw["train"].filter(_paws_ok)
paws_va = paws_raw["validation"].filter(_paws_ok)
paws_te = paws_raw["test"].filter(_paws_ok)
if PAWS_TRAIN_MAX: paws_tr = paws_tr.shuffle(seed=42).select(range(min(PAWS_TRAIN_MAX, len(paws_tr))))
if PAWS_VAL_MAX:   paws_va = paws_va.shuffle(seed=42).select(range(min(PAWS_VAL_MAX,  len(paws_va))))

def make_loaders(bert_name):
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=("deberta" not in bert_name))
    col = make_collator(tok)
    bs  = BACKBONE_CONFIGS[bert_name]["batch_size"]
    ds  = {
        "mrpc_train": SimpleDataset(mrpc_raw["train"]),
        "mrpc_val":   SimpleDataset(mrpc_raw["validation"]),
        "mrpc_test":  SimpleDataset(mrpc_raw["test"]),
        "qqp_train":  SimpleDataset(qqp_tr2,   "question1","question2","label"),
        "qqp_val":    SimpleDataset(qqp_va_hf, "question1","question2","label"),
        "qqp_test":   SimpleDataset(qqp_te,    "question1","question2","label"),
        "paws_train": SimpleDataset(paws_tr),
        "paws_val":   SimpleDataset(paws_va),
        "paws_test":  SimpleDataset(paws_te),
    }
    ldr = {}
    for name, d in ds.items():
        bsz = bs if "train" in name else 64
        ldr[name] = DataLoader(d, bsz, shuffle="train" in name, collate_fn=col, num_workers=0)
    return tok, col, ds, ldr

print(f"MRPC  {len(mrpc_raw['train']):,} / {len(mrpc_raw['validation']):,} / {len(mrpc_raw['test']):,}")
print(f"QQP   {len(qqp_tr2):,} / {len(qqp_va_hf):,} / {len(qqp_te):,}")
print(f"PAWS  {len(paws_tr):,} / {len(paws_va):,} / {len(paws_te):,}")
if hans_raw:
    print(f"HANS  {len(hans_raw[HANS_SPLIT]):,} examples (zero-shot eval only)")
else:
    print("HANS  not available — will skip")
print(f"STS-B {len(stsb_raw['validation']):,} (diagnostic only)")
print("✅ Datasets loaded")

# ======== notebook cell 19 ========
def run_backbone(bert_name, skip_lr_mrpc=False, fixed_lambda=None, qqp_train_max=None,
                 resume=None):
    """
    resume: dict of already-completed results (loaded from partial checkpoint).
            Sections already present are skipped automatically.
    """
    import os as _os, json as _json

    cfg   = BACKBONE_CONFIGS[bert_name]
    bname = bert_name.split('/')[-1]
    bs    = cfg['batch_size']; ga = cfg['grad_accum']
    ckpt_path = f'results/checkpoint_{bname}_partial.json'

    print('\n' + '=' * 60)
    print(f'  BACKBONE: {bert_name}')
    print(f'  batch={bs}  eff_batch={bs * ga}')
    if torch.cuda.is_available():
        print(f'  Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')
    print('=' * 60)

    tok, col, ds, ldr = make_loaders(bert_name)

    # Start from prior partial results if provided
    results = dict(resume) if resume else {}

    def _save_partial():
        """Save whatever is done so far so a crash mid-run is resumable."""
        _os.makedirs('results', exist_ok=True)
        with open(ckpt_path, 'w') as _f:
            _json.dump(results, _f, indent=2, default=float)
        print(f'  [checkpoint] partial results saved → {ckpt_path}')

    fractions = LR_FRACTIONS

    # ── MRPC ──────────────────────────────────────────────────────────────────
    if 'mrpc' in results:
        print(f'\n-- MRPC  [SKIPPED — already in checkpoint]')
        best_lam_mrpc = results['best_lam_mrpc']
    else:
        print(f'\n-- MRPC  (epochs={cfg["epochs_mrpc"]})')
        best_lam_mrpc = fixed_lambda or select_lambda(
            ds['mrpc_train'], col, bert_name,
            lr=cfg['lr_mrpc'], grad_accum=cfg['grad_accum'])
        mrpc_res = {'baseline': [], 'simcse': [], 'jepa': []}
        for seed in SEEDS:
            print(f'  Seed {seed}')
            for mt in ('baseline', 'simcse', 'jepa'):
                r = run_experiment(
                    ldr['mrpc_train'], ldr['mrpc_val'], ldr['mrpc_test'],
                    mt, bert_name, seed, lr=cfg['lr_mrpc'],
                    aux_lambda=best_lam_mrpc if mt == 'jepa' else 0.1,
                    epochs=cfg['epochs_mrpc'], grad_accum=cfg['grad_accum'],
                    tag=f'{bname}/MRPC/{mt}', save_key=f'{bname}_mrpc_{mt}')
                mrpc_res[mt].append(r)
                print(f'    {mt:<10} f1={r["test"]["f1"]:.4f}')
            torch.cuda.empty_cache()
        results['mrpc'] = mrpc_res
        results['best_lam_mrpc'] = best_lam_mrpc
        _save_partial()

    # ── Low-Resource MRPC ─────────────────────────────────────────────────────
    if 'lr_mrpc' in results:
        print('\n-- Low-Resource MRPC  [SKIPPED — already in checkpoint]')
    elif skip_lr_mrpc:
        results['lr_mrpc'] = {f: {'baseline':[0.0],'simcse':[0.0],'jepa':[0.0]}
                                for f in fractions}
    else:
        print('\n-- Low-Resource MRPC')
        lr_res = {f: {} for f in fractions}
        for frac in fractions:
            n = max(1, int(len(ds['mrpc_train']) * frac))
            g = torch.Generator().manual_seed(42)
            idx = torch.randperm(len(ds['mrpc_train']), generator=g)[:n].tolist()
            sub = DataLoader(Subset(ds['mrpc_train'], idx),
                             cfg['batch_size'], shuffle=True, collate_fn=col, num_workers=0)
            print(f'  {int(frac * 100)}% ({n} samples)')
            for mt in ('baseline', 'simcse', 'jepa'):
                f1s = []
                for s in SEEDS:
                    try:
                        r = run_experiment(
                            sub, ldr['mrpc_val'], ldr['mrpc_test'],
                            mt, bert_name, s, lr=cfg['lr_mrpc'],
                            aux_lambda=results['best_lam_mrpc'] if mt == 'jepa' else 0.1,
                            epochs=cfg['epochs_mrpc'], grad_accum=cfg['grad_accum'],
                            tag=f'{bname}/LR{int(frac*100)}/{mt}/s{s}')
                        f1s.append(r['test']['f1'])
                    except Exception as e:
                        print(f'      [WARN] seed={s} {mt} failed: {e}')
                        f1s.append(float('nan'))
                lr_res[frac][mt] = f1s
                valid = [v for v in f1s if v == v]
                mu = np.mean(valid) if valid else float('nan')
                sd = np.std(valid)  if valid else float('nan')
                print(f'    {mt:<10} {mu:.4f}+/-{sd:.4f}')
            torch.cuda.empty_cache()
        results['lr_mrpc'] = lr_res
        _save_partial()

    # ── QQP ───────────────────────────────────────────────────────────────────
    if 'qqp' in results:
        print(f'\n-- QQP  [SKIPPED — already in checkpoint]')
    else:
        print(f'\n-- QQP  (hybrid JEPA+SimCSE, epochs={cfg["epochs_qqp"]})')
        if qqp_train_max:
            _qqp_idx = torch.randperm(len(ds['qqp_train']),
                                      generator=torch.Generator().manual_seed(42))[:qqp_train_max].tolist()
            _qqp_sub = DataLoader(Subset(ds['qqp_train'], _qqp_idx),
                                  cfg['batch_size'], shuffle=True, collate_fn=col, num_workers=0)
            print(f'  QQP capped at {qqp_train_max:,} samples for this backbone')
        else:
            _qqp_sub = ldr['qqp_train']
        best_lam_qqp = fixed_lambda or select_lambda(
            ds['qqp_train'], col, bert_name, lr=cfg['lr_qqp'],
            grad_accum=cfg['grad_accum'], model_type='hybrid')
        qqp_res = {'baseline': [], 'simcse': [], 'hybrid': []}
        for seed in SEEDS:
            print(f'  Seed {seed}')
            for mt in ('baseline', 'simcse', 'hybrid'):
                r = run_experiment(
                    _qqp_sub, ldr['qqp_val'], ldr['qqp_test'],
                    mt, bert_name, seed, lr=cfg['lr_qqp'],
                    aux_lambda=best_lam_qqp if mt == 'hybrid' else 0.1,
                    epochs=cfg['epochs_qqp'], grad_accum=cfg['grad_accum'],
                    tag=f'{bname}/QQP/{mt}', save_key=f'{bname}_qqp_{mt}')
                qqp_res[mt].append(r)
                print(f'    {mt:<10} f1={r["test"]["f1"]:.4f}')
            torch.cuda.empty_cache()
        results['qqp'] = qqp_res
        results['best_lam_qqp'] = best_lam_qqp
        _save_partial()

    # ── PAWS ──────────────────────────────────────────────────────────────────
    if 'paws' in results:
        print(f'\n-- PAWS  [SKIPPED — already in checkpoint]')
    else:
        print(f'\n-- PAWS  (adversarial, epochs={cfg["epochs_paws"]})')
        best_lam_paws = fixed_lambda or select_lambda(
            ds['paws_train'], col, bert_name,
            lr=cfg['lr_paws'], grad_accum=cfg['grad_accum'])
        paws_res = {'baseline': [], 'simcse': [], 'jepa': []}
        for seed in SEEDS:
            print(f'  Seed {seed}')
            for mt in ('baseline', 'simcse', 'jepa'):
                r = run_experiment(
                    ldr['paws_train'], ldr['paws_val'], ldr['paws_test'],
                    mt, bert_name, seed, lr=cfg['lr_paws'],
                    aux_lambda=best_lam_paws if mt == 'jepa' else 0.1,
                    epochs=cfg['epochs_paws'], grad_accum=cfg['grad_accum'],
                    tag=f'{bname}/PAWS/{mt}', save_key=f'{bname}_paws_{mt}')
                paws_res[mt].append(r)
                print(f'    {mt:<10} f1={r["test"]["f1"]:.4f}')
            torch.cuda.empty_cache()
        results['paws'] = paws_res
        results['best_lam_paws'] = best_lam_paws
        _save_partial()

    print(f'\n{chr(61)*55}  SUMMARY: {bname}')
    for dname, res in [('MRPC',results['mrpc']),('QQP',results['qqp']),('PAWS',results['paws'])]:
        for mt, runs in res.items():
            f1s = [r['test']['f1'] for r in runs]
            print(f'  {dname:<10} {mt:<12} {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}')
    return results

print('run_backbone() ready -- MRPC + QQP + PAWS  [crash-safe, per-dataset checkpointing]')

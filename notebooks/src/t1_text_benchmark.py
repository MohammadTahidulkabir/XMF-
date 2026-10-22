# %% [meta]
# {"slug": "asd-xmf-t1-text-benchmark", "title": "ASD-XMF T1 Text Benchmark",
#  "gpu": true, "internet": true, "datasets": [], "kernels": ["asd-xmf-d4-text-dataset"], "phase": "A"}

# %% [markdown]
# # T1 - Text sector: proposed XMF-Text vs baselines (ASD vs non-ASD statements) + questionnaire leakage audit
# **Proposed (plan 5.1 text branch):** XMF-Text = multilingual XLM-RoBERTa-base encoder (ready for Bangla text)
# -> 256-d token projection -> attention pooling -> classifier.
# **Baselines:** TF-IDF + LR, TF-IDF + SVM, BiLSTM, DistilBERT, BERT-base, RoBERTa-base.
# Repeated stratified 5-fold CV (same folds for all), inner validation split for early stopping,
# train vs validation accuracy reported for every model.

# %%
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import asd_models as M, asd_bench as B
SRC = U.find_path("text.csv").parent
OUT = U.WORK / "t1"; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SRC / "text.csv"); X, y = df.text.values, df.label.values
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
if PREFLIGHT:
    FOLDS = FOLDS[:2]
dev = M.device()

class AttnPoolCls(nn.Module):
    def __init__(self, enc, hid, proposed):
        super().__init__()
        self.enc, self.proposed = enc, proposed
        self.proj = nn.Sequential(nn.Linear(hid, 256), nn.GELU(), nn.LayerNorm(256)) if proposed else nn.Identity()
        d = 256 if proposed else hid
        self.att = nn.Linear(d, 1) if proposed else None
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 2))

    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        if self.proposed:
            h = self.proj(h)
            a = self.att(h).squeeze(-1).masked_fill(am == 0, -1e4).softmax(-1)
            return self.fc((a.unsqueeze(-1) * h).sum(1))
        return self.fc(h[:, 0])

def run_transformer(name, ckpt, proposed=False):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    enc = tok(list(X), truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    pred, trs, vas = np.full(len(y), -1), [], []
    for f, (tr, te) in enumerate(FOLDS):
        U.seed_everything(f)
        rng = np.random.default_rng(f); va = rng.choice(tr, len(tr) // 8, replace=False); fit = np.setdiff1d(tr, va)
        base = AutoModel.from_pretrained(ckpt)
        model = AttnPoolCls(base, base.config.hidden_size, proposed).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scaler = M.make_scaler(); best, best_state, best_ep = -1, None, 0
        Y = torch.tensor(y, device=dev)
        pr = lambda idx: torch.cat([model(ids[idx[s:s + 64]], am[idx[s:s + 64]]).argmax(1) for s in range(0, len(idx), 64)]).cpu().numpy()
        for ep in range(1 if PREFLIGHT else 12):
            model.train()
            for s in range(0, len(fit), 16):
                i = torch.tensor(fit[rng.permutation(len(fit))][s:s + 16], device=dev)
                with M.amp_ctx():
                    loss = F.cross_entropy(model(ids[i], am[i]).float(), Y[i], label_smoothing=0.1)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            model.eval()
            with torch.no_grad(), M.amp_ctx():
                acc_va = (pr(va) == y[va]).mean()
            if acc_va > best:
                best, best_ep, best_state = acc_va, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= 3:
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(), M.amp_ctx():
            pred[te] = pr(te); trs.append((pr(fit) == y[fit]).mean()); vas.append(best)
        del model, base; torch.cuda.empty_cache()
    return pred, trs, vas

def run_tfidf(clf):
    pred, trs = np.full(len(y), -1), []
    for tr, te in FOLDS:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True); A = v.fit_transform(X[tr])
        c = clf().fit(A, y[tr]); pred[te] = c.predict(v.transform(X[te])); trs.append((c.predict(A) == y[tr]).mean())
    return pred, trs, [np.nan] * len(trs)

class BiLSTMText(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0); self.lstm = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * d, 2))
    def forward(self, ids):
        h, _ = self.lstm(self.emb(ids)); m = (ids > 0).float().unsqueeze(-1)
        return self.fc((h * m).sum(1) / m.sum(1).clamp_min(1))

def run_bilstm():
    from collections import Counter
    pred, trs, vas = np.full(len(y), -1), [], []
    for f, (tr, te) in enumerate(FOLDS):
        rng = np.random.default_rng(f); va = rng.choice(tr, len(tr) // 8, replace=False); fit = np.setdiff1d(tr, va)
        cnt = Counter(w for s in X[fit] for w in s.lower().split()); vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(20000))}
        enc = lambda s: ([vocab.get(w, 1) for w in s.lower().split()][:64] + [0] * 64)[:64]
        I = torch.tensor([enc(s) for s in X], device=dev); Y = torch.tensor(y, device=dev)
        net = BiLSTMText(len(vocab) + 2).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
        best, best_state, best_ep = -1, None, 0
        for ep in range(2 if PREFLIGHT else 60):
            net.train()
            for s in range(0, len(fit), 32):
                i = torch.tensor(fit[rng.permutation(len(fit))][s:s + 32], device=dev)
                loss = F.cross_entropy(net(I[i]), Y[i], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                a = (net(I[va]).argmax(1).cpu().numpy() == y[va]).mean()
            if a > best:
                best, best_ep, best_state = a, ep, {k: v.clone() for k, v in net.state_dict().items()}
            elif ep - best_ep >= 8:
                break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            pred[te] = net(I[te]).argmax(1).cpu().numpy(); trs.append((net(I[fit]).argmax(1).cpu().numpy() == y[fit]).mean()); vas.append(best)
    return pred, trs, vas

# %%
MODELS = {"tfidf_lr": lambda: run_tfidf(lambda: LogisticRegression(C=4.0, max_iter=3000)),
          "tfidf_svm": lambda: run_tfidf(lambda: LinearSVC(C=0.5)),
          "bilstm": run_bilstm,
          "distilbert": lambda: run_transformer("distilbert", "distilbert-base-uncased"),
          "bert_base": lambda: run_transformer("bert_base", "bert-base-uncased"),
          "roberta_base": lambda: run_transformer("roberta_base", "roberta-base"),
          "xmf_text": lambda: run_transformer("xmf_text", "xlm-roberta-base", proposed=True)}
if PREFLIGHT:
    MODELS = {k: MODELS[k] for k in ("tfidf_lr", "bilstm", "xmf_text")}
rows = []
for name, fn in MODELS.items():
    with U.Timer(name):
        p, trs, vas = fn()
    m = B.cls_metrics(y[p >= 0], p[p >= 0], 2)
    lo, hi = B.boot_ci(y[p >= 0], p[p >= 0], lambda a, b: float((a == b).mean()))
    rows.append({"task": "tasd_asd_vs_nonasd", "model": name, "proposed": name == "xmf_text", "test_acc": m["acc"], "ci_lo": lo, "ci_hi": hi,
                 "test_macro_f1": m["macro_f1"], "train_acc": float(np.mean(trs)), "val_acc": float(np.nanmean(vas)) if np.isfinite(vas).any() else np.nan,
                 "gap": float(np.mean(trs) - m["acc"])})
    B._save(OUT, f"tasd__{name}", rows[-1] | {"cm": m["cm"]}, [])
    print(rows[-1], flush=True)
tab = B.comparison_table(pd.DataFrame(rows))
print(tab.to_markdown(index=False, floatfmt=".3f"))
tab.to_csv(OUT / "text_comparison.csv", index=False)

# %% [markdown]
# ## Questionnaire leakage audit (plan 4.2: label is derived from the item scores)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
q = pd.read_csv(SRC / "quest.csv")
audit = []
for src, g in q.groupby("source"):
    items = g[[f"A{i}" for i in range(1, 11)]].values
    score = items.sum(1)
    best_thr = max(range(11), key=lambda t: ((score > t) == g.label.values).mean())
    rule_acc = ((score > best_thr) == g.label.values).mean()
    rf = cross_val_predict(RandomForestClassifier(300, random_state=0), items, g.label.values, cv=5)
    audit.append({"source": src, "n": len(g), "rule": f"sum(A1..A10) > {best_thr}", "rule_accuracy": rule_acc,
                  "random_forest_cv_accuracy": (rf == g.label.values).mean()})
audit = pd.DataFrame(audit)
print(audit.to_markdown(index=False, floatfmt=".4f"))
print("A rule accuracy of ~1.00 means the label is a deterministic function of the items: these sets cannot "
      "support an ASD-screening claim (plan 4.2) and are reported only as a leakage check.")
audit.to_csv(OUT / "questionnaire_leakage_audit.csv", index=False)

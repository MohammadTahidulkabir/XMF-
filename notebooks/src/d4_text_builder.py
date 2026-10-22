# %% [meta]
# {"slug": "asd-xmf-d4-text-dataset", "title": "ASD-XMF D4 Text Dataset",
#  "gpu": false, "internet": true,
#  "datasets": ["sharafatahmed/tasd-dataset-text-based-early-asd-detection", "fabdelja/autism-screening-for-toddlers"],
#  "kernels": [], "phase": "D"}

# %% [markdown]
# # D4 - XMF-Text: free text + questionnaire items
# * `text.csv` - TASD: statements about a child's behaviour, labelled ASD / non-ASD (plan 5.1: free-text parent
#   concerns; BanglaBERT once Bangla text is collected).
# * `quest.csv` - Q-CHAT-10 toddlers and AQ-10 screening answers.  Plan 4.2 warns that their label is derived
#   from the item scores, so they are used only for a **leakage audit**, never as a headline result.
# Exact duplicate rows are removed.

# %%
import numpy as np, pandas as pd
OUT = U.WORK / "xmf_text"; OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(U.find_path("Dataset-v2.csv"))
text = pd.DataFrame({"text": t.Text.astype(str).str.strip(), "label": (t.ASD.str.strip() == "ASD").astype(int),
                     "behaviour": t.Class, "source": "tasd"}).drop_duplicates("text").reset_index(drop=True)
text.to_csv(OUT / "text.csv", index=False)
q1 = pd.read_csv(U.find_path("Toddler Autism dataset July 2018.csv"))
lab = [c for c in q1.columns if c.strip().startswith("Class")][0]
quest = [pd.DataFrame({**{f"A{i}": q1[f"A{i}"] for i in range(1, 11)}, "age": q1.Age_Mons / 12.0, "sex": (q1.Sex == "m").astype(int),
                       "label": (q1[lab].str.strip().str.lower() == "yes").astype(int), "source": "qchat10_toddler"})]
comb = U.find_path("Autism_Screening_Data_Combined.csv")
if comb is not None:
    q2 = pd.read_csv(comb)
    quest.append(pd.DataFrame({**{f"A{i}": q2[f"A{i}"] for i in range(1, 11)}, "age": q2.Age, "sex": (q2.Sex == "m").astype(int),
                               "label": (q2.Class.str.strip().str.upper() == "YES").astype(int), "source": "aq10_combined"}))
quest = pd.concat(quest, ignore_index=True).drop_duplicates().reset_index(drop=True)
quest.to_csv(OUT / "quest.csv", index=False)
print(text.label.value_counts().to_dict(), "| quest:", quest.groupby("source").label.agg(["size", "mean"]))
(OUT / "DATASHEET.md").write_text(f"""# XMF-Text
text.csv: {len(text)} TASD statements (ASD {int(text.label.sum())}, non-ASD {int((1 - text.label).sum())}).
quest.csv: {len(quest)} questionnaire rows ({quest.source.value_counts().to_dict()}); label derived from item sum -> leakage audit only.
""")

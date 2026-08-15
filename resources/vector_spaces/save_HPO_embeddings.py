import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import pronto
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Load ONNX model
model_path = "./resources/sapbert_onnx"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForFeatureExtraction.from_pretrained(model_path)

# Load HPO
hpo = pronto.Ontology("http://purl.obolibrary.org/obo/hp.owl")


#load snomed ct descriptions
desc_path = Path(
    r"resources\SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z\Full\Terminology\sct2_Description_Full-en_INT_20260301.txt"
)
df_desc = pd.read_csv(
    desc_path,
    sep="\t",
    dtype=str,          # keep SNOMED IDs as strings
    encoding="utf-8"
)
# Keep only active descriptions
df_desc = df_desc[df_desc["active"] == "1"]
df_desc = df_desc[df_desc["typeId"].isin(["900000000000003001",  "900000000000013009" ])] #keep only FSN and synonyms
#load SNOMED ct concepts
concept_path = Path(
    r"resources\SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z\Full\Terminology\sct2_Concept_Full_INT_20260301.txt"
)
df_concept = pd.read_csv(
    concept_path,
    sep="\t",
    dtype=str,          # keep SNOMED IDs as strings
    encoding="utf-8"
)
df_concept = df_concept[df_concept["active"] == "1"]
valid_concept_ids = set(df_concept["id"].dropna().astype(str))
#load snomed ct def
textdef_path = Path(
    r"resources\SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z\Full\Terminology\sct2_TextDefinition_Full-en_INT_20260301.txt"
)
df_textdef = pd.read_csv(textdef_path, sep="\t", dtype=str)
df_textdef = df_textdef[df_textdef["active"] == "1"]
#group by concept ID
desc_grouped = df_desc.groupby("conceptId")["term"].apply(list).to_dict()
textdef_grouped = df_textdef.groupby("conceptId")["term"].apply(list).to_dict()


term_texts = []
term_metadata = []

for term in hpo.terms():
    definition = term.definition if term.definition else None
    seen_texts = set()  # To track already added texts for this term
    # 1. Name only
    if "obsolete" in term.name.lower():
        continue
    else:
        term_texts.append(term.name.lower())
        term_metadata.append({
            "hpo_id": str(term.id),
            "hpo_term": term.name
        })
        seen_texts.add(term.name)
        #2a definition only
        if definition:
            term_texts.append(definition.lower())
            term_metadata.append({
                "hpo_id": str(term.id),
                "hpo_term": term.name
            })
            seen_texts.add(definition)

        #2 Name + definition 
        if definition:
            term_texts.append(f"{term.name.lower()}: {definition.lower()}")
            term_metadata.append({
                "hpo_id": str(term.id),
                "hpo_term": term.name
            })
            seen_texts.add(f"{term.name}: {definition}")

        
        

print(len(term_texts))

        

bs = 64
out_dir = BASE_DIR / "hpo_embed_chunks"
out_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = out_dir / "checkpoint.txt"

# load checkpoint if exists
start_idx = 0
if checkpoint_file.exists():
    start_idx = int(checkpoint_file.read_text().strip())
    print(f"Resuming from index {start_idx}")

for i in tqdm(range(start_idx, len(term_texts), bs)):
    batch = term_texts[i:i+bs]

    try:
        inputs = tokenizer(
            batch,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            c_rep = model(**inputs).last_hidden_state[:, 0, :]

        emb = c_rep.cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb = emb / norms   

        # save each batch immediately
        np.save(out_dir / f"emb_{i:08d}.npy", emb)

        # update checkpoint
        checkpoint_file.write_text(str(i + bs))

    except Exception as e:
        print(f"Error at batch {i}: {e}")
        print("Retrying in 10 seconds...")
        import time
        time.sleep(10)
        continue

out_dir = BASE_DIR / "hpo_embed_chunks"

files = sorted(out_dir.glob("emb_*.npy"))

all_embs = np.concatenate([np.load(f) for f in files], axis=0)

all_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)

# save vector spaces and metadata locally
np.save("hpo_embeddings_N+D.npy", all_embs)
np.save("hpo_term_ID_names_N+D.npy", term_metadata)






from pathlib import Path
import pandas as pd
import pronto
import json
import spacy

def download_hpo():
    hpo = pronto.Ontology("http://purl.obolibrary.org/obo/hp.owl")
    return hpo

    
# Build hashmap for HPO terms with their metadata, parents, children, grandparents, grandchildren, and siblings
def build_hashmap(hpo):

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
    df_desc = df_desc[df_desc["typeId"] == "900000000000013009"] #keep only synonyms
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



    hashmap = {}
    sorted_hpo_lookup = {}
    i=0
    for term in hpo.terms():
        if "obsolete" in term.name.lower():
            continue
        elif "HP:" in term.id:

            #Add SNOMEDCT 
            seen_texts = set()
            snomed_terms = []
            for xref in (term.xrefs or []):
                xref_str = str(xref)
                if "SNOMEDCT_US:" not in xref_str:
                    continue

                snomed_id = xref_str.split(":", 1)[1].strip("')")

                # Skip if SNOMED concept is not present in Concept file
                if snomed_id not in valid_concept_ids:
                    continue
                
                for term_snomed in desc_grouped.get(snomed_id, []):
                    if term_snomed not in seen_texts and term_snomed != term.name and term_snomed not in term.synonyms:
                        snomed_terms.append(term_snomed)
                        seen_texts.add(term_snomed)
                        


            i += 1
            if i % 1000 == 0:
                print(f"Processing term {i}")
            parents = set(p for p in term.superclasses(distance=1) if p.id != term.id)
            ancestors_2 = set(p for p in term.superclasses(distance=2) if p.id != term.id and p not in parents)

            children = set(c for c in term.subclasses(distance=1) if c.id != term.id)
            descendants_2 = set(c for c in term.subclasses(distance=2) if c.id != term.id and c not in children)

            hashmap[term.name.lower()] = {
                'phenotype': term.id,
                'definition': term.definition.lower() if term.definition else "",
                'snomed_terms': snomed_terms,
                'parent': [parent.id for parent in parents],
                'children': [child.id for child in children],
                'grandparent': [grandparent.id for grandparent in ancestors_2],
                'grandchildren': [grandchild.id for grandchild in descendants_2],
                'siblings': [sibling.id for parent in parents for sibling in parent.subclasses(distance=1) if parent.id != term.id and sibling.id != term.id]
            }
            sorted_term = " ".join(sorted(term.name.lower().split()))
            sorted_hpo_lookup[sorted_term] = term.id
            for synonym in term.synonyms:
                hashmap[synonym.description.lower()] = {
                    'phenotype': term.id,
                    'definition': term.definition.lower() if term.definition else "",
                    'snomed_terms': snomed_terms,
                    'parent': [parent.id for parent in parents],
                    'children': [child.id for child in children],
                    'grandparent': [grandparent.id for grandparent in ancestors_2],
                    'grandchildren': [grandchild.id for grandchild in descendants_2],
                    'siblings': [sibling.id for parent in parents for sibling in parent.subclasses(distance=1) if parent.id != term.id and sibling.id != term.id]
                }
                sorted_term = " ".join(sorted(synonym.description.lower().split()))
                sorted_hpo_lookup[sorted_term] = term.id
            
    return hashmap, sorted_hpo_lookup

hpo = download_hpo()
print("HPO downloaded successfully.")


hashmap, sorted_hpo_lookup = build_hashmap(hpo)

with open("resources/HPO/hpo_lookup.json", "w", encoding="utf-8") as f:
    json.dump(hashmap, f, ensure_ascii=False, indent=2)

with open("resources/HPO/sorted_hpo_lookup.json", "w", encoding="utf-8") as f:
    json.dump(sorted_hpo_lookup, f, ensure_ascii=False, indent=2)
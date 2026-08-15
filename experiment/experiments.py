import sys
import csv
import json
from pathlib import Path
from time import time
import pronto


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Pipeline.pipeline import run_pipeline


def save_explanation_files(explanation, output_dir, filename_prefix="explanation"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{filename_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)

    return json_path#, csv_path, txt_path

def parse_hpo_tsv(filepath):
    results = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    
    i = 0
    n = len(lines)
    
    while i < n:
        
        # Skip empty lines
        if lines[i].strip() == "":
            i += 1
            continue
        
        # Document ID
        doc_id = lines[i].strip()
        i += 1
        
        # Full text
        text = lines[i].strip()
        i += 1
        
        phenotypes = []
        annotations = []
        
        # Read annotation lines until blank line or next doc
        while i < n and lines[i].strip() != "":
            
            parts = lines[i].split("\t") 
            
            if len(parts) >= 4:
                hpo_id = [parts[3].strip()]
                if hpo_id not in phenotypes:
                    phenotypes.append(hpo_id)
                    annotations.append({
                        "hpo_id": hpo_id,
                        "symptom_text": parts[2].strip(),
                    })
            elif len(parts) == 2:
                hpo_id = parts[1].strip().split(",")

                phenotypes.append(hpo_id)
                annotations.append({
                    "hpo_id": hpo_id,
                    "symptom_text": parts[0].strip(),
                })
            i += 1

        results.append({
            "id": doc_id,
            "text": text,
            "annotations": annotations,
        })
        
        i += 1  # skip blank line
    
    return results

#data = parse_hpo_tsv("C:\\Users\\micha\\OneDrive - Maastricht University\\Documents\\Thesis\\experiment\\Dataset\\corpus\\CSC\\datasetCSC.tsv")
data = parse_hpo_tsv("C:\\Users\\micha\\OneDrive - Maastricht University\\Documents\\Thesis\\experiment\\Dataset\\corpus\\GSC\\GSC_gold_merged_filtered.tsv")
hpo = pronto.Ontology("http://purl.obolibrary.org/obo/hp.owl")
with open("resources/HPO/hpo_lookup.json", "r", encoding="utf-8") as f:
    hpo_lookup = json.load(f)

explanation_unfound_fam = []
explanation_rank1_fam = []
explanation_fam = []

explanation = []
hits_top_1 = 0
hits_top_3 = 0
hits_top_5 = 0
hits_top_10 = 0
misses_top_1=0
misses_top_3=0
misses_top_5=0
misses_top_10=0
reciprocal_rank_sum = 0

hits_top_1_fam = 0
hits_top_3_fam = 0
hits_top_5_fam = 0
hits_top_10_fam = 0
misses_top_1_fam=0
misses_top_3_fam=0
misses_top_5_fam=0
misses_top_10_fam=0
total=0
total_extracted = 0
total_time = 0
reciprocal_rank_sum_fam = 0
for gold_set in data:
    start_time = time()
    pheno_found = run_pipeline(gold_set["text"], 10)
    end_time = time()
    elapsed = end_time - start_time
    total_time += elapsed
    total_extracted += len(pheno_found)
    #print(pheno_found)
    for annotation in gold_set["annotations"]:
        gold_item = annotation["hpo_id"]
        symptom_text = annotation["symptom_text"]
        in_top_1_candidates_fam = False
        in_top_3_candidates_fam = False
        in_top_5_candidates_fam = False
        in_top_10_candidates_fam = False
        best_rank_fam = None
        best_candidate_fam = None
        best_top_k_cand_explained_fam = []
        total_top_k_cand_explained_fam = []
        extracted_text = ""

        in_top_1_candidates = False
        in_top_3_candidates = False
        in_top_5_candidates = False
        in_top_10_candidates = False
        best_rank = 100

        for phen_found in pheno_found:

            if phen_found["hpo_id"] in gold_item:
                in_top_1_candidates = True
                best_rank=1
                

            for index, candidate in enumerate(phen_found.get("top_k_candidates", [])):
                if candidate["hpo_id"] in gold_item :
                    current_rank = index + 1
                    if current_rank < best_rank:
                        best_rank = current_rank


            
            if phen_found["hpo_id"] in gold_item or any(gold_id in phen_found.get("parent_ids", []) for gold_id in gold_item) or any(gold_id in phen_found.get("children_ids", []) for gold_id in gold_item):
                in_top_1_candidates_fam = True
                best_rank_fam=1
                best_candidate_fam = phen_found
                best_candidate_fam["hpo_term"] =""
                extracted_text = phen_found["text"]
                best_top_k_cand_explained_fam.append({
                    "phenotype_text": phen_found["text"],
                    "candidate_id": f"{phen_found['hpo_id']}",
                    "rank": 1,
                    "relationship": "exact" if phen_found["hpo_id"] in gold_item else "parent" if any(gold_id in phen_found.get("parent_ids", []) for gold_id in gold_item) else "child" if any(gold_id in phen_found.get("children_ids", []) for gold_id in gold_item) else None
                })
                

            top_k_cand_explained = []
            for index, candidate in enumerate(phen_found.get("top_k_candidates", [])):
                info = hpo_lookup.get(candidate["hpo_term"], {})
                total_top_k_cand_explained_fam.append({
                    "phenotype_text": phen_found["text"],
                    "candidate_id": f"{candidate['hpo_id']}: {candidate['hpo_term']}",
                    "rank": index + 1,
                    "relationship": "exact" if candidate["hpo_id"] in gold_item else "parent" if any(gold_id in candidate.get("parent_ids", []) for gold_id in gold_item) else "child" if any(gold_id in candidate.get("children_ids", []) for gold_id in gold_item) else "grandparent" if any(gold_id in info.get("grandparent", []) for gold_id in gold_item) else "grandchild" if any(gold_id in info.get("grandchildren", []) for gold_id in gold_item) else "sibling" if any(gold_id in info.get("siblings", []) for gold_id in gold_item) else None
                })
                if candidate["hpo_id"] in gold_item :
                    current_rank = index + 1
                    if best_rank_fam is None or current_rank < best_rank_fam:# or (best_candidate_fam and best_candidate_fam['hpo_id'] != gold_item):
                        best_rank_fam = current_rank
                        best_candidate_fam = candidate
                        best_top_k_cand_explained_fam = top_k_cand_explained
                        extracted_text = phen_found["text"]
                elif (any(gold_id in candidate.get("parent_ids", []) for gold_id in gold_item) or any(gold_id in candidate.get("children_ids", []) for gold_id in gold_item)):# and ((best_candidate_fam and best_candidate_fam['hpo_id'] != gold_item) or best_candidate_fam is None):
                    current_rank = index + 1
                    if best_rank_fam is None or current_rank < best_rank_fam:
                        best_rank_fam = current_rank
                        best_candidate_fam = candidate
                        best_top_k_cand_explained_fam = top_k_cand_explained
                        extracted_text = phen_found["text"]
                info = hpo_lookup.get(candidate["hpo_term"], {})
                top_k_cand_explained.append({
                    "phenotype_text": phen_found["text"],
                    "candidate_id": f"{candidate['hpo_id']}: {candidate['hpo_term']}",
                    "rank": index + 1,
                    "relationship": "exact" if candidate["hpo_id"] in gold_item else "parent" if any(gold_id in candidate.get("parent_ids", []) for gold_id in gold_item) else "child" if any(gold_id in candidate.get("children_ids", []) for gold_id in gold_item) else "grandparent" if any(gold_id in info.get("grandparent", []) for gold_id in gold_item) else "grandchild" if any(gold_id in info.get("grandchildren", []) for gold_id in gold_item) else "sibling" if any(gold_id in info.get("siblings", []) for gold_id in gold_item) else None
                })

        if best_rank < 100:
            reciprocal_rank_sum += 1 / best_rank
            if best_rank <= 10:  # Check if it's in the top 10 candidates
                in_top_10_candidates = True
            if best_rank <= 5:  # Check if it's in the top 5 candidates
                in_top_5_candidates = True
            if best_rank <= 3:  # Check if it's in the top 3 candidates
                in_top_3_candidates = True
            if best_rank <= 1:  # Check if it's in the top 1 candidate
                in_top_1_candidates = True
        if in_top_1_candidates:
            hits_top_1 += 1
            hits_top_3 += 1
            hits_top_5 += 1
            hits_top_10 += 1 
        elif in_top_3_candidates:
            hits_top_3 += 1
            hits_top_5 += 1
            hits_top_10 += 1
            misses_top_1 += 1   
        elif in_top_5_candidates:
            hits_top_5 += 1
            hits_top_10 += 1
            misses_top_1 += 1
            misses_top_3 += 1 
        elif in_top_10_candidates:
            hits_top_10 += 1
            misses_top_1 += 1
            misses_top_3 += 1
            misses_top_5 += 1
        else:
            misses_top_10 += 1
            misses_top_1 += 1
            misses_top_3 += 1
            misses_top_5 += 1



        if best_rank_fam is not None:
            reciprocal_rank_sum_fam += 1 / best_rank_fam
            if best_rank_fam <= 10:  # Check if it's in the top 10 candidates
                in_top_10_candidates_fam = True
            if best_rank_fam <= 5:  # Check if it's in the top 5 candidates
                in_top_5_candidates_fam = True
            if best_rank_fam <= 3:  # Check if it's in the top 3 candidates
                in_top_3_candidates_fam = True
            if best_rank_fam <= 1:  # Check if it's in the top 1 candidate
                in_top_1_candidates_fam = True
            if len(gold_item) == 1:
                gold_truth = f"{gold_item[0]}: {hpo[gold_item[0]].name if gold_item[0] in hpo else None}"
            elif len(gold_item) > 1:
                gold_truth = f"{gold_item[0]}: {hpo[gold_item[0]].name if gold_item[0] in hpo else None}, {gold_item[1]}: {hpo[gold_item[1]].name if gold_item[1] in hpo else None}"
            if best_rank_fam != 1 or best_candidate_fam['hpo_id'] not in gold_item:
                explanation_fam.append({
                    "doc_id": gold_set["id"],
                    "gold_truth": gold_truth,
                    "text": symptom_text,
                    "predicted": f"{best_candidate_fam['hpo_id']}: {best_candidate_fam['hpo_term']}" if best_candidate_fam else None,
                    "extracted_text" : extracted_text,
                    "rank": best_rank_fam,
                    "relationship": "exact" if best_candidate_fam['hpo_id'] in gold_item else "parent" if any(gold_id in best_candidate_fam.get("parent_ids", []) for gold_id in gold_item) else "child" if any(gold_id in best_candidate_fam.get("children_ids", []) for gold_id in gold_item) else None,
                    "higher_candidates": best_top_k_cand_explained_fam[:best_rank_fam-1] if best_top_k_cand_explained_fam else [],
                })
            if best_top_k_cand_explained_fam[0]['relationship'] in ["parent", "child", "sibling"]:
                explanation_rank1_fam.append({
                    "doc_id": gold_set["id"],
                    "gold_truth": gold_truth,
                    "text": symptom_text,
                    "predicted": f"{best_candidate_fam['hpo_id']}: {best_candidate_fam['hpo_term']}" if best_candidate_fam else None,
                    "extracted_text" : extracted_text,
                    "rank": best_rank_fam,
                    "relationship": "exact" if best_candidate_fam['hpo_id'] in gold_item else "parent" if any(gold_id in best_candidate_fam.get("parent_ids", []) for gold_id in gold_item) else "child" if any(gold_id in best_candidate_fam.get("children_ids", []) for gold_id in gold_item) else None,
                    "higher_candidates": best_top_k_cand_explained_fam[:best_rank_fam-1] if best_top_k_cand_explained_fam else [],
                })
        if best_candidate_fam is None:
            correct_can = []
            for top_k in total_top_k_cand_explained_fam:
                if top_k["phenotype_text"].lower() == symptom_text.lower() or top_k["phenotype_text"].lower() in symptom_text.lower() or symptom_text.lower() in top_k["phenotype_text"].lower():
                    correct_can.append(top_k)
                if len(correct_can) >= 10:
                    break

            if len(correct_can) == 0:
                for phen in pheno_found:
                    if phen['text'] not in correct_can:
                        correct_can.append(phen['text'])
            if len(gold_item) == 1:
                gold_truth = f"{gold_item[0]}: {hpo[gold_item[0]].name if gold_item[0] in hpo else None}"
            elif len(gold_item) > 1:
                gold_truth = f"{gold_item[0]}: {hpo[gold_item[0]].name if gold_item[0] in hpo else None}, {gold_item[1]}: {hpo[gold_item[1]].name if gold_item[1] in hpo else None}"
            explanation_unfound_fam.append({
                "doc_id": gold_set["id"],
                "gold_truth": gold_truth,
                "text": symptom_text,
                "predicted": None,
                "rank": None,
                "relationship": "none",
                "candidates": correct_can if len(correct_can) == 10 else f"Not extracted correctly - {correct_can}"
            })
            explanation_fam.append({
                "doc_id": gold_set["id"],
                "gold_truth": gold_truth,
                "text": symptom_text,
                "predicted": None,
                "rank": None,
                "relationship": "none",
                "candidates": correct_can if len(correct_can) == 10 else f"Not extracted correctly - {correct_can}"
            })
        if in_top_1_candidates_fam:
            hits_top_1_fam += 1
            hits_top_3_fam += 1
            hits_top_5_fam += 1
            hits_top_10_fam += 1
        elif in_top_3_candidates_fam:
            hits_top_3_fam += 1
            hits_top_5_fam += 1
            hits_top_10_fam += 1
            misses_top_1_fam += 1
        elif in_top_5_candidates_fam:
            hits_top_5_fam += 1
            hits_top_10_fam += 1
            misses_top_1_fam += 1
            misses_top_3_fam += 1
        elif in_top_10_candidates_fam:
            hits_top_10_fam += 1
            misses_top_1_fam += 1
            misses_top_3_fam += 1
            misses_top_5_fam += 1
        else:
            misses_top_10_fam += 1
            misses_top_1_fam += 1
            misses_top_3_fam += 1
            misses_top_5_fam += 1
    num_phenotypes = len(gold_set["annotations"])
    total += num_phenotypes
    print(f"Processed document {gold_set['id']}")
    print(f"Recall@10: {hits_top_10_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_10_fam}, Misses: {misses_top_10_fam}, Total: {total}, Recall: 0.0", f"Recall@5: {hits_top_5_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_5_fam}, Misses: {misses_top_5_fam}, Total: {total}, Recall: 0.0", f"Recall@3: {hits_top_3_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_3_fam}, Misses: {misses_top_3_fam}, Total: {total}, Recall: 0.0", f"Recall@1: {hits_top_1_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_1_fam}, Misses: {misses_top_1_fam}, Total: {total}, Recall: 0.0")
print(f"Total time: {total_time:.2f} seconds, Average time per document: {total_time/len(data):.2f} seconds")
json_path = save_explanation_files(explanation_unfound_fam, Path(__file__).resolve().parent, filename_prefix="explanation_unfound_CSC")
json_path_2 = save_explanation_files(explanation_rank1_fam, Path(__file__).resolve().parent, filename_prefix="explanation_rank1_CSC")
json_path_3 = save_explanation_files(explanation_fam, Path(__file__).resolve().parent, filename_prefix="explanation_all_CSC")
print(f"Saved explanation files to: {json_path}")
print(f"Saved explanation files to: {json_path_2}")
print(f"Saved explanation files to: {json_path_3}")
print("Top 10:")
print(f"Hits: {hits_top_10_fam}, Misses: {misses_top_10_fam}, Total: {total}, Recall: {hits_top_10_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_10_fam}, Misses: {misses_top_10_fam}, Total: {total}, Recall: 0.0")
print("Top 5:")
print(f"Hits: {hits_top_5_fam}, Misses: {misses_top_5_fam}, Total: {total}, Recall: {hits_top_5_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_5_fam}, Misses: {misses_top_5_fam}, Total: {total}, Recall: 0.0")
print("Top 3:")
print(f"Hits: {hits_top_3_fam}, Misses: {misses_top_3_fam}, Total: {total}, Recall: {hits_top_3_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_3_fam}, Misses: {misses_top_3_fam}, Total: {total}, Recall: 0.0")
print("Top 1:")
print(f"Hits: {hits_top_1_fam}, Misses: {misses_top_1_fam}, Total: {total}, Recall: {hits_top_1_fam/total:.2f}" if total > 0 else f"Hits: {hits_top_1_fam}, Misses: {misses_top_1_fam}, Total: {total}, Recall: 0.0")
print(f"MRR: {reciprocal_rank_sum_fam/total:.2f}" if total > 0 else "No correct predictions to calculate mean rank.")
print(f"Precision@10: {hits_top_10_fam/total_extracted:.2f}" if total_extracted > 0 else f"Hits: {hits_top_10_fam}, Misses: {misses_top_10_fam}, Total Extracted: {total_extracted}, Precision: 0.0")
json_path = save_explanation_files(explanation, Path(__file__).resolve().parent, filename_prefix="explanation_CSC")
print(f"Saved explanation files to: {json_path}")
print("Top 10:")
print(f"Hits: {hits_top_10}, Misses: {misses_top_10}, Total: {total}, Recall: {hits_top_10/total:.2f}" if total > 0 else f"Hits: {hits_top_10}, Misses: {misses_top_10}, Total: {total}, Recall: 0.0")
print("Top 5:")
print(f"Hits: {hits_top_5}, Misses: {misses_top_5}, Total: {total}, Recall: {hits_top_5/total:.2f}" if total > 0 else f"Hits: {hits_top_5}, Misses: {misses_top_5}, Total: {total}, Recall: 0.0")
print("Top 3:")
print(f"Hits: {hits_top_3}, Misses: {misses_top_3}, Total: {total}, Recall: {hits_top_3/total:.2f}" if total > 0 else f"Hits: {hits_top_3}, Misses: {misses_top_3}, Total: {total}, Recall: 0.0")
print("Top 1:")
print(f"Hits: {hits_top_1}, Misses: {misses_top_1}, Total: {total}, Recall: {hits_top_1/total:.2f}" if total > 0 else f"Hits: {hits_top_1}, Misses: {misses_top_1}, Total: {total}, Recall: 0.0")
print(f"MRR: {reciprocal_rank_sum/total:.2f}" if total > 0 else "No correct predictions to calculate mean rank.")
print(f"Precision@10: {hits_top_10/total_extracted:.2f}" if total_extracted > 0 else f"Hits: {hits_top_10}, Misses: {misses_top_10}, Total Extracted: {total_extracted}, Precision: 0.0")

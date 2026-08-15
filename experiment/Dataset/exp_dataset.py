import json
import hpotk
import pronto


hpo = pronto.Ontology("http://purl.obolibrary.org/obo/hp.owl")
with open("HPO/hpo_lookup.json", "r", encoding="utf-8") as f:
    hpo_lookup = json.load(f)


def get_hpo_depth(hpo_id):
    
    term = hpo[hpo_id]
    if "obsolete" in term.name.lower():
        print(f"Term {hpo_id} is obsolete, skipping depth calculation.")

    visited = set()

    def recursive_depth(t):

        if t.id in visited:
            return float("inf")

        visited.add(t.id)

        parents = []

        for parent in t.superclasses(distance=1, with_self=False):

            # keep only true HPO terms
            if hasattr(parent, "id") and str(parent.id).startswith("HP:"):
                parents.append(parent)

        if len(parents) == 0:
            if t.id != "HP:0000001":
                print(f"Term {t.id} has no parents, returning depth 0")
            return 0
        
        return 1 + min(recursive_depth(parent) for parent in parents)

    return recursive_depth(term)



def avg_hpo_depth(depth_arr):
    if not depth_arr:
        return 0
    return sum(depth_arr) / len(depth_arr)

def get_hpo_depth_array(hpo_ids):
    return [get_hpo_depth(hpo_id) for hpo_id in hpo_ids]

def get_avg_terms(hpo_ids, notes):
    return len(hpo_ids) / len(notes) if notes else 0

def get_avg_word_length(notes):
    word_count = 0
    for note in notes:
        words = note.split()
        word_count += len(words)
    return word_count / len(notes) if notes else 0


def get_hpo_ids(filepath):
    results = []
    notes = []
    
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
        notes.append(text)
        
        phenotypes = []
        
        # Read annotation lines until blank line or next doc
        while i < n and lines[i].strip() != "":
            
            parts = lines[i].split("\t") 
            
            if len(parts) >= 4:
                hpo_id = [parts[3].strip()]
                if hpo_id not in phenotypes:
                    phenotypes.extend(hpo_id)
            elif len(parts) == 2:
                hpo_id = parts[1].strip().split(", ")
                phenotypes.extend(hpo_id)
            i += 1

        results.extend(phenotypes)
        
        i += 1  # skip blank line
    
    return results, notes

hpo_ids_GSC, notes_GSC = get_hpo_ids("C:\\Users\\micha\\OneDrive - Maastricht University\\Documents\\Thesis\\Dataset\\corpus\\GSC\\GSCplus_test_gold.tsv")

depth_arr_GSC = get_hpo_depth_array(hpo_ids_GSC)
avg_depth_GSC = avg_hpo_depth(depth_arr_GSC)

print("For GSC:")
print(f"Average HPO depth: {avg_depth_GSC}")
print(f"Median of HPO depths: {sorted(depth_arr_GSC)[len(depth_arr_GSC)//2]}")
print(f"Average number of terms per note: {get_avg_terms(hpo_ids_GSC, notes_GSC)}")
print(f"Average number of words per note: {get_avg_word_length(notes_GSC)}")
print(f"Number of note: {len(notes_GSC)}")
print(f"Number of HPO terms: {len(hpo_ids_GSC)}")
print(f"Unique HPO terms: {len(set(hpo_ids_GSC))}")

hpo_ids_CSC, notes_CSC = get_hpo_ids("C:\\Users\\micha\\OneDrive - Maastricht University\\Documents\\Thesis\\Dataset\\datasetCSC.tsv")

depth_arr_CSC = get_hpo_depth_array(hpo_ids_CSC)
avg_depth_CSC = avg_hpo_depth(depth_arr_CSC)

print("For CSC:")
print(f"Average HPO depth: {avg_depth_CSC}")
print(f"Median of HPO depths: {sorted(depth_arr_CSC)[len(depth_arr_CSC)//2]}")
print(f"Average number of terms per note: {get_avg_terms(hpo_ids_CSC, notes_CSC)}")
print(f"Average number of words per note: {get_avg_word_length(notes_CSC)}")
print(f"Number of note: {len(notes_CSC)}")
print(f"Number of HPO terms: {len(hpo_ids_CSC)}")
print(f"Unique HPO terms: {len(set(hpo_ids_CSC))}")
# Plot depth frequency for both datasets
import matplotlib.pyplot as plt
from collections import Counter

# Count frequencies of depths
depth_freq_GSC = Counter(depth_arr_GSC)
depth_freq_CSC = Counter(depth_arr_CSC)

# Get all unique depths
all_depths = sorted(set(depth_freq_GSC.keys()) | set(depth_freq_CSC.keys()))

# Create figure with side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot GSC
gsc_depths = [depth_freq_GSC.get(d, 0) for d in all_depths]
ax1.bar(all_depths, gsc_depths, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('HPO Depth', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('GSC - Depth Frequency Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(all_depths)

# Plot CSC
csc_depths = [depth_freq_CSC.get(d, 0) for d in all_depths]
ax2.bar(all_depths, csc_depths, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('HPO Depth', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('CSC - Depth Frequency Distribution', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(all_depths)

plt.tight_layout()
plt.savefig('depth_frequency_comparison.png', dpi=300, bbox_inches='tight')
plt.show()





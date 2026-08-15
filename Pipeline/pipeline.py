import spacy
import medspacy
from transformers import pipeline
import time
import json
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from rapidfuzz import fuzz
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HPO_DIR = PROJECT_ROOT / "resources" / "HPO"
LOCAL_SCIBERT_MODEL = (
    PROJECT_ROOT
    / "resources"
    / "en_core_sci_scibert-0.5.4"
    / "en_core_sci_scibert"
    / "en_core_sci_scibert-0.5.4"
)

time_start = time.time()


def load_scibert_model():
    try:
        return spacy.load("en_core_sci_scibert")
    except OSError:
        if LOCAL_SCIBERT_MODEL.exists():
            return spacy.load(str(LOCAL_SCIBERT_MODEL))
        raise


#Tokenizer method (Scibert)
def test_load_model_scibert(text):
    nlp = load_scibert_model()
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("medspacy_context") # add context (e.g. negation or uncertainty)

    doc = nlp(text)
    return nlp,doc

#NER method, use the build in tokenizer from the medical NER
def load_medical_ner_model(text):
    pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
    result = pipe(text)
    return result

# possibility to lemmatize symptom phrases (not used in the default pipeline)
def lemmatize_phrase(text, nlp):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

# returns the embeddings for a given text using the SapBERT model (i.e. 'tokenizer')
def embed_text(text, tokenizer, model):
    inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt"
    )
    
    outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    
    # normalize for cosine similarity
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    
    return vec

#returns a list with the k top candidates for each symptom in symptoms
def add_semantic_candidates(symptoms, k, tokenizer = None, model =None,index=None, term_ID_names=None, hpo_lookup=None):
    
    pheno = [] # list of symptoms with their k candidates
    
    for symptom in symptoms:
        
        # transform into embeddings
        query_vec = embed_text(symptom["text"], tokenizer, model)
        # search in the HPO embedding space the best 40 candidates
        D, I = index.search(query_vec, 40)
        
        #remove any duplicates (keep the one with the best score) and keep only the 15 best
        unique_candidates = {}
        for score, idx in zip(D[0], I[0]):
            candidate_meta = term_ID_names[idx]
            if isinstance(candidate_meta, dict):
                hpo_id = candidate_meta.get("hpo_id", "")
                hpo_term = candidate_meta.get("hpo_term", "")
            else:
                hpo_id = candidate_meta[0]
                hpo_term = candidate_meta[1]
            # keep best score per HPO
            if hpo_id not in unique_candidates or score > unique_candidates[hpo_id]["score"]:
                unique_candidates[hpo_id] = {
                    "hpo_id": hpo_id,
                    "hpo_term": hpo_term,
                    "score": float(score)
                }
            # stop early if enough unique HPOs
            if len(unique_candidates) >= 15:
                break


        # convert to list
        top_candidates = list(unique_candidates.values())
 
        # rerank candidates based on a combination of cosine similarity and lexical similarity (using fuzzy matching) with the original symptom text
        reranked = []
        for candidate in top_candidates:
            #get metadata for candidate
            query = symptom['text']
            info = hpo_lookup.get(candidate['hpo_term'], {})
            definition = info.get('definition', '')
            snomed_terms = info.get('snomed_terms', [])
            texts = [candidate['hpo_term']]
            if definition:
                texts.append(definition)
            texts.extend(snomed_terms)
            # Compute best lexical score for this candidate
            best_lex_score = 0
            for text in texts:
                score = fuzz.token_sort_ratio(query, text) / 100
                if score > best_lex_score:
                    best_lex_score = score
            #combine cosine similarity score and lexical score with a weighted average (90% cosine similarity, 10% lexical similarity)
            reranked.append({
                **candidate,
                "cross_score": 0.9 * candidate['score'] + 0.10 * best_lex_score
            }) 
        #sort candidates based on the combined score and keep the top k
        reranked = sorted(reranked, key=lambda x: x["cross_score"], reverse=True)
        if reranked[0]['cross_score'] >=0.55:  
            symptom["top_k_candidates"] = reranked[:k]
            pheno.append(symptom)

    return pheno

#add parents and children for each final candidates (not necessary, butcan be informative)
def add_parent_and_child(symptoms, hpo_lookup):
    for symptom in symptoms:
        if "top_k_candidates" in symptom:
            for candidate in symptom["top_k_candidates"]:
                info = hpo_lookup.get(candidate['hpo_term'].lower(), {})
                parent_ids = info.get("parent", [])
                child_ids = info.get("children", [])
                candidate["parent_ids"] = parent_ids
                candidate["children_ids"] = child_ids
        if symptom['hpo_id'] is not None:
            info = hpo_lookup.get(symptom['text'].lower(), {})
            parent_ids = info.get("parent", [])
            child_ids = info.get("children", [])
            symptom["parent_ids"] = parent_ids
            symptom["children_ids"] = child_ids
    return symptoms

# helper function to repair fragmented entites from the NER model
def repair_fragmented_words(entities, text):
    for word in text.split(" "):
        for ent in entities:
            if word in ent['word'].replace(" ", "") and word not in ent['word']:
                new_word = ""
                first_part_found = False
                final_part_found = False
                for word_ent in ent['word'].split(" "):
                    if word_ent in word :
                        if first_part_found == False:
                            new_word += word + " "
                            first_part_found = True
                        elif first_part_found == True:
                            continue
                    elif first_part_found == True:
                        final_part_found = True
                    elif final_part_found == True or first_part_found == False:
                        new_word += word_ent + " "
                ent['word'] = new_word.strip()
    return entities

#main pipeline function that returns the final list of symptoms with their top candidates, parents and children
def run_pipeline(text,k):

    #exctract symptoms using sciBERT ensemble and medical NER
    nlp, result_tokenizer = test_load_model_scibert(text)
    result_ner_fragmented = load_medical_ner_model(text)


    #merge symptom phrases from NER model if they are separated by a gap word
    result_ner = []
    if len(result_ner_fragmented) > 0:
        result_ner.append(result_ner_fragmented[0])
    for i in range(1, len(result_ner_fragmented)):
        ent = result_ner_fragmented[i]
        gap_text = text[result_ner[-1]['end']:ent['start']].strip().lower()
        if ent['start'] == result_ner[-1]['end']:
            if result_ner[-1]['word'] +ent['word'] in text:
                result_ner[-1]['word'] += ent['word']
            else:
                result_ner[-1]['word'] += " " +ent['word']
            result_ner[-1]['end'] = ent['end']  # Update end position
            if ent['entity_group'] != result_ner[-1]['entity_group']:
                if result_ner[-1]['entity_group'] == 'SIGN_SYMPTOM' or ent['entity_group'] == 'SIGN_SYMPTOM':
                    result_ner[-1]['entity_group'] = 'SIGN_SYMPTOM'
                elif result_ner[-1]['entity_group'] == 'DISEASE_DISORDER' or ent['entity_group'] == 'DISEASE_DISORDER':
                    result_ner[-1]['entity_group'] = 'DISEASE_DISORDER'
                elif result_ner[-1]['entity_group'] == 'DIAGNOSTIC_PROCEDURE' or ent['entity_group'] == 'DIAGNOSTIC_PROCEDURE':
                    result_ner[-1]['entity_group'] = 'DIAGNOSTIC_PROCEDURE'
                elif result_ner[-1]['entity_group'] == 'MEDICATION' or ent['entity_group'] == 'MEDICATION':
                    result_ner[-1]['entity_group'] = 'MEDICATION'
                elif result_ner[-1]['entity_group'] == 'BIOLOGICAL_STRUCTURE' or ent['entity_group'] == 'BIOLOGICAL_STRUCTURE':
                    result_ner[-1]['entity_group'] = 'BIOLOGICAL_STRUCTURE'
        elif gap_text in { "of", "of some", "of the", "without", "within", "in", "in the", "on", "on the", "at", "at the", "for", "for the", "to", "to the", "by", "by the", "from", "from the","and/or", "/"}:
            result_ner[-1]['word'] += " " + gap_text + " " + ent['word']
            result_ner[-1]['end'] = ent['end']  # Update end position
            if ent['entity_group'] != result_ner[-1]['entity_group']:
                if result_ner[-1]['entity_group'] == 'SIGN_SYMPTOM' or ent['entity_group'] == 'SIGN_SYMPTOM':
                    result_ner[-1]['entity_group'] = 'SIGN_SYMPTOM'
                elif result_ner[-1]['entity_group'] == 'DISEASE_DISORDER' or ent['entity_group'] == 'DISEASE_DISORDER':
                    result_ner[-1]['entity_group'] = 'DISEASE_DISORDER'
                elif result_ner[-1]['entity_group'] == 'DIAGNOSTIC_PROCEDURE' or ent['entity_group'] == 'DIAGNOSTIC_PROCEDURE':
                    result_ner[-1]['entity_group'] = 'DIAGNOSTIC_PROCEDURE'
                elif result_ner[-1]['entity_group'] == 'BIOLOGICAL_STRUCTURE' or ent['entity_group'] == 'BIOLOGICAL_STRUCTURE':
                    result_ner[-1]['entity_group'] = 'BIOLOGICAL_STRUCTURE'
        else:
            result_ner.append(ent)
    
    result_ner = repair_fragmented_words(result_ner, text)
    
    #combine the results from the tokenizer and the NER model to determine if the symptoms are present, negated, or uncertain
    symptoms = []
    for ent in result_ner:
        already_found = False
        for token in result_tokenizer.ents:
            if ent['word'] in token.text and (ent['entity_group'] == 'DISEASE_DISORDER' or ent['entity_group'] == 'MEDICATION' or ent['entity_group'] == 'SIGN_SYMPTOM' or ent['entity_group'] == 'DIAGNOSTIC_PROCEDURE' or ent['entity_group'] == 'BIOLOGICAL_STRUCTURE'):
                if len(ent['word']) <= len(token.text):
                    symptoms.append({
                        'text': token.lemma_,
                        'label': ent['entity_group'],
                        'is_present': (token._.is_negated == False and token._.is_uncertain == False and 'no ' not in token.text),
                        'hpo_id': None
                    })
                    already_found = True
                else:
                    symptoms.append({
                        'text': ent['word'],
                        'label': ent['entity_group'],
                        'is_present': (token._.is_negated == False and token._.is_uncertain == False and 'no ' not in ent['word']),
                        'hpo_id': None
                    })
                    already_found = True
        if (ent['entity_group'] == 'SIGN_SYMPTOM' or ent['entity_group'] == 'DISEASE_DISORDER' or ent['entity_group'] == 'DIAGNOSTIC_PROCEDURE' or ent['entity_group'] == 'BIOLOGICAL_STRUCTURE') and not already_found:
            symptoms.append({
                'text': ent['word'], 
                'label': ent['entity_group'],
                'is_present': ('no ' not in ent['word']),
                'hpo_id': None
            })

    #making sure there are no duplicates in the list of symptoms
    unique_symptoms = []
    for symptom in symptoms:
        key = symptom['text'].lower()
        if key not in [s['text'].lower() for s in unique_symptoms]:
            unique_symptoms.append(symptom)
    filtered_symptoms = []
    for symptom in unique_symptoms:
        symptom_normalized = symptom['text'].replace(" ", "").lower()
        is_substring = False
        for other_symptom in unique_symptoms:
            other_normalized = other_symptom['text'].replace(" ", "").lower()
            if (
                symptom_normalized in other_normalized
                and symptom_normalized != other_normalized
            ):
                is_substring = True
                break
        if not is_substring:
            symptom['text'] = symptom['text'].lower()
            filtered_symptoms.append(symptom)
    
    symptoms = filtered_symptoms

    #perfect matching with HPO
    with open(HPO_DIR / "hpo_lookup.json", "r", encoding="utf-8") as f:
        hpo_lookup = json.load(f)

    with open(HPO_DIR / "sorted_hpo_lookup.json", "r", encoding="utf-8") as f:
        sorted_hpo_lookup = json.load(f)

    for symptom in symptoms:
        if symptom['text'] == "":
            symptoms.remove(symptom)
            continue

        symptom_name = symptom['text'].lower().strip()

        # Exact match
        phenotype = hpo_lookup.get(symptom_name)

        # Flexible word-order match
        if phenotype is None:
            sorted_text = " ".join(sorted(symptom_name.split()))
            phenotype = sorted_hpo_lookup.get(sorted_text)

        # Partial containment
        if phenotype is None:
            for term in hpo_lookup:
                if symptom_name in term or term in symptom_name:
                    phenotype = hpo_lookup[term]
                    break

        if phenotype is not None:
            if isinstance(phenotype, dict):
                symptom['hpo_id'] = phenotype['phenotype']
            else:
                symptom['hpo_id'] = phenotype



    #semantic matching with HPO using SapBERT (change path to use different vector space)
    hpo_embeddings = np.load("resources\\vector_spaces\\hpo_embeddings_N+D+S+C.npy")
    term_ID_names = np.load("resources\\vector_spaces\\hpo_term_ID_names_N+D+S+C.npy", allow_pickle=True)

    dim = hpo_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine since normalized
    index.add(hpo_embeddings)
    model_path = "./resources/sapbert_onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForFeatureExtraction.from_pretrained(model_path)

    pheno = add_semantic_candidates(symptoms, k, tokenizer=tokenizer, model=model, index=index, term_ID_names=term_ID_names, hpo_lookup=hpo_lookup)
    final_pheno = add_parent_and_child(pheno, hpo_lookup)

    print("done")
    return final_pheno



if __name__ == "__main__":
    text = "A 32-year-old man presented to a regional general surgical unit for severe symptomatic anaemia. Within a span of 4 months his haemoglobin dropped by 62 points from 137 gL (N: 135–180 gL). His medical history included epilepsy on Valproate, heart arrhythmia on Flecainide, carpal tunnel syndrome and C5 nerve root compression. He is a non-smoker, rare alcohol user, has two dogs at home, and two children. His family history included his grandfather who had bowel cancer diagnosed at 78 years old, and a grandmother with breast cancer. The first imaging modality was a CT scan showing a large fungating mass in the body of the stomach extending into the pylorus and first part of the duodenum. He underwent upper gastrointestinal endoscopy (UGE) to investigate the pathology. Histopathology frequently returned as benign hyperplastic tissue. The first histopathology results reported gastric mucosa with minor foveolar hyperplasia, stromal oedema with minor increase in chronic inflammatory cells. There was no evidence of helicobacter organisms, active gastritis or atypia. Initial scope findings showed carpet-like polyps lining the gastric antrum and body. On a second UGE, tunnel biopsies were performed, showing the carpet-like polyps have progressed into the first part of the duodenum, in comparison to previous UGE The biopsy returned as Helicobacter heilmannii-associated chronic gastritis with hyperplastic changes of fundic gland polyp. Serum ANA, ANCA, antiparietal cell antibodies, Cytomegalovirus infection serology, hepatitis panel, serum protein electrophoresis, Helicobacter serology and CA 72–4 tumour marker were ordered. Results came back negative except for a positive Helicobacter IgG serology. H. heilmannii was treated with two courses of Helicobacter eradication regime. Repeat UGE, 2 months later, showed no evidence of Helicobacter organism. Histopathology results were hyperplastic gastric polyp with background of mild chronic inflammation. Repeat UGE 3 months after the initial UGE showed Paris classification 0-Isp in the duodenum and the polyps have extended further into the oesophagus. An endoscopic mucosal resection was performed on the oesophageal polyp with histopathology returning as juvenile polyp. Narrow-band imaging of a polyp in the gastric body showed irregular microvascular and irregular microsurface pattern. A colonoscopy revealed two 2–3 mm polyps in the ascending colon and caecum, and one 5 mm semipedunculated polyp in the sigmoid colon. The colonic lesions resembled Kudo’s Pit pattern. Histopathology of caecum and descending colon polyps returned as inflammatory juvenile polyps. Capsule endoscopy revealed scattered 0–1 p small bowel polyps and areas of telangiectasia."
    symptoms = run_pipeline(text, k=10)
    with open(HPO_DIR / "hpo_lookup.json", "r", encoding="utf-8") as f:
        hpo_lookup = json.load(f)
    
    for symptom in symptoms:
        if symptom['hpo_id'] is not None:
            print(f"Symptom: {symptom['text']}, HPO ID: {symptom['hpo_id']}")
        else:
            print(f"Symptom: {symptom['text']}, candidates: ")
            for candidate in symptom.get('top_k_candidates', []):
                print(f"  - {candidate['hpo_id']} ({candidate['hpo_term']})")
        print(symptom)



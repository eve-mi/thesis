This pipeline performs biomedical concept normalization and phenotype extraction using transformer-based 
semantic matching and ontology-aware retrieval over HPO and SNOMED CT resources.

Pipeline's process:
1. takes clinical text as input
2. use of Medical NER and spaCy ensemble to extract symptom phrases
3. lexical strategies to match to phenotype concepts (ontology matching)
4. hybrid lexical-semantic re-ranking (Fuzzy & SapBERT)
5. outputs retrived phenotypes


Repository structure
project/
│
├── experiment/         # Experiment scripts and configs
│   ├── Dataset/        #datasets used during the experiment (GSC and CSC)
│   └── insights/       # summaries of misses/hits
│
├── interface/          # User/API interface
│
├── piepline/          # External resources and embeddings
│   ├── vector_spaces/
│   ├── hpo_hashmaps/
│   ├── snomed_ct/
│   ├── sapbert/
│   └── scibert/
│
├── RAG-HPO/        # benchmark model
│
├── resources/          # External resources and embeddings
│   ├── scibert/        # scaPy ensemble
│   ├── HPO/            # HPO ontology and hashmaps
│   ├── sapbert/        # SapBERT transformer for semantic embeddings
│   ├── snomed_ct/      # SNOMED CT ontology
│   └── vector_spaces/  # different vector spaces
│
└── README.md


Interface:
run with 
    python -m streamlit run interface/app.py  


Resources needed are all present except the Clinical-AI-Apollo/Medical-NER. You can download it from Hugging Face
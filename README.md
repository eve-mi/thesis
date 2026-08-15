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

Required resources

Before running PhenoBridge, download all five resource folders listed below:

- en_core_sci_scibert-0.5.4 (through https://allenai.github.io/scispacy/, model called en_core_sci_scibert) 
- HPO: run the 'build_hashmap'
sapbert_onnx
SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z
vector_spaces

Once downloaded, place all five folders directly inside the project's resource/ directory. The expected structure is:

project/
├── resource/
│   ├── en_core_sci_scibert-0.5.4/
│   ├── HPO/
│   ├── sapbert_onnx/
│   ├── SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/
│   └── vector_spaces/
├── ...

Do not rename the folders or place them in subdirectories, as PhenoBridge expects these resources at these exact locations.
You also need to download the Clinical-AI-Apollo/Medical-NER. You can download it from Hugging Face.

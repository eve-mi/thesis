from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Load and convert automatically to ONNX
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    model_name,
    export=True  # This triggers conversion
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save ONNX model locally
ort_model.save_pretrained("./sapbert_onnx")
tokenizer.save_pretrained("./sapbert_onnx")
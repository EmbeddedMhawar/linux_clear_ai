import onnxruntime as ort
import sys

model_path = "models/model.onnx"

try:
    session = ort.InferenceSession(model_path)
    print("Model Loaded Successfully")
    
    print("\n--- Inputs ---")
    for i in session.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
    print("\n--- Outputs ---")
    for o in session.get_outputs():
        print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
        
    print("\n--- Metadata ---")
    meta = session.get_modelmeta()
    print(f"Description: {meta.description}")
    print(f"Domain: {meta.domain}")
    print(f"Graph Name: {meta.graph_name}")
    print(f"Producer Name: {meta.producer_name}")
    print(f"Version: {meta.version}")
    print(f"Custom Metadata Map: {meta.custom_metadata_map}")

except Exception as e:
    print(f"Error: {e}")

import h5py
import numpy as np


countermeasure = "base"
DATASET_DIR = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/"
CHAMELEON_FILE=f"{DATASET_DIR}chameleon_{countermeasure}_chunk_1.h5"
OUTPUT_FILE=f"{DATASET_DIR}chameleon_{countermeasure}_chunk_1_transformed.h5"
# with h5py.File(CHAMELEON_FILE, "r") as f:
#     pp = f["metadata/pinpoints/pinpoints_0"][:]

#     print("Shape:", pp.shape)
#     print("Dtype:", pp.dtype)
#     print("First 20:", pp[:20])
#     print("Last 20:", pp[-20:])

with h5py.File(CHAMELEON_FILE, "r") as f:

    trace = f["data/traces/trace_0"]
    pp = f["metadata/pinpoints/pinpoints_0"]

    print("Trace shape:", trace.shape)
    print("Trace length:", len(trace))
    print("Last pinpoint:", pp[-1])
    print("Last valid index:", len(trace) - 1)
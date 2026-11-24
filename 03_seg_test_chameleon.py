import numpy as np # type: ignore
from tqdm.auto import tqdm # type: ignore
import os

from inference_pipeline.segmentation import *
from inference_pipeline.debug import *
from matplotlib import pyplot as plt # type: ignore
from inference_pipeline.heuristic import removeFalseNegatives, removeFalsePositives, removeFalsePositives_basic
import h5py

##################################################
# Directories Setup
##################################################
countermeasure="rd"
chameleon_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/CHAM_IN/"
dataset_out_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/CHAM_OUT_2/"
model_dir=f"{dataset_out_path}ResNet_v1_{countermeasure}/checkpoints"
config_folder = f"CNN/configs/chameleon_{countermeasure}/" # /path/to/experiment/config/folder/

##################################################
# Sliding Window Classification
##################################################
print("Start: Sliding Window Classification")
SID = ""
chameleon_file = f"{chameleon_path}chameleon_base_chunk_1.h5"
output_file = f"{dataset_out_path}output_file.npy"

win_sizes_per_counter = {
    "base"  : 10_000,
    "dfs"   : 10_000,
    "rd"    : 20_000,
    "mrp"   : 20_000,
    "chf"   : 30_000,
}

stride_per_counter = {
    "base"  : 50,
    "dfs"   : 50,
    "rd"    : 100,
    "mrp"   : 100,
    "chf"   : 100,
}

window_size = win_sizes_per_counter[countermeasure] #20_000
stride = stride_per_counter[countermeasure]#100

print("Start: Testing")
classification_file = output_file #"</path/to/classification_output_file>"

labels = loaderGt(chameleon_file)

##################################################################################
#classifications = np.load(classification_file, mmap_mode='r')
##################################################################################
classification_values = np.load(classification_file, mmap_mode='r')

offsets_file = classification_file.rsplit('.', 1)[0] + '_offsets.npy'
classification_offsets = np.load(offsets_file, mmap_mode='r')

classifications = [
    classification_values[classification_offsets[i]:classification_offsets[i + 1]]
    for i in range(len(classification_offsets) - 1)
]

with h5py.File(chameleon_file, "r") as f:
    for idx in range(min(10, len(classifications))):
        trace_length = len(f[f"data/traces/trace_{idx}"])

        expected = (trace_length - window_size) // stride + 1
        actual = len(classifications[idx])

        print(
            f"Trace {idx}: "
            f"trace length={trace_length:,}, "
            f"expected windows={expected:,}, "
            f"actual predictions={actual:,}, "
            f"classification shape={classifications[idx].shape}"
        )


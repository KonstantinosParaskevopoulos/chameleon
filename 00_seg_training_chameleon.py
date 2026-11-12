import numpy as np # type: ignore
from tqdm.auto import tqdm # type: ignore

from CNN.train import train 
from inference_pipeline.sliding_window_classification import getModule, classifyTrace, saveClassification
from inference_pipeline.segmentation import *
from CNN.build_dataset_chameleon import createSubsets
from inference_pipeline.debug import *
from matplotlib import pyplot as plt # type: ignore
from inference_pipeline.heuristic import removeFalseNegatives, removeFalsePositives, removeFalsePositives_basic

##################################################
# CNN dataset creation
##################################################
print("Start: CNN dataset creation")
countermeasure="base"
chameleon_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/CHAM_IN/"
dataset_out_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/CHAM_OUT_2/"

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

# exp_name = f"{exp_name}_{countermeasure}"
# exp_dir = os.path.join(exp_config['log_dir'], exp_name)
model_dir=f"{dataset_out_path}ResNet_v1_{countermeasure}/checkpoints"
createSubsets(chameleon_path, dataset_out_path, window_size)

##################################################
# CNN training
##################################################
print("Start: CNN training")
config_folder = f"CNN/configs/chameleon_{countermeasure}/" # /path/to/experiment/config/folder/
train(config_folder, countermeasure)


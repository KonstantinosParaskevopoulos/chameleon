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
# Directories Setup
##################################################
countermeasure="base"
chameleon_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/"
dataset_out_path = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/CHAM_OUT/"
model_dir=f"{dataset_out_path}ResNet_v1_{countermeasure}/checkpoints"
config_folder = f"CNN/configs/chameleon_{countermeasure}/" # /path/to/experiment/config/folder/

##################################################
# Sliding Window Classification
##################################################
print("Start: Sliding Window Classification")
SID = ""
chameleon_file = f"{chameleon_path}chameleon_base_chunk_1.h5"
output_file = f"{dataset_out_path}output_file.npy"

module = getModule(model_dir, config_folder)

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

classifications = classifyTrace(chameleon_file, module, stride, window_size, batch_size=1024)
saveClassification(classifications, output_file)

##################################################
# Screening
##################################################
print("Start: Screening")
classification_file = output_file #"</path/to/classification_output_file>"

labels = loaderGt(chameleon_file)
classifications = np.load(classification_file, mmap_mode='r')

gts_starts = []
gts_ends = []
for label in labels:
    gts_starts.append(label['start'])
    gts_ends.append(label['end'])

init_min_distance = 150_000

major_filter_size = 150
CPs = []

for classification, gt, gt_e in tqdm(zip(classifications, gts_starts, gts_ends), total=len(classifications)):
    CPs.append(segment(classification, major_filter_size, stride, init_min_distance))

# Remove false positive and false negatives from the identified COs

starts = []
ends = []

for idx in tqdm(range(len(CPs))):
    s, e = removeFalsePositives_basic(CPs[idx]['starts'], CPs[idx]['ends'])
    s, e = removeFalseNegatives(s, e, classifications[idx])
    starts.append(s)
    ends.append(e)

fnrs_s, fprs_s = [],[]
fnrs_e, fprs_e = [],[]

for idx in tqdm(range(len(CPs))):
    fpr, fnr = errorRate(gts_starts[idx], np.asarray(starts[idx]), stride)
    fprs_s.append(fpr)
    fnrs_s.append(fnr)
    fpr, fnr = errorRate(gts_ends[idx], np.asarray(ends[idx]), stride)
    fprs_e.append(fpr)
    fnrs_e.append(fnr)
    
print(f"Total start mean FPR: {round(np.mean(fprs_s)*100,2)}%")
print(f"Total start mean FNR: {round(np.mean(fnrs_s)*100,2)}%")
print(f"Total end mean FPR: {round(np.mean(fprs_e)*100,2)}%")
print(f"Total end mean FNR: {round(np.mean(fnrs_e)*100,2)}%") 
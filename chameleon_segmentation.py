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
chameleon_path = "/local/scratch/paraskevopoulosk/Datasets/BASE"
dataset_out_path = "/local/scratch/paraskevopoulosk/Datasets/BASE/CHAM_OUT"

createSubsets(chameleon_path, dataset_out_path)

##################################################
# CNN training
##################################################
print("Start: CNN training")
config_folder = "CNN/configs/chameleon_base/" # /path/to/experiment/config/folder/
train(config_folder)

##################################################
# Sliding Window Classification
##################################################
print("Start: Sliding Window Classification")
SID = ""
chameleon_file = "/local/scratch/paraskevopoulosk/Datasets/BASE/chameleon_base_chunk_1.h5"
output_file = "/local/scratch/paraskevopoulosk/Datasets/BASE/CHAM_OUT/output_file.npy"

module = getModule(SID)

window_size = 20_000
stride = 100

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

##################################################
# Visualization
##################################################
print("Visualization . . .")
n_trace = 0
center = 100_000
margin = 20_000
lim = (center-margin, center+margin)

fig, ax = plt.subplots(3, figsize=(13, 7))
plt.rcParams.update({'font.size': 18})
fig.tight_layout(pad=2.0)

ax[0].set_xlim(lim)
for sample in starts[n_trace][1:]:
    ax[0].axvline(x=sample, color='black', linestyle='-')
ax[0].axvline(x=starts[n_trace][0], color='black', linestyle='-', label="Found CO's start")

for sample in gts_starts[n_trace][1:]:
    ax[0].axvline(x=sample//stride, color='r', linestyle='--')
ax[0].axvline(x=gts_starts[n_trace][0]//stride, color='r', linestyle='--', label='Start GT')
ax[0].legend(loc='lower right', bbox_to_anchor=(1, 0.95))


ax[1].set_xlim(lim)
for sample in ends[n_trace][1:]:
    ax[1].axvline(x=sample, color='grey', linestyle='-')
ax[1].axvline(x=ends[n_trace][0], color='grey', linestyle='-', label="Found CO's end")

for sample in gts_ends[n_trace][1:]:
    ax[1].axvline(x=sample//stride, color='y', linestyle='--')
ax[1].axvline(x=gts_ends[n_trace][0]//stride, color='y', linestyle='--', label='End GT')
ax[1].legend(loc='lower right', bbox_to_anchor=(1, 0.95))


ax[2].set_xlim(lim)
ax[2].plot(np.argmax(classifications[n_trace], axis=1))
print(f"End! Results in: {dataset_out_path}")
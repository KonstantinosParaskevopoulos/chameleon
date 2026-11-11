import numpy as np # type: ignore
from tqdm.auto import tqdm # type: ignore
import os

from inference_pipeline.segmentation import *
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

plot_output_dir = f"{dataset_out_path}plots"
os.makedirs(plot_output_dir, exist_ok=True)


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

# Original Code

# for idx in tqdm(range(len(CPs))):
#     s, e = removeFalsePositives_basic(CPs[idx]['starts'], CPs[idx]['ends'])
#     s, e = removeFalseNegatives(s, e, classifications[idx])
#     starts.append(s)
#     ends.append(e)

# Test Code
for idx in tqdm(range(len(CPs))):
    co_starts = CPs[idx]['starts']
    co_ends = CPs[idx]['ends']

    print(
        f"\nTrace {idx}: "
        f"{len(co_starts)} detected starts, "
        f"{len(co_ends)} detected ends"
    )

    if len(co_starts) != len(co_ends):
        print("WARNING: Different numbers of starts and ends!")
        #print("Starts:", co_starts)
        #print("Ends:", co_ends)

    s, e = removeFalsePositives_basic(co_starts, co_ends)

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


ious_s = []
ious_e = []

for idx in tqdm(range(len(CPs))):
    iou_s = IoU(gts_starts[idx], gts_ends[idx], np.asarray(starts[idx]), stride)
    ious_s.append(iou_s)

    iou_e = IoU(gts_ends[idx], gts_starts[idx], np.asarray(ends[idx]), stride)
    ious_e.append(iou_e)

print(f"Total start mean IoU: {round(np.mean(ious_s)*100,2)}%")
#print(f"Total end mean IoU: {round(np.mean(ious_e)*100,2)}%") 

##################################################
# Save Segmentation Results
##################################################
print("Saving segmentation results...")

segmentation_output_dir = "segmentation_results" #os.path.join(dataset_out_path, "segmentation_results")
os.makedirs(segmentation_output_dir, exist_ok=True)

for idx, (trace_starts, trace_ends) in enumerate(zip(starts, ends)):
    output_file = os.path.join(
        segmentation_output_dir,
        f"trace_{idx}_segments.txt"
    )

    with open(output_file, "w") as f:
        for start, end in zip(trace_starts, trace_ends):
            f.write(f"{start}\t{end}\n")

print(f"Segmentation results saved to: {segmentation_output_dir}")


##################################################
# Validate Segmentation Results
##################################################
print("Checking segmentation results...")

for idx, (trace_starts, trace_ends) in enumerate(zip(starts, ends)):

    issues = []

    # Check whether the number of starts and ends matches
    if len(trace_starts) != len(trace_ends):
        issues.append(
            f"Unequal number of starts ({len(trace_starts)}) "
            f"and ends ({len(trace_ends)})"
        )

    # Check individual regions
    for i, (start, end) in enumerate(zip(trace_starts, trace_ends)):

        if start >= end:
            issues.append(
                f"Invalid region {i}: start={start}, end={end}"
            )

        # Check ordering and overlap with the previous region
        if i > 0:
            prev_start = trace_starts[i - 1]
            prev_end = trace_ends[i - 1]

            if start < prev_start:
                issues.append(
                    f"Out-of-order start at region {i}: "
                    f"{start} < previous start {prev_start}"
                )

            if end < prev_end:
                issues.append(
                    f"Out-of-order end at region {i}: "
                    f"{end} < previous end {prev_end}"
                )

            if start < prev_end:
                issues.append(
                    f"Overlap between regions {i-1} and {i}: "
                    f"previous end={prev_end}, current start={start}"
                )

    # Report results
    # if issues:
    #     print(f"\nTrace {idx}: {len(issues)} issue(s) found")
    #     for issue in issues:
    #         print(f"  - {issue}")
    # else:
    #     print(f"Trace {idx}: OK ({len(trace_starts)} regions)")


##################################################
# Visualization
##################################################
print("Visualization . . .")

# Output directory for plots
plot_output_dir = "plots"#f"{dataset_out_path}/plots"
os.makedirs(plot_output_dir, exist_ok=True)

n_trace = 0
for n_trace in range(16):
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

    # # Save figure
    # output_file = os.path.join(plot_output_dir, f"trace_{n_trace}_visualization.png")
    # fig.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.close(fig)

    # print(f"Visualization saved to: {output_file}")
    # print(f"End! Results in: {dataset_out_path}")

    # Save figure as PNG and PDF
    plot_output_file = os.path.join(plot_output_dir, f"trace_{n_trace}_visualization")

    fig.savefig(f"{plot_output_file}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{plot_output_file}.pdf", bbox_inches='tight')

    plt.close(fig)

print(f"Visualizations saved to: {plot_output_file}.png and {plot_output_file}.pdf")
print(f"End! Results in: {dataset_out_path}")


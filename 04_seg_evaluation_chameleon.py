import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from inference_pipeline.debug import loaderGt


##################################################
# Configuration
##################################################

COUNTERMEASURE = "rd"

CHAMELEON_PATH = (
    f"/local/scratch/paraskevopoulosk/Datasets/"
    f"{COUNTERMEASURE.upper()}/CHAM_IN/"
)
OUTPUT_PATH = f"/local/scratch/paraskevopoulosk/Datasets/{COUNTERMEASURE.upper()}/CHAM_OUT_2/"

H5_FILE = f"{CHAMELEON_PATH}chameleon_base_chunk_1.h5"

SEGMENTATION_DIR = f"{OUTPUT_PATH}segmentation_results"

OUTPUT_DIR = f"{OUTPUT_PATH}segmentation_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ERROR_FILE = os.path.join(OUTPUT_DIR, "pinpoint_errors.txt")

# Classification parameters
win_sizes_per_counter = {
    "base"  : 10_000,
    "dfs"   : 10_000,
    "rd"    : 20_000,
    "mrp"   : 20_000,
    "chf"   : 30_000,
}
window_size = win_sizes_per_counter[COUNTERMEASURE] #20_000

stride_per_counter = {
    "base"  : 50,
    "dfs"   : 50,
    "rd"    : 100,
    "mrp"   : 100,
    "chf"   : 100,
}


STRIDE = stride_per_counter[COUNTERMEASURE]#100

# Histogram parameters
HISTOGRAM_BINS = 50
PERCENTILE = 95


##################################################
# Load Ground Truth
##################################################

print("Loading ground truth...")

labels = list(loaderGt(H5_FILE))

n_traces = len(labels)

print(f"Loaded ground truth for {n_traces} traces.")


##################################################
# Evaluation
##################################################

results = []

# Trace-level statistics
TP = 0
FN = 0

# Region-level statistics
FP = 0
INVALID = 0
TOTAL_DETECTIONS = 0

MULTIPLE_DETECTIONS = 0
NO_DETECTIONS = 0

for idx in range(n_traces):

    gt_start = int(labels[idx]["start"])
    gt_end = int(labels[idx]["end"])

    segmentation_file = os.path.join(
        SEGMENTATION_DIR,
        f"trace_{idx}_segments.txt"
    )

    detected_regions = []

    # Read segmentation results
    if os.path.exists(segmentation_file):

        with open(segmentation_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) < 2:
                    print(
                        f"WARNING: Invalid line in trace {idx}: {line}"
                    )
                    continue

                start = int(float(values[0]))
                end = int(float(values[1]))

                # Convert classification coordinates to trace samples
                start *= STRIDE
                end *= STRIDE

                detected_regions.append((start, end))

    else:
        print(f"WARNING: Missing segmentation file for trace {idx}")

    TOTAL_DETECTIONS += len(detected_regions)

    if len(detected_regions) == 0:
        NO_DETECTIONS += 1

    if len(detected_regions) > 1:
        MULTIPLE_DETECTIONS += 1

    # Separate valid and invalid regions
    valid_regions = []
    invalid_regions = []

    for start, end in detected_regions:

        if start < 0 or end <= start:
            invalid_regions.append((start, end))
        else:
            valid_regions.append((start, end))

    INVALID += len(invalid_regions)

    # Every invalid region is a false positive
    FP += len(invalid_regions)

    # Find the best valid detection
    if len(valid_regions) > 0:

        region_errors = []

        for start, end in valid_regions:

            start_error = abs(start - gt_start)
            end_error = abs(end - gt_end)

            average_error = (start_error + end_error) / 2

            region_errors.append(
                (average_error, start_error, end_error, start, end)
            )

        # Select the region with the lowest average error
        best_region = min(region_errors, key=lambda x: x[0])

        average_error, start_error, end_error, best_start, best_end = best_region

        # One valid region is matched to the ground truth
        TP += 1

        # All remaining valid regions are false positives
        FP += len(valid_regions) - 1

        status = "TP"

    else:

        # No valid region was detected
        FN += 1

        average_error = np.nan
        start_error = np.nan
        end_error = np.nan

        status = "FN"

    results.append({
        "trace": idx,
        "average_error": average_error,
        "start_error": start_error,
        "end_error": end_error,
        "status": status,
        "total_detections": len(detected_regions),
        "valid_detections": len(valid_regions),
        "invalid_detections": len(invalid_regions)
    })


##################################################
# Save Results
##################################################

print("\nSaving pinpoint errors...")

with open(ERROR_FILE, "w") as f:

    f.write(
        "trace\taverage_error\tstart_error\tend_error\t"
        "status\ttotal_detections\tvalid_detections\t"
        "invalid_detections\n"
    )

    for result in results:

        f.write(
            f"{result['trace']}\t"
            f"{result['average_error']}\t"
            f"{result['start_error']}\t"
            f"{result['end_error']}\t"
            f"{result['status']}\t"
            f"{result['total_detections']}\t"
            f"{result['valid_detections']}\t"
            f"{result['invalid_detections']}\n"
        )

print(f"Results saved to: {ERROR_FILE}")


##################################################
# Detection Statistics
##################################################

print("\n========== DETECTION STATISTICS ==========")

print(f"Total traces: {n_traces}")

print(f"True positives: {TP}")
print(f"False negatives: {FN}")
print(f"False positives (extra/invalid regions): {FP}")

print(f"Traces with no detections: {NO_DETECTIONS}")
print(f"Traces with multiple detections: {MULTIPLE_DETECTIONS}")

print(f"Invalid regions: {INVALID}")
print(f"Total detected regions: {TOTAL_DETECTIONS}")

# Trace-level detection rates
miss_rate = FN / n_traces if n_traces else 0
detection_rate = TP / n_traces if n_traces else 0

# Region-level false discovery proportion
false_discovery_rate = (
    FP / TOTAL_DETECTIONS if TOTAL_DETECTIONS else 0
)

# Average number of false detections per trace
fp_per_trace = FP / n_traces if n_traces else 0

print("\n========== ERROR RATES ==========")

print(f"Detection rate: {detection_rate * 100:.2f}%")
print(f"Miss rate (FN rate): {miss_rate * 100:.2f}%")
print(
    f"False discovery proportion: "
    f"{false_discovery_rate * 100:.2f}%"
)
print(f"False positives per trace: {fp_per_trace:.4f}")


##################################################
# Error Statistics
##################################################

error_types = {
    "Average error": "average_error",
    "Start error": "start_error",
    "End error": "end_error"
}

print("\n========== PINPOINT ERROR STATISTICS ==========")

for name, key in error_types.items():

    errors = np.array(
        [r[key] for r in results],
        dtype=float
    )

    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        print(f"{name}: No valid errors available.")
        continue

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    percentile_error = np.percentile(errors, PERCENTILE)

    print(f"\n{name}:")
    print(f"  Valid measurements: {len(errors)}")
    print(f"  Mean: {mean_error:.2f}")
    print(f"  Median: {median_error:.2f}")
    print(f"  {PERCENTILE}th percentile: {percentile_error:.2f}")


##################################################
# Histograms
##################################################

print("\nGenerating histograms...")

for name, key in error_types.items():

    errors = np.array(
        [r[key] for r in results],
        dtype=float
    )

    # Exclude traces without a valid detection
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        continue

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    percentile_error = np.percentile(errors, PERCENTILE)

    plt.figure(figsize=(11, 6))

    plt.hist(
        errors,
        bins=HISTOGRAM_BINS,
        edgecolor="black",
        alpha=0.75
    )

    plt.axvline(
        mean_error,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_error:.2f}"
    )

    plt.axvline(
        median_error,
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_error:.2f}"
    )

    plt.axvline(
        percentile_error,
        linestyle=":",
        linewidth=2,
        label=f"{PERCENTILE}th percentile: {percentile_error:.2f}"
    )

    plt.xlabel("Absolute error (samples)")
    plt.ylabel("Number of traces")

    plt.title(f"{name} Distribution")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    filename = (
        name.lower().replace(" ", "_") + "_histogram.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print(f"Histograms saved to: {OUTPUT_DIR}")
print("\nEvaluation completed.")

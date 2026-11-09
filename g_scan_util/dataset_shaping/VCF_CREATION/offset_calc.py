import h5py # type: ignore
import numpy as np # type: ignore

## Used to reverse engineer the offset used by the dataset authors during the measurements
## This step is needed to center the measurements around 0
## Currently only used for the RD part of the chameleon dataset

H5_FILE = "/local/scratch/paraskevopoulosk/Datasets/RD/chameleon_rd_chunk_1.h5"
CHUNK_SIZE = 1_000_000

means = []

with h5py.File(H5_FILE, "r") as f:

    for i in range(32):

        trace_name = f"data/traces/trace_{i}"
        trace = f[trace_name]

        total = 0.0
        count = 0

        # Process the trace in chunks to avoid loading
        # the entire 135M-sample trace into memory.
        for start in range(0, trace.shape[0], CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, trace.shape[0])

            chunk = trace[start:end]

            total += np.sum(chunk, dtype=np.float64)
            count += len(chunk)

        mean = total / count
        means.append(mean)

        print(f"{trace_name:8s}: mean = {mean:.10f}")


# Convert to NumPy array
means = np.array(means)


# Statistics across the 32 traces
minimum = np.min(means)
maximum = np.max(means)
overall_mean = np.mean(means)
std = np.std(means)

suggested_offset = overall_mean


print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

print(f"Minimum trace mean : {minimum:.10f}")
print(f"Maximum trace mean : {maximum:.10f}")
print(f"Mean of means     : {overall_mean:.10f}")
print(f"Std. deviation    : {std:.10f}")
print(f"Range             : {maximum - minimum:.10f}")

print("\nSuggested common offset:")
print(f"    {suggested_offset:.10f}")
import h5py
import numpy as np

dataset_path = "/local/scratch/paraskevopoulosk/Datasets/BASE/CHAM_IN/chameleon_base_chunk_1.h5"

with h5py.File(dataset_path, "r") as hf:

    print("\n========== HDF5 STRUCTURE ==========")

    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        else:
            print(f"GROUP:   {name}")

    hf.visititems(inspect)

    print("\n========== TRACE INFORMATION ==========")

    trace_group = hf["data/traces"]
    trace_keys = sorted(trace_group.keys())

    print(f"Number of traces: {len(trace_keys)}")

    for key in trace_keys[:5]:
        ds = trace_group[key]
        print(f"{key}: shape={ds.shape}, dtype={ds.dtype}")

    print("\n========== PINPOINT INFORMATION ==========")

    pinpoint_group = hf["metadata/pinpoints"]
    pinpoint_keys = sorted(pinpoint_group.keys())

    print(f"Number of pinpoint datasets: {len(pinpoint_keys)}")

    for key in pinpoint_keys[:5]:
        data = pinpoint_group[key][()]

        print(f"\n{key}:")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"First entries:\n{data[:10]}")
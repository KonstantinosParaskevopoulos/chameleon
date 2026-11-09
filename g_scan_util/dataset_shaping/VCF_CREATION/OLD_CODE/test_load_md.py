import h5py # type: ignore
import numpy as np # type: ignore
import random
import os

TRACE_PATH = "metadata/pinpoints/pinpoints_0"  #int16  Change for different traces
CHAMELEON_PATH = "/home/konstantinos/Documents/GitHub/Genome-Scanning/Datasets/BASE/chameleon_base_chunk_1.h5"

def load_md(path, md_path, txt_path, zoom_trace=0, start_sample=0):
    """
    This function loads 1 chameleon pinpoints file, specified by the path + trace_path.
    Part of the pinpoints file can be loaded using the zoom_trace input
    
    :String path: Path where the H5 dataset file is located
    :String trace_path: Path where the trace is located inside the dataset file
    :Integer zoom_trace: Used to specify the number of the first N samples to load, instead of the full trace
    """
    with h5py.File(path, "r") as f:
        dset = f[md_path]
        pinpoints = dset[:]   # ALWAYS load full metadata first

    if zoom_trace != 0:     #If zoom_trace is NOT 0 load only that part of traces - Used for debugging purposes
        end_sample = start_sample + zoom_trace

        # Boolean mask for overlapping pinpoints
        mask = (pinpoints['end'] > start_sample) & (pinpoints['start'] < end_sample)
        pinpoints = pinpoints[mask]

        # Clip pinpoints to the window
        pinpoints['start'] = np.maximum(pinpoints['start'], start_sample)
        pinpoints['end']   = np.minimum(pinpoints['end'], end_sample)
    
    if txt_path is not None:    #Save to text
        # Convert to regular (N,2) array
        arr = np.stack([pinpoints['start'], pinpoints['end']], axis=1)
        np.savetxt(txt_path, arr, fmt='%d', delimiter='\t')
        print(f"Pinpoints saved to {txt_path}")

    print(f"First 10 Samples: {pinpoints[:10]}")

    return pinpoints

def main():
    pinpoints = load_md(CHAMELEON_PATH, TRACE_PATH,"test.txt", 710840, 52)
    print(pinpoints)

if __name__ == "__main__":
    main()

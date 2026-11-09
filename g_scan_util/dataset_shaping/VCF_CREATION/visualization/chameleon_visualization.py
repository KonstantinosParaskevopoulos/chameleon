import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to Chameleon dataset
CHAMELEON_PATH = "../chameleon/traces/RD/RD/chameleon_rd_chunk_1.h5"
#CHAMELEON_PATH = "../chameleon/traces/RD/RD/chameleon_rd_chunk_2.h5"

# HDF5 path to traces (change if needed)
TRACE_PATH = "data/traces/trace_0"  #int16
MD_PATH = "metadata/pinpoints/pinpoints_0" #

# Number of traces to plot
#NUM_TRACES = 10
NUM_AES = 5 #counted manually

# Zoom window (prefix)
ZOOM_END = 1_000_000    #Show first million samples


def load_traces(path, trace_path, zoom_end):
    with h5py.File(path, "r") as f:
        trace = f[trace_path][:zoom_end]   # materialize slice
        print(trace[:10])   #for debug
    return trace

def load_metadata(path, md_path):
    with h5py.File(path, "r") as f:
        metadata = f[md_path][:NUM_AES]
        print(metadata)    #for debug
    return metadata

def plot_traces(traces, metadata):
    plt.figure(figsize=(14, 5))

    
    plt.plot(traces, linewidth=0.5, label="Power trace")

    for i, pp in enumerate(metadata):
        plt.axvline(pp["start"], linestyle="--", color = 'g', linewidth=1)
        plt.axvline(pp["end"], linestyle="--",color = 'r', linewidth=1)


    plt.title("Chameleon Dataset")
    plt.xlabel("Sample index")
    plt.ylabel("Power")

    plt.xlim(0, ZOOM_END)

    plt.legend(fontsize=8)
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig("chameleon_traces_prefix_zoom.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    traces = load_traces(CHAMELEON_PATH, TRACE_PATH, ZOOM_END)
    metadata = load_metadata(CHAMELEON_PATH, MD_PATH)
    plot_traces(traces, metadata)


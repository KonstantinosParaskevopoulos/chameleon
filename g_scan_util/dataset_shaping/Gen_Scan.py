import h5py # type: ignore
import numpy as np # type: ignore
import random
import os

################################################################
# This file contains clean code for Trace -> VCF style encoding
# for Genome Scanning operations, to be used for SCA purposes
################################################################

# Dataset paths
#CHAMELEON_PATH = "../chameleon/traces/RD/RD/chameleon_rd_chunk_2.h5"
#CHAMELEON_PATH_NORMALIZED = "./chameleon_norm/chameleon_rd_chunk_1_normalized.h5"
#Output path

# Trace paths
#TRACE_PATH = "data/traces/trace_0"  #int16
#TRACE_PATH_NORMALIZED = "trace"     #int16
#MD_PATH = "metadata/pinpoints/pinpoints_0"  #int16  Change for different traces


# Parameters for Trace Loading
#TRACE_ZOOM = 20 # Number of trace samples
#SAMPLES = 100   # Number of VCF samples per trace

# Parameters for Normalization
#OFFSET = 0.0            # Variable to fix trace offset
TRACE_MAX = 2047        # Max Theoretical Sample Value
TRACE_MIN = -2048       # Min Theoretical Sample Value

#---------------------TRACE LOAD & NORMALIZATION---------------------#

def load_trace(path, trace_path, zoom_trace=0, start_sample=0):
    """
    This function loads 1 chameleon trace, specified by the path + trace_path.
    Part of the trace can be loaded using the zoom_trace input
    
    :String path: Path where the H5 dataset file is located
    :String trace_path: Path where the trace is located inside the dataset file
    :Integer zoom_trace: Used to specify the number of the first N samples to load, instead of the full trace
    """
    with h5py.File(path, "r") as f:
        dset = f[trace_path]

        if zoom_trace != 0:                                 #If zoom_trace is NOT 0 load only that part of traces - Used for debugging purposes
            end_sample = start_sample + zoom_trace          # Load Trace
            trace = dset[start_sample:end_sample]
        else:
            trace = dset[:]

        # Debug
        #print(f"First 10 Samples: {trace[:10]}")            # Load FULL Trace if zoom_trace = 0

    return trace


# def load_metadata(path, md_path):
#     with h5py.File(path, "r") as f:
#         metadata = f[md_path][:NUM_AES]
#         print(metadata)    #for debug
#     return metadata

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


def normalize_trace(trace, offset=0):
    """
    Normalizes input trace between 0 and 1
    
    :param trace: Input Trace
    :param offset: Correcting offset
    """
    #print(f"  Original mean/std: {trace.mean():.3f} / {trace.std():.3f}")
    trace = trace + offset  # Offset Correction
    #print(f"  Offseted mean/std: {trace.mean():.3f} / {trace.std():.3f}")
    trace = trace.astype(np.float32)

    #print(f"Original Value (with Offset) = {trace[:10]}")
    

    #print(f"  Converted mean/std: {trace.mean():.3f} / {trace.std():.3f}")
    trace_norm = (trace - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)    ##Min-Max Normalization
    #print(f"Normalized Value = {trace_norm[:10]}")
    #trace_norm = trace_norm  
    #trace_norm = round(trace_norm * norm_factor)
    return trace_norm

#---------------------NORMALIZATION STATISTICS---------------------#

def check_statistics(trace, trace_norm):
    print("Statistics check:")
    print(f"  Original mean/std: {trace.mean():.3f} / {trace.std():.3f}")
    print(f"  Normalized mean/std: {trace_norm.mean():.3f} / {trace_norm.std():.3f}")
def check_adc_range(trace, min_val=TRACE_MIN, max_val=TRACE_MAX):
    below = np.sum(trace < min_val)
    above = np.sum(trace > max_val)

    print("ADC range check:")
    print(f"  Expected range: [{min_val}, {max_val}]")
    print(f"  Below min: {below}")
    print(f"  Above max: {above}")

    if below > 0 or above > 0:
        print("  WARNING: Out-of-range ADC samples detected!")
    else:
        print("  OK: All samples within ADC range.")
def check_normalized_bounds(trace_norm, eps=1e-6):
    min_val = trace_norm.min()
    max_val = trace_norm.max()

    print("Normalization bounds check:")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")

    if min_val < -eps or max_val > 1.0 + eps:
        print("  WARNING: Normalized values outside [0,1]!")
    else:
        print("  OK: Normalized values within [0,1].")
def check_endpoint_mapping():
    min_norm = (TRACE_MIN - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)
    max_norm = (TRACE_MAX - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)

    print("Endpoint mapping check:")
    print(f"  TRACE_MIN → {min_norm}")
    print(f"  TRACE_MAX → {max_norm}")
def check_zero_mapping():
    zero_norm = (0 - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)
    print("Zero-code mapping check:")
    print(f"  ADC 0 → {zero_norm}")

#---------------------TRACE SAVE & LOAD---------------------#
def save_trace(path, trace_norm):
    with h5py.File(path, "w") as f:
        f.create_dataset("trace", data=trace_norm, dtype=np.float32)#,chunks=(32768,))

#---------------------BIN ARRAY ENCODING---------------------#

def encode(n, norm_factor):
    """
    Input sample
    Output Series of 1s followed by 0's
    """
    
    num_of_ones = n * norm_factor
    #print(f"Final Value (Unrounded) = {num_of_ones}")
    num_of_ones = round(num_of_ones)
    #print(f"Final Value (Rounded) = {num_of_ones}")
    #print(f"Num of One's: {num_of_ones}")
    num_of_zeroes = norm_factor - num_of_ones
    #print(f"Num of Zero's: {num_of_zeroes}")
    
    bin_mat = [0] * norm_factor
    
    for i in range(num_of_ones):      #):
        bin_mat[i] = 1

    for i in range(num_of_zeroes):
        bin_mat[i+num_of_ones] = 0    #] = 0

    #print(bin_mat)
    return bin_mat

#---------------------VCF WRITING FUNCTION---------------------#

# SSD / NVMe
# buffer_mb = 8–32
# FLUSH_EVERY = 500k – 1M

# HDD / Network FS
# buffer_mb = 1–4
# FLUSH_EVERY = 50k – 200k


FLUSH_EVERY = 500_000    # tune: 100k–1M depending on SSD/HDD


def init_vcf(
    output_file,
    num_samples,
    chrom="1",
    buffer_mb=8
):
    f = open(
        output_file,
        mode="w",                      # append-only, no truncation
        encoding="utf-8",
        buffering=buffer_mb * 1024 * 1024
    )

    # Write header ONLY if file is empty
    #if ((f.tell() == 0)):# or (f.tell() != 0)):        #For debugging reasons. Will DELETE later
    f.write("##fileformat=VCFv4.2\n")
    f.write(
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    )
    sample_headers = "\t".join(
        f"sample{i+1}" for i in range(num_samples)
    )
    f.write(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + sample_headers + "\n"
    )
    f.flush()
    os.fsync(f.fileno())   # header is now durable

    return f



def append_vcf_row(
    f,
    binary_row,
    pos,
    chrom="1",
):
    #print(f"pos = {pos}")
    # Pre-allocate list (faster)
    row = [
        chrom,
        str(pos),
        ".",
        "A", #random.choice(ref_alleles),
        "G", #random.choice(alt_alleles),
        ".",
        ".",
        ".",
        "GT"
    ]
    
    row.extend(str(v) for v in binary_row)
 

    f.write("\t".join(row))
    f.write("\n")


def flush_buffer(f):
    # Periodic durability checkpoint
    f.flush()
    os.fsync(f.fileno())

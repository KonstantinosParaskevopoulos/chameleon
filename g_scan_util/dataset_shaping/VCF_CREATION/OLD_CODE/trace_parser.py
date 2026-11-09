import h5py
import numpy as np

# Dataset paths
CHAMELEON_PATH = "../chameleon/traces/RD/RD/chameleon_rd_chunk_1.h5"
OUTPUT_PATH = "./chameleon_norm/AES_5000_chameleon_rd_chunk_1_normalized.h5"#OUTPUT_FILE = "chameleon_base_chunk_1_normalized.h5"
TRACE_PATH = "data/traces/trace_0"  #int16

OFFSET = -0.1
TRACE_MAX = 2047        # 2^11 - 1
TRACE_MIN = -2048       # -(2^11)
TRACE_ZOOM = 5000
START_SAMPLE = 7675
# def load_trace(path, trace_path):
#     with h5py.File(path, "r") as f:
#         trace = f[trace_path][0]   # single long trace
#     return trace

def load_trace(path, trace_path, zoom_trace=0, start_sample=0):
    with h5py.File(path, "r") as f:
        dset = f[trace_path]

        if zoom_trace != 0:                                 #If zoom_trace is NOT 0 load only that part of traces - Used for debugging purposes
            end_sample = start_sample + zoom_trace          # Load Trace
            trace = dset[start_sample:end_sample]
        else:
            trace = dset[:]

        # Debug
        print(f"First 10 Samples: {trace[:10]}")            # Load FULL Trace if zoom_trace = 0

    return trace


def normalize_trace(trace):
    trace = trace.astype(np.float32)    
    #min_val = trace.min()
    #max_val = trace.max()

    #if TRACE_MAX == TRACE_MIN:
    #    raise ValueError("Trace is constant; cannot normalize.")
    trace_norm = (trace - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)    ##Min-Max Normalization
    trace_norm = trace_norm - OFFSET  # Take possible offset into account
    return trace_norm
#---------------------DEBUG---------------------#

def check_statistics(trace, trace_norm):
    print("Statistics check:")
    print(f"  Original mean/std: {trace.mean():.6f} / {trace.std():.6f}")
    print(f"  Normalized mean/std: {trace_norm.mean():.6f} / {trace_norm.std():.6f}")

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


#-----------------------------------------------#

def save_trace(path, trace_norm):
    with h5py.File(path, "w") as f:
        f.create_dataset("trace", data=trace_norm, dtype=np.float32)#,chunks=(32768,))

if __name__ == "__main__":
  
    trace = load_trace(CHAMELEON_PATH, TRACE_PATH, TRACE_ZOOM, START_SAMPLE)
    trace_norm = normalize_trace(trace)
   
    save_trace(OUTPUT_PATH, trace_norm)
    
    debug=1
    if(debug):
        print(f"Original min/max: {trace.min()} / {trace.max()}")
        print(f"Normalized min/max: {trace_norm.min()} / {trace_norm.max()}")
        check_adc_range(trace)
        check_normalized_bounds(trace_norm)
        check_endpoint_mapping()
        check_zero_mapping()
        check_statistics(trace, trace_norm)
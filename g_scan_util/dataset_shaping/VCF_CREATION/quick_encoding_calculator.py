import numpy as np # type: ignore



# Parameters for Normalization
#OFFSET = 0.0            # Variable to fix trace offset
TRACE_MAX = 2047        # Max Theoretical Sample Value
TRACE_MIN = -2048       # Min Theoretical Sample Value
NORM_FACTOR=100

NORM_ERROR=((TRACE_MAX-TRACE_MIN)/NORM_FACTOR)/2    # Max error during normalization & encoding due to rounding

def denormalize_trace(num_of_ones, norm_factor, offset=0):
    """
    Normalizes input trace between 0 and 1
    
    :param trace: Input Trace
    :param offset: Correcting offset
    """
    trace = float(num_of_ones * (TRACE_MAX-TRACE_MIN) / norm_factor)

    trace = trace - offset + TRACE_MIN
    
    trace = int(trace)
    # trace = trace + offset  # Offset Correction
    # trace = trace.astype(np.float32)

    # trace_norm = (trace - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)    ##Min-Max Normalization
    # trace_norm = trace_norm  
    return trace

def normalize_trace(trace, norm_factor, offset=0):
    """
    Normalizes input trace between 0 and 1
    
    :param trace: Input Trace
    :param offset: Correcting offset
    """
    print(f"Original Value = {trace}")
    trace = trace + offset  # Offset Correction
    trace = float(trace)

    trace_norm = (trace - TRACE_MIN) / (TRACE_MAX - TRACE_MIN)    ##Min-Max Normalization
    #trace_norm = trace_norm  
    print(f"Original Value (with Offset) = {trace}")
    print(f"Normalized Value = {trace_norm}")
    trace_norm = trace_norm * norm_factor
    print(f"Final Value (Unrounded) = {trace_norm}")
    trace_norm = round(trace_norm)
    print(f"Final Value (Rounded) = {trace_norm}")
    
    return trace_norm

def main():
    trace_sample = int(input("Trace Value or Number of 1's: "))
    offset = int(input("Offset: "))
    choice = int(input("0 for encoding, 1 for decoding: "))
    output = 0

    if(choice==0):
        output = normalize_trace(trace_sample, NORM_FACTOR, offset)
        print(f"Output: {output}")
    else:
        output = denormalize_trace(trace_sample, NORM_FACTOR, offset)
        print(f"Output: {output}")

if __name__ == "__main__":
    main()

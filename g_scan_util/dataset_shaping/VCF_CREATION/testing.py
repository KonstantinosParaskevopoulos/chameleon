import h5py # type: ignore
import numpy as np # type: ignore
import random
import os
from Gen_Scan import *
import math


# Trace paths
TRACE_PATH = "data/traces/trace_0"  #int16  Change for different traces
#TRACE_PATH_NORMALIZED = "trace"     #int16

CHAMELEON_PATH="/home/megauser/chameleon/traces/RD/RD/chameleon_rd_chunk_1.h5"
TRACE_SAMPLES=5000
START_SAMPLE=0
NORM_FACTOR=50
OFFSET = 1267
#OFFSET1 = -1638.5


VCF_FILE = "/home/megauser/Genome-Scanning/Testing_VCFs/VCF_x50_0-5000_o1267_chameleon_rd_chunk_1.vcf"
TOTAL_LINES = TRACE_SAMPLES + 3   # includes headers
DATA_START_LINE = 4
K = 100  #Number of tests
SEED = 42


# Parameters for Normalization
#OFFSET = 0.0            # Variable to fix trace offset
TRACE_MAX = 2047        # Max Theoretical Sample Value
TRACE_MIN = -2048       # Min Theoretical Sample Value

NORM_ERROR=math.ceil(((TRACE_MAX-TRACE_MIN)/NORM_FACTOR)/2)    # Max error during normalization & encoding due to rounding

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

def decode(bin_mat, norm_factor):
    """
    Input Series of 1s followed by 0's
    Output sample
    """
    num_of_ones = 0
    for i in range(bin_mat.len):
        if(bin_mat[i]==1):
            num_of_ones+=1

    num_of_zeroes = bin_mat.len - num_of_ones
    #print(f"Num of Zero's: {num_of_zeroes}")
    
    if((num_of_ones+num_of_zeroes)!=norm_factor):
        print("ERROR: Total length of input array different than expected")
    
    # #bin_mat = [0] * norm_factortest_trace
    
    # for i in range(num_of_ones):      #):
    #     bin_mat[i] = 1

    # for i in range(num_of_zeroes):
    #     bin_mat[i+num_of_ones] = 0    #] = 0

    #print(bin_mat)
    return num_of_ones
 
    # trace = trace.astype(np.float32)


def decode_vcf():
    random.seed(SEED)

    # Pick random *file line numbers*
    targets = sorted(
        random.sample(range(DATA_START_LINE, TOTAL_LINES + 1), K)
    )
    targets_set = set(targets)
    results = []

    with open(VCF_FILE) as f:
        for file_line_num, line in enumerate(f, start=1):
            if file_line_num <= 3:#not in targets_set:
                continue
            if file_line_num not in targets_set:
                continue
            fields = line.rstrip().split("\t")
            samples = fields[9:]
            count_ones = samples.count("1")

            results.append(((file_line_num-4), count_ones))

    # Output
    #for line_num, ones in results:
        #print(line_num, ones)
    print(results)
    return results

def reverse_checking():
    #print(f"NORM_ERROR: {NORM_ERROR}")
    error = 0
    results = decode_vcf()
    og_trace = load_trace(CHAMELEON_PATH, TRACE_PATH, TRACE_SAMPLES, START_SAMPLE)
    for line_num, ones in results:
       test_trace = denormalize_trace(ones, NORM_FACTOR, OFFSET)
       if ((test_trace <= (og_trace[line_num] + NORM_ERROR)) and (test_trace >= (og_trace[line_num] - NORM_ERROR))):    #Correct
        print(f"Line: {line_num} |{og_trace[line_num]-NORM_ERROR} < {test_trace} < {og_trace[line_num]+NORM_ERROR}")
        #print(f"Line: {line_num} | VCF_trace: {test_trace} | Original Trace: {og_trace[line_num]}")
       else:                                                                                                                        #Wrong
        print(f"Error: {line_num} |Original Trace: {og_trace[line_num]} | Calculated Trace: {test_trace} | Margin of Error: +/-{NORM_ERROR}") 
        error+=1 
    print(f"Num of total errors: {error}")
    #decode(bin_mat, norm_factor):
    #denormalize_trace(norm_trace, norm_factor, offset=0):

def main():
    #print("Hello")
    #decode_vcf()
    reverse_checking()
  
if __name__ == "__main__":
    main()
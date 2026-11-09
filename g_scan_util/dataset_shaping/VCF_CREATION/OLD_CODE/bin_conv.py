import h5py
import numpy as np
import random

# Dataset paths
#INPUT_PATH = "./chameleon_norm/chameleon_rd_chunk_1_normalized.h5"#OUTPUT_FILE = "chameleon_base_chunk_1_normalized.h5"
#INPUT_PATH = "./chameleon_norm/AES_5000_chameleon_rd_chunk_1_normalized.h5"
INPUT_PATH = "./chameleon_norm/GP_5000_chameleon_rd_chunk_1_normalized.h5"
#INPUT_PATH = "./chameleon_norm/GP_1000_chameleon_rd_chunk_1_normalized.h5"

#OUTPUT_PATH = "AES_5000_chameleon.vcf"
#OUTPUT_PATH = "GP_5000_chameleon.vcf"
OUTPUT_PATH = "AES_1000_chameleon.vcf"
#OUTPUT_PATH = "GP_1000_chameleon.vcf"

TRACE_PATH = "trace"#data/traces/trace_0"  #int16

TRACE_ZOOM = 0
START_SAMPLE = 7565
SAMPLES = 100

# def load_trace(path, trace_path):
#     with h5py.File(path, "r") as f:
#         trace = f[trace_path][0]   # single long trace
#     return trace

# def load_trace(path, trace_path, zoom_trace):
#     with h5py.File(path, "r") as f:
#         if(zoom_trace!=0):                                  #If zoom_trace is NOT 0 load only that part of traces - Used for debugging purposes
#             trace = f[trace_path][:zoom_trace]              # Load Trace
#             print(f"First 10 Samples: {trace[:10]}")        #for debug
#         else:
#             trace = f[trace_path][:]                        # Load FULL Trace if zoom_trace = 0
#             print(f"First 10 Samples: {trace[:10]}")        #for debug
#     return trace

def load_trace(path, trace_path, zoom_trace=0):
    with h5py.File(path, "r") as f:
        dset = f[trace_path]

        if zoom_trace != 0:                                 #If zoom_trace is NOT 0 load only that part of traces - Used for debugging purposes
            #end_sample = start_sample + zoom_trace          # Load Trace
            trace = dset[:zoom_trace]
        else:
            trace = dset[:]

        # Debug
        print(f"First 10 Samples: {trace[:10]}")            # Load FULL Trace if zoom_trace = 0

    return trace    

# Placeholder function for converting a number to binary/Gray code
def encode(n):
    """
    Input sample
    Output Series of 1s followed by 0's
    """
    
    num_of_ones = round(n * SAMPLES)
    #print(f"Num of One's: {num_of_ones}")
    num_of_zeroes = SAMPLES - num_of_ones
    #print(f"Num of Zero's: {num_of_zeroes}")
    
    bin_mat = [0] * SAMPLES
    
    for i in range(num_of_ones):
        bin_mat[i] = 1

    for i in range(num_of_zeroes):
        bin_mat[i+num_of_ones] = 0

    #print(bin_mat)
    return bin_mat


def write_vcf(
    binary_matrix,
    ref_alleles,
    alt_alleles,
    start_pos,
    chrom="1",
    output_file="binary_matrix.vcf"
):
    """
    binary_matrix: list of lists (variants x samples), values in {0,1,None}
    ref_alleles:   list of REF alleles per variant
    alt_alleles:   list of ALT alleles per variant
    start_pos:     integer starting genomic position
    chrom:         chromosome label
    """

    num_variants = len(binary_matrix)
    num_samples = len(binary_matrix[0])

    with open(output_file, "w") as f:
        # Header
        f.write("##fileformat=VCFv4.2\n")
        f.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        )

        # Column header
        sample_headers = "\t".join([f"sample{i+1}" for i in range(num_samples)])
        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + sample_headers + "\n"
        )

        # Data rows
        for i in range(num_variants):
            pos = start_pos + i

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

            for val in binary_matrix[i]:
                if val is None:
                    row.append(".")
                else:
                    row.append(str(val))

            f.write("\t".join(row) + "\n")
    




def main():
    
    trace = load_trace(INPUT_PATH, TRACE_PATH, TRACE_ZOOM)
    #offset ALREADY Handled //trace = trace + 0.09   #for offset normalization
    binary_matrix = [encode(n) for n in trace]
   
    #print(binary_matrix)

    ref = ["A","C","G","T"]
    alt = ["G","T","A","C"]

    write_vcf(
        binary_matrix=binary_matrix,
        ref_alleles=ref,
        alt_alleles=alt,
        start_pos=START_SAMPLE+1,
        output_file=OUTPUT_PATH
    )

if __name__ == "__main__":
    main()
    


# Generate the binary matrix
# binary_matrix = [encode(n) for n in ]
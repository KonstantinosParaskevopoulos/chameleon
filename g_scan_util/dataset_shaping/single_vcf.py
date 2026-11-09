from Gen_Scan import *
import argparse
from pathlib import Path

LOCAL_PATH = "/home/konstantinos/Documents/GitHub/Genome-Scanning"
# Trace paths


# Parameters for Normalization
#OFFSET = 0.0            # Variable to fix trace offset
TRACE_MAX = 2047        # Max Theoretical Sample Value
TRACE_MIN = -2048       # Min Theoretical Sample Value


def single_vcf():
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Genome Scanning VCF generator")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable sanity checks and debug output"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix of output file"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Export Metadata to a txt file named the same as the vcf"
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        default=f"{LOCAL_PATH}/Datasets/BASE/chameleon_base_chunk_1.h5",
        help="Path to the chameleon file"
    )
    parser.add_argument(
        "--trace_samples",
        type=int,
        default=1000,
        help="Number of trace samples to load"
    )
    parser.add_argument(
        "--start_sample",
        type=int,
        default=0,
        help="Starting point"
    )
    parser.add_argument(
        "--vcf_samples",
        type=int,
        default=100,
        help="Number of VCF samples per trace"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset value for normalization"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=f"{LOCAL_PATH}/VCF_CREATION/Testing_VCFs",
        help="Offset value for normalization"
    )
    parser.add_argument(
        "--ID",
        type=int,
        default=0,
        help="Trace / Metadata ID Number"
    )
    args = parser.parse_args()
    
    DEBUG           = args.debug            # Debug Flag, Shows statistics
    PREFIX          = args.prefix           # Prefix, Output file prefix
    METADATA        = args.metadata         # Metadata Flag, exports metadata to txt file
    CHAMELEON_PATH  = args.input_h5         # Chameleon Dataset Path
    TRACE_SAMPLES   = args.trace_samples    # Number of Samples to load
    VCF_SAMPLES     = args.vcf_samples      # Number of VCF samples per Trace sample
    OFFSET          = args.offset           # Offset for correction 
    START_SAMPLE    = args.start_sample     # Starting point of traces to load
    SAVE_PATH       = args.save_folder      # Specify output folder path
    ID              = args.ID               # Trace / Metadata ID Number
    #LOCAL_PATH
    #path = "/home/megauser/chameleon/traces/RD/RD/chameleon_rd_chunk_1.h5"
    METADATA_PATH = f"metadata/pinpoints/pinpoints_{ID}"  #int16  Change for different traces
    TRACE_PATH = f"data/traces/trace_{ID}"  #int16  Change for different traces
    TRACE_PATH_NORMALIZED = "trace"     #int16

    chunk_id = int(CHAMELEON_PATH.split("_chunk_")[1].split(".")[0])
    #chameleon_base 1.h5
    
    filename = os.path.basename(CHAMELEON_PATH)
    name_no_ext = os.path.splitext(filename)[0] # Save input filename without extension
    countermeasure = str(name_no_ext.split("_chunk_")[0].split("_")[1]).upper()
    end_sample = START_SAMPLE + TRACE_SAMPLES
    tracename = os.path.basename(TRACE_PATH)
    #TRACE_NUMBER = os.path.splitext(tracename)[0] # Save trace number without extension
    
    #Namefile Convention {User Specified prefix}_VCF_x{number of samples per trace}_{starting sample}-{ending sample}_o{OFFSET}_{trace_file}_{trace_number}.vcf
    VCF_PATH = f"{SAVE_PATH}/{PREFIX}_VCF_x{VCF_SAMPLES}_{START_SAMPLE}-{end_sample}_o{OFFSET}_C-{countermeasure}-{chunk_id}_T-{ID}.vcf"  # VCF Output File path
    METADATA_txt_PATH = f"{SAVE_PATH}/{PREFIX}_VCF_x{VCF_SAMPLES}_{START_SAMPLE}-{end_sample}_o{OFFSET}_C-{countermeasure}-{chunk_id}_T-{ID}.txt"  # metadata Output File path
    #OUTPUT_PATH = 
    
    folder = Path(SAVE_PATH) / "chameleon_norm"
    folder.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = folder / (f"{PREFIX}_normalized_x{VCF_SAMPLES}_{START_SAMPLE}-{end_sample}_o{OFFSET}_C-{countermeasure}-{chunk_id}_T-{ID}.h5")  # Normalized Trace Output path)

    # Load and normalize trace
    trace = load_trace(CHAMELEON_PATH, TRACE_PATH, TRACE_SAMPLES, START_SAMPLE)
    
    trace_norm = normalize_trace(trace, OFFSET)
    
    # Save normalized trace to file
    save_trace(OUTPUT_PATH, trace_norm)
#####################################################################################    
    # Debugging and sanity checks if --debug flag is set
    if DEBUG:
        print(f"Original min/max: {trace.min()} / {trace.max()}")
        print(f"Offseted min/max: {trace.min()+OFFSET} / {trace.max()+OFFSET}")
        print(f"Normalized min/max: {trace_norm.min()} / {trace_norm.max()}")
        #check_adc_range(trace)
        check_normalized_bounds(trace_norm)
        check_endpoint_mapping()
        check_zero_mapping()
        check_statistics(trace, trace_norm)
        print("\n\nWARNING: Didn't create VCF File -> DEBUG MODE!")
        del trace # Delete trace from memory

    else:   
        if((trace.min()+OFFSET)<TRACE_MIN):
            print("ERROR! Abort process! Offseted value LESS than TRACE_MIN")
            exit()
        if((trace.max()+OFFSET)>TRACE_MAX):
            print("ERROR! Abort process! Offseted value MORE than TRACE_MAX")
            exit()
        del trace # Delete trace from memory
        #don't execute anything in debug mode
        #trace = load_trace(INPUT_PATH, TRACE_PATH, TRACE_ZOOM)
        ##Create a loop to load X lines of the trace 
        f = init_vcf(VCF_PATH,VCF_SAMPLES,buffer_mb=8)
        ####
        if(TRACE_SAMPLES!=0):
            for i in range(TRACE_SAMPLES):
                append_vcf_row(
                    f = f,
                    binary_row = encode(trace_norm[i], VCF_SAMPLES),
                    pos = START_SAMPLE + i + 1 #i + 1 to start from 1
                )
                if(i%FLUSH_EVERY==0):
                    flush_buffer(f)

                # Final flush + close
        else:
            total_sample_num = len(trace_norm)
            #print(total_sample_num)
            for i in range(total_sample_num):
                append_vcf_row(
                    f = f,
                    binary_row = encode(trace_norm[i], VCF_SAMPLES),
                    pos = START_SAMPLE + i + 1
                )
                if(i%FLUSH_EVERY==0):
                    flush_buffer(f)
        f.flush()
        os.fsync(f.fileno())
        f.close()
        if(METADATA):
            pinpoints = load_md(CHAMELEON_PATH, METADATA_PATH, METADATA_txt_PATH, TRACE_SAMPLES, START_SAMPLE)

if __name__ == "__main__":
    single_vcf()


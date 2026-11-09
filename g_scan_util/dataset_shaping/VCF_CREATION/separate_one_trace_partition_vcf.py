#python -m vcf_creation --countermeasure BASE --vcf_samples 5 --offset 0
#from dataclasses import dataclass
from single_vcf import single_vcf
from Gen_Scan import load_md
import numpy as np # type: ignore
import os
import random
import scipy    # type: ignore
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from tqdm import tqdm # type: ignore
import shutil



## subprocess.run(["python", "script_b.py", "--input", "data.h5"]) ## SUBPROCESS EXAMPLE
RUN             = "python"
MODULE          = "-m"
NAME            = "single_vcf"
VCF_F           = "--vcf_samples"
PREFIX_F        = "--prefix"
METADATA_F      = "--metadata"
SAVEFOLDER_F    = "--save_folder"
ST_SMPL_F       = "--start_sample"
TRC_SMPL_F      = "--trace_samples"
INPUT_F         = "--input_h5"
OFFSET_F        = "--offset"
ID_F            = "--ID"

#  --trace_samples 383144 --input_h5 /home/konstantinos/Documents/GitHub/Genome-Scanning/Datasets/BASE/chameleon_base_chunk_1.h5

DATASET_PARENT_FOLDER = f"/local/scratch/paraskevopoulosk/Datasets" # PATH TO DATASETS
OUTPUT_FOLDER = f"/local/scratch/paraskevopoulosk/Datasets" # PATH TO OUTPUT

TRACE_LENGTH = 134_217_550 # Same for all traces

#############################################
############ Limits for Testing #############
#############################################
CHUNK_LIMIT = 1                 #############
TRACE_PER_CHUNK_LIMIT = 1       #############
WORKERS_LIMIT = 8               #############
#############################################
#############################################
#############################################

TRACES_PER_CHUNK_MAP = { # How many Traces per Chunk in each Countermeasure 
    "BASE"  : 16,
    "DFS"   : 16,
    "RD"    : 32,
    "MRP"   : 32,
    "CHF"   : 64,
}

# def create_unique_countermeasure_folder(countermeasure):
#     base_path = Path(OUTPUT_FOLDER)
#     base_name = countermeasure

#     root = base_path / base_name
#     stamp = datetime.now().strftime("%Y-%m-%d")#_%H%M%S")
#     candidate = root / f"test_{countermeasure}_unzipped"#{stamp}"
#     index = 1

#     while candidate.exists():
#         candidate = base_path / base_name / f"test_{countermeasure}_unzipped_{index}"
#         index += 1

#     candidate.mkdir(parents=True, exist_ok=False)
#     return candidate

def create_countermeasure_folder(countermeasure):
    base_path = Path(OUTPUT_FOLDER)
    root = base_path / countermeasure
    candidate = root / f"test_{countermeasure}_unzipped"

    if candidate.exists():
        shutil.rmtree(candidate)

    candidate.mkdir(parents=True, exist_ok=True)

    return candidate

def getNumberOfFiles(traces_folder):  
    ##  This function counts and returns the NUMBER_OF_CHUNKS in the countermeasure folder
    
    traces_files = np.sort(
        [traces_folder + f for f in os.listdir(traces_folder) if f.endswith('.h5')]).tolist()

    NUMBER_OF_CHUNKS = len(traces_files) # Number of Chunk Files detected in folder
    
    #NO_FILE_FOUND and LESS_THAN_EXPECTED checks
    assert NUMBER_OF_CHUNKS != 0, f"No Chunk_file found for this Countermeasure {traces_folder}"
    assert NUMBER_OF_CHUNKS >= CHUNK_LIMIT, f"NUMBER_OF_CHUNKS is less than LIMIT. There have to be at least {CHUNK_LIMIT} chunks!"
    
    return NUMBER_OF_CHUNKS


# def highpass(traces: np.ndarray,        ### HIGHPASS FILTER
#              Wn: float = 0.001) -> np.ndarray:
#     b, a = scipy.signal.butter(3, Wn)
#     y = scipy.signal.filtfilt(b, a, traces).astype(np.float32)

#     return (traces - y).astype(np.float32)

## READ METADATA OF TRACE ##
#   def load_md(path, md_path, txt_path, zoom_trace=0, start_sample=0):

def create_VCF_single_trace(trace_id, chunk_id, countermeasure, vcf_samples, offset, save_folder):
    path_h5 = f"{DATASET_PARENT_FOLDER}/{countermeasure}/chameleon_{countermeasure.lower()}_chunk_{chunk_id}.h5"
    trace_path = f"data/traces/trace_{trace_id}"
    md_path = f"metadata/pinpoints/pinpoints_{trace_id}"
    TRACES_PER_CHUNK = TRACES_PER_CHUNK_MAP[countermeasure] - 1
    pinpoints = load_md(path_h5, md_path, f"{save_folder}/pinpoints_T{trace_id}_C{chunk_id}.txt", 0, 0)  # First load all pinpoints to have all AES Regions and be able to calculate GP(RAND) Regions
    

    #Normal model for a trace:  GP(0) - AES(0) - GP(1) - AES(1) - ... - AES(n-1) - GP(n-1)
    print(f"Processing: Trace {trace_id}/{TRACES_PER_CHUNK}, Chunk {chunk_id}/16\n")
    
    startVCF = 0
    endVCF = (pinpoints[1]['start'])-1
    VCF_region_length = endVCF-startVCF
    ### Extract VCF from that region
    subprocess.run([
        RUN, 
        MODULE, 
        NAME, 
        VCF_F, str(vcf_samples), 
        PREFIX_F, f"TARGET_{0}",
        METADATA_F, 
        SAVEFOLDER_F, save_folder, 
        ST_SMPL_F, str(startVCF), 
        TRC_SMPL_F, str(VCF_region_length), 
        INPUT_F, path_h5, 
        OFFSET_F, str(offset), 
        ID_F, str(trace_id)
    ]) 
        #print(f"\tProcessing AES {0}/{len(pinpoints)-1}, T:{trace_id} C:{chunk_id}\n")

                
    iter_start_point = 1 

    ## MAIN LOOP ##
    for i in tqdm( range(iter_start_point, len(pinpoints)), desc="Trace Partitioning" ):  #range(iter_start_point, len(pinpoints)):
        
        startVCF = pinpoints[i-1]['end'] + 1
        ### Extract VCF from that region
        if(i < (len(pinpoints)-1)): # if pinpoints[i+1] exists | Last CO is pinpoints[len(pinpoints)-1]
            endVCF = pinpoints[i+1]['start'] - 1
        else:
            endVCF = TRACE_LENGTH - 1
        
        VCF_region_length = endVCF-startVCF
        subprocess.run([
                RUN, 
                MODULE, 
                NAME, 
                VCF_F, str(vcf_samples), 
                PREFIX_F, f"TARGET_{i}", 
                METADATA_F,
                SAVEFOLDER_F, save_folder, 
                ST_SMPL_F, str(startVCF), 
                TRC_SMPL_F, str(VCF_region_length), 
                INPUT_F, path_h5, 
                OFFSET_F, str(offset), 
                ID_F, str(trace_id)
            ]) 

        #print(f"\tProcessing AES {i}/{len(pinpoints)-1}, T:{trace_id} C:{chunk_id}\n")
    print(
        f"[PID {os.getpid()}] "
        f"Countermeasure={countermeasure} "
        f"Chunk={chunk_id} Trace={trace_id}"
    )

 
    
    
def VCF_creation_one_chunk(chunk_id, countermeasure, vcf_samples, offset, save_folder, trace_id):   
    path_h5 = f"{DATASET_PARENT_FOLDER}/{countermeasure}/chameleon_{countermeasure.lower()}_chunk_{chunk_id}.h5"
    
    TRACES_PER_CHUNK = TRACES_PER_CHUNK_MAP[countermeasure]
    
    print(f"[PID {os.getpid()}] Processing Chunk [{chunk_id}] : {path_h5}")

    # if((TRACE_PER_CHUNK_LIMIT<=TRACES_PER_CHUNK) and (TRACE_PER_CHUNK_LIMIT>0)):    #For Debugging cases where we don't want to convert all traces in a chunk LIMIT must be 
    #                                                                                                                                                         #BIGGER THAN 0 
    #                                                                                                                                                         #SMALLER THAN TPC
    #     loop_limit = TRACE_PER_CHUNK_LIMIT
    # else:
    #     loop_limit = TRACES_PER_CHUNK
    
    #for i in range(loop_limit):
    create_VCF_single_trace(trace_id, chunk_id, countermeasure, vcf_samples, offset, save_folder)
    
    return f"Chunk: {chunk_id}  Trace: {trace_id} finished"

# def VCF_creation_full(countermeasure):
    
#     traces_full_folder = f"{DATASET_PARENT_FOLDER}/{countermeasure}"

#     traces_files = getNumberOfFiles(traces_full_folder)
#     ##{DATASET_PARENT_FOLDER}/{countermeasure}/chameleon_{countermeasure.lower()}_chunk_{i+1}.h5
#     for i in range(traces_files):
#         chunk_id = i+1
#         VCF_creation_one_chunk(chunk_id, countermeasure)
#     ###############

def VCF_creation_full(countermeasure, vcf_samples, offset, save_folder, trace_id): #Parallel Execution for each chunk

    traces_full_folder = f"{DATASET_PARENT_FOLDER}/{countermeasure}"

    traces_files = getNumberOfFiles(traces_full_folder)

    if((CHUNK_LIMIT<=traces_files) and (CHUNK_LIMIT>0)):    #For Debugging cases where we don't want to convert all traces in a chunk LIMIT must be 
                                                                                                                                                            #BIGGER THAN 0 
                                                                                                                                                            #SMALLER THAN TPC
        loop_limit = CHUNK_LIMIT
    else:
        loop_limit = traces_files

    chunk_ids = [i + 1 for i in range(loop_limit)]


    CPU_COUNT = os.cpu_count()
    if CPU_COUNT >= WORKERS_LIMIT:
        workers = 8 #os.cpu_count()
    else:
        workers = os.cpu_count()

    print(f"Detected {loop_limit}|{traces_files} chunks")
    print(f"Using {workers} workers\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:

        results = executor.map(
            VCF_creation_one_chunk,
            chunk_ids,
            [countermeasure] * len(chunk_ids),
            [vcf_samples] * len(chunk_ids), 
            [offset] * len(chunk_ids), 
            [save_folder] * len(chunk_ids),
            [trace_id] * len(chunk_ids)
        )

        for r in results:
            print(r)

    #start0 = pinpoints[0]['start']
    #end0   = pinpoints[0]['end']
    #for p in pinpoints:
    #   print(p['start'], p['end'])

def main():
    #Create Folders

    #Arg Parser
    ##########################################################################
    # Argument parsing
    parser = argparse.ArgumentParser(description="Genome Scanning VCF generator")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable sanity checks and debug output"
    )
    parser.add_argument(
        "--countermeasure",
        type=str,
        default="BASE",
        help="Type of Countermeasure"
    )
    # parser.add_argument(
    #     "--metadata",
    #     action="store_true",
    #     help="Export Metadata to a txt file named the same as the vcf"
    # )
    # parser.add_argument(
    #     "--input_h5",
    #     type=str,
    #     default=f"{LOCAL_PATH}Datasets/BASE/chameleon_base_chunk_1.h5",
    #     help="Path to the chameleon file"
    # )
    # parser.add_argument(
    #     "--trace_samples",
    #     type=int,
    #     default=1000,
    #     help="Number of trace samples to load"
    # )
    # parser.add_argument(
    #     "--start_sample",
    #     type=int,
    #     default=0,
    #     help="Starting point"
    # )
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
    # parser.add_argument(
    #     "--save_folder",
    #     type=str,
    #     default=f"{OUTPUT_FOLDER}Testing_VCFs",
    #     help="Save Folder"
    # )
    parser.add_argument(
        "--trace_ID",
        type=int,
        default=0,
        help="Trace ID"
    )
    args = parser.parse_args()
    
    DEBUG           = args.debug            # Debug Flag, Shows statistics
    COUNTERMEASURE  = args.countermeasure   # Prefix, Output file prefix
    COUNTERMEASURE  = COUNTERMEASURE.upper()# Convert to Upper Case
    #METADATA       = args.metadata         # Metadata Flag, exports metadata to txt file
    #CHAMELEON_PATH = args.input_h5         # Chameleon Dataset Path
    #TRACE_SAMPLES  = args.trace_samples    # Number of Samples to load
    VCF_SAMPLES     = args.vcf_samples      # Number of VCF samples per Trace sample
    OFFSET          = args.offset           # Offset for correction 
    #START_SAMPLE    = args.start_sample     # Starting point of traces to load
    #SAVE_PATH       = args.save_folder      # Specify output folder path
    ID              = args.trace_ID         # Specify which trace of the dataset to convert
    
    folder = create_countermeasure_folder(COUNTERMEASURE)

    VCF_creation_full(COUNTERMEASURE, VCF_SAMPLES, OFFSET, folder, ID)

    print(f"\n\nOutput Saved at: {folder} !")

if __name__ == "__main__":
    main()
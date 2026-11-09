import os
from concurrent.futures import ProcessPoolExecutor

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATASET_PARENT_FOLDER = "/path/to/dataset"
TRACES_PER_CHUNK = 16


# -------------------------------------------------------
# USER FUNCTION
# -------------------------------------------------------
def create_VCF_single_trace(trace_id, chunk_id, countermeasure):
    """
    Your real implementation here
    """
    print(
        f"[PID {os.getpid()}] "
        f"Countermeasure={countermeasure} "
        f"Chunk={chunk_id} Trace={trace_id}"
    )


# -------------------------------------------------------
# PROCESS ONE CHUNK (RUNS IN PARALLEL)
# -------------------------------------------------------
def VCF_creation_one_chunk(chunk_id, countermeasure):

    path = (
        f"{DATASET_PARENT_FOLDER}/{countermeasure}/"
        f"chameleon_{countermeasure.lower()}_chunk_{chunk_id}.h5"
    )

    print(f"[PID {os.getpid()}] Processing {path}")

    for i in range(TRACES_PER_CHUNK):
        create_VCF_single_trace(i, chunk_id, countermeasure)

    return f"Chunk {chunk_id} finished"


# -------------------------------------------------------
# COUNT FILES
# -------------------------------------------------------
def getNumberOfFiles(folder):
    return len([
        f for f in os.listdir(folder)
        if f.endswith(".h5")
    ])


# -------------------------------------------------------
# MAIN PARALLEL FUNCTION
# -------------------------------------------------------
def VCF_creation_full(countermeasure):

    traces_full_folder = f"{DATASET_PARENT_FOLDER}/{countermeasure}"

    traces_files = getNumberOfFiles(traces_full_folder)

    chunk_ids = [i + 1 for i in range(traces_files)]

    workers = os.cpu_count()

    print(f"Detected {traces_files} chunks")
    print(f"Using {workers} workers\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:

        results = executor.map(
            VCF_creation_one_chunk,
            chunk_ids,
            [countermeasure] * len(chunk_ids)
        )

        for r in results:
            print(r)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    VCF_creation_full("BASE")
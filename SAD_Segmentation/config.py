# Edit these settings before running.

HDF5_FILE = "/path/to/dataset.h5"
OUTPUT_DIR = "./sad_results"

# HDF5 layout
TRACES_PATH = "data/traces"
PINPOINTS_PATH = "metadata/pinpoints"
TRACE_PREFIX = "trace_"
PINPOINT_PREFIX = "pinpoints_"

# Trace used to build the SAD reference.
REFERENCE_TRACE_INDEX = 0
REFERENCE_PINPOINT_INDEX = 0

# Number of samples used as the SAD template.
REFERENCE_LENGTH = 256

# Offset of the template from the beginning of the reference CO.
# Example: 0 uses the first 256 samples of the CO.
REFERENCE_OFFSET = 0

# Search speed/precision:
# Coarse stride 10 checks every tenth sample, then refines around the best hit.
COARSE_STRIDE = 10
BATCH_SIZE = 4096

# Optional rejection threshold for the best SAD score.
# Leave as None to always return the best match.
MAX_SAD = None

# Skip the reference trace when producing output files.
SKIP_REFERENCE_TRACE = False

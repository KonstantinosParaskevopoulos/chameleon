SAD-based offline CO localisation
==================================

Requirements
------------
- Python 3
- numpy
- h5py

Install:
    pip install numpy h5py

Run
---
1. Edit config.py:
   - HDF5_FILE
   - OUTPUT_DIR
   - HDF5 group names/prefixes if needed
   - reference trace index and template settings

2. Run:
       python main.py

Output
------
One file per successfully matched trace:
    sad_results/trace_1.txt

Each file contains exactly one line:
    start<TAB>end

Coordinates use the same convention as the HDF5 ground truth:
[start, end), where end is exclusive.

Method and assumptions
----------------------
A short reference segment is extracted from a known CO in one reference
trace. SAD locates that segment in each other trace. The full CO boundaries
are reconstructed using the segment's offset and the reference CO length.

Therefore, this simple implementation assumes:
- one CO per trace;
- the CO has the same duration in all traces;
- the chosen reference segment is sufficiently distinctive;
- the reference segment remains recognisable across traces.

The default coarse stride is 10. The best coarse candidate is refined
sample-by-sample within one stride on either side. Reduce COARSE_STRIDE
to 1 for an exhaustive search, at a potentially higher computational cost.

MAX_SAD is disabled by default. Set it to an integer if you want to reject
matches whose best SAD score is too high.

The program skips the reference trace by default. Set
SKIP_REFERENCE_TRACE = False to produce an output for it as well.

The evaluation uses the stored ground-truth pinpoints but does not use them
to locate the CO in each target trace.

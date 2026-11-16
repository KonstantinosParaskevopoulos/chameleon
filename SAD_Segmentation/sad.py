import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _scores_for_positions(windows, reference, positions, batch_size):
    best_score = None
    best_position = None
    reference = reference.astype(np.int32, copy=False)

    for offset in range(0, len(positions), batch_size):
        batch_positions = positions[offset:offset + batch_size]
        batch = windows[batch_positions].astype(np.int32, copy=False)

        # int64 accumulation avoids overflow for long templates.
        scores = np.abs(batch - reference).sum(axis=1, dtype=np.int64)
        local_index = int(np.argmin(scores))
        local_score = int(scores[local_index])

        if best_score is None or local_score < best_score:
            best_score = local_score
            best_position = int(batch_positions[local_index])

    return best_position, best_score


def find_best_match(trace, reference, search_start=0, search_end=None,
                    coarse_stride=10, batch_size=4096):
    """
    Find the lowest-SAD occurrence of reference in trace.

    search_end is the exclusive upper bound for the template's start position.
    Coarse search is followed by a sample-by-sample local refinement.
    """
    trace = np.asarray(trace)
    reference = np.asarray(reference)

    if trace.ndim != 1 or reference.ndim != 1:
        raise ValueError("Trace and reference must be one-dimensional.")
    if len(reference) == 0 or len(reference) > len(trace):
        raise ValueError("Invalid reference length.")

    max_start = len(trace) - len(reference)
    if search_end is None:
        search_end = max_start + 1

    search_start = max(0, int(search_start))
    search_end = min(max_start + 1, int(search_end))

    if search_start >= search_end:
        raise ValueError("No valid positions to search.")

    stride = max(1, int(coarse_stride))
    windows = sliding_window_view(trace, len(reference))

    coarse_positions = np.arange(search_start, search_end, stride, dtype=np.int64)
    coarse_best, _ = _scores_for_positions(
        windows, reference, coarse_positions, batch_size
    )

    # Refine around the best coarse position, including its neighbours.
    refine_start = max(search_start, coarse_best - stride)
    refine_end = min(search_end, coarse_best + stride + 1)
    fine_positions = np.arange(refine_start, refine_end, dtype=np.int64)

    return _scores_for_positions(windows, reference, fine_positions, batch_size)

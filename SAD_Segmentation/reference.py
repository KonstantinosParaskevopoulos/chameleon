import numpy as np


def build_reference(trace, co_start, co_end, reference_length, reference_offset=0):
    """
    Extract a short template from a known CO.

    reference_offset is measured from co_start.
    Returns (template, full_co_length).
    """
    co_length = co_end - co_start
    left = co_start + reference_offset
    right = left + reference_length

    if co_start < 0 or co_end > len(trace) or co_start >= co_end:
        raise ValueError("Invalid reference CO boundaries.")

    if reference_offset < 0 or right > co_end:
        raise ValueError("Reference segment falls outside the reference CO.")

    template = np.asarray(trace[left:right]).copy()
    return template, co_length

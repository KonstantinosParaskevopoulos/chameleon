from sad import find_best_match


def detect_co(trace, reference, co_length, reference_offset,
              coarse_stride=10, batch_size=4096, max_sad=None):
    """
    Locate the reference segment and reconstruct the full CO interval.
    Returns (start, end, best_sad_score).
    """
    # Constrain the template match so the reconstructed CO fits in the trace.
    search_start = reference_offset
    search_end = len(trace) - co_length + reference_offset + 1

    match_position, score = find_best_match(
        trace,
        reference,
        search_start=search_start,
        search_end=search_end,
        coarse_stride=coarse_stride,
        batch_size=batch_size
    )

    if max_sad is not None and score > max_sad:
        return None, None, score

    start = match_position - reference_offset
    end = start + co_length
    return start, end, score

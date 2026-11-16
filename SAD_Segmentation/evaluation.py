def boundary_errors(pred_start, pred_end, true_start, true_end):
    return abs(pred_start - true_start), abs(pred_end - true_end)


def print_evaluation(rows):
    """Print mean absolute start/end error for successfully detected traces."""
    if not rows:
        print("No traces were evaluated.")
        return

    start_errors = [row[0] for row in rows]
    end_errors = [row[1] for row in rows]

    print(f"Evaluated traces: {len(rows)}")
    print(f"Mean absolute start error: {sum(start_errors) / len(start_errors):.2f} samples")
    print(f"Mean absolute end error:   {sum(end_errors) / len(end_errors):.2f} samples")

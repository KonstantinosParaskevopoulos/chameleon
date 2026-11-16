from pathlib import Path

import config
from dataset import HDF5Dataset
from reference import build_reference
from detection import detect_co
from evaluation import boundary_errors, print_evaluation
from output import write_boundaries


def main():
    with HDF5Dataset(
        config.HDF5_FILE,
        config.TRACES_PATH,
        config.PINPOINTS_PATH,
        config.TRACE_PREFIX,
        config.PINPOINT_PREFIX
    ) as dataset:

        if len(dataset) == 0:
            raise RuntimeError("No traces found in the configured HDF5 group.")

        # Build one reference from a trace with known CO boundaries.
        ref_trace = dataset.trace(config.REFERENCE_TRACE_INDEX)
        ref_start, ref_end = dataset.pinpoint(
            config.REFERENCE_TRACE_INDEX,
            config.REFERENCE_PINPOINT_INDEX
        )
        reference, co_length = build_reference(
            ref_trace,
            ref_start,
            ref_end,
            config.REFERENCE_LENGTH,
            config.REFERENCE_OFFSET
        )

        print(f"Reference CO length: {co_length} samples")
        print(f"Reference template length: {len(reference)} samples")

        evaluation_rows = []

        for index in range(len(dataset)):
            if config.SKIP_REFERENCE_TRACE and index == config.REFERENCE_TRACE_INDEX:
                continue

            trace = dataset.trace(index)
            start, end, score = detect_co(
                trace,
                reference,
                co_length,
                config.REFERENCE_OFFSET,
                config.COARSE_STRIDE,
                config.BATCH_SIZE,
                config.MAX_SAD
            )

            if start is None:
                print(f"Trace {index}: no match accepted (best SAD={score})")
                continue

            output_file = write_boundaries(
                config.OUTPUT_DIR, index, start, end
            )
            print(
                f"Trace {index}: {start}\t{end} "
                f"(SAD={score}, output={output_file})"
            )

            # Ground-truth pinpoints are used only for evaluation.
            true_start, true_end = dataset.pinpoint(index, 0)
            evaluation_rows.append(
                boundary_errors(start, end, true_start, true_end)
            )

        print_evaluation(evaluation_rows)


if __name__ == "__main__":
    main()

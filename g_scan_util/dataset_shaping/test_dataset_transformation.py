
import h5py
from tqdm import tqdm
from pathlib import Path
import argparse


def copy_attrs(src, dst):
    """Copy HDF5 attributes."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def create_test_dataset(input_path, output_path, N):
    """
    Create a smaller dataset containing the first N transformed
    traces and their corresponding metadata.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as src:

        total_traces = len(src["data/traces"])

        if not 0 < N <= total_traces:
            raise ValueError(
                f"N must be between 1 and {total_traces}, got {N}"
            )

        with h5py.File(output_path, "w") as dst:

            # Preserve root attributes
            copy_attrs(src, dst)

            # Recreate the original hierarchy
            data_group = dst.create_group("data")
            traces_group = data_group.create_group("traces")

            metadata_group = dst.create_group("metadata")
            pinpoints_group = metadata_group.create_group("pinpoints")
            ciphers_group = metadata_group.create_group("ciphers")

            # Preserve group attributes
            copy_attrs(src["data"], data_group)
            copy_attrs(src["data/traces"], traces_group)
            copy_attrs(src["metadata"], metadata_group)
            copy_attrs(src["metadata/pinpoints"], pinpoints_group)
            copy_attrs(src["metadata/ciphers"], ciphers_group)

            # Copy the first N traces and corresponding metadata
            for i in tqdm(range(N), desc="Creating test dataset"):

                trace_name = f"trace_{i}"
                pinpoint_name = f"pinpoints_{i}"
                cipher_name = f"ciphers_{i}"

                src.copy(
                    src[f"data/traces/{trace_name}"],
                    traces_group,
                    name=trace_name
                )

                src.copy(
                    src[f"metadata/pinpoints/{pinpoint_name}"],
                    pinpoints_group,
                    name=pinpoint_name
                )

                src.copy(
                    src[f"metadata/ciphers/{cipher_name}"],
                    ciphers_group,
                    name=cipher_name
                )

    print(f"Original subtraces: {total_traces}")
    print(f"Test subtraces: {N}")
    print(f"Output: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create a smaller Chameleon HDF5 test dataset."
    )

    parser.add_argument(
        "-n", "--num-traces",
        type=int,
        required=True,
        help="Number of initial subtraces to retain."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default=(
            "/local/scratch/paraskevopoulosk/Datasets/BASE/CHAM_IN/"
            "chameleon_base_chunk_1.h5"
        ),
        help="Input transformed HDF5 dataset."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=(
            "/local/scratch/paraskevopoulosk/Datasets/BASE/"
            "chameleon_base_chunk_1_test.h5"
        ),
        help="Output test HDF5 dataset."
    )

    args = parser.parse_args()

    create_test_dataset(
        args.input,
        args.output,
        args.num_traces
    )

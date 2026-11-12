import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

countermeasure = "base"
DATASET_DIR = f"/local/scratch/paraskevopoulosk/Datasets/{countermeasure.upper()}/"
CHAMELEON_FILE=f"{DATASET_DIR}chameleon_{countermeasure}_chunk_1.h5"
OUTPUT_FILE=f"{DATASET_DIR}chameleon_{countermeasure}_chunk_1_transformed.h5"

def copy_attrs(src, dst):
    """Copy HDF5 attributes."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def transform_chameleon_chunk(input_path, output_path):
    """
    Split every original Chameleon trace into subtraces containing
    exactly one cryptographic operation (CO), surrounded by the
    adjacent random regions.

    The original HDF5 hierarchy and naming conventions are preserved.
    Trace indices are reassigned globally and sequentially.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    total_subtraces = 0

    with h5py.File(input_path, "r") as src, \
         h5py.File(output_path, "w") as dst:

        # Preserve root attributes
        copy_attrs(src, dst)

        # Create the original hierarchy
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

        # Discover original trace indices and sort numerically
        trace_names = sorted(
            src["data/traces"].keys(),
            key=lambda name: int(name.split("_")[-1])
        )

        # Global index for the transformed traces
        new_idx = 0

        for trace_name in tqdm(trace_names, desc="Processing traces"):

            original_idx = int(trace_name.split("_")[-1])

            trace = src[f"data/traces/{trace_name}"]
            pinpoints = src[
                f"metadata/pinpoints/pinpoints_{original_idx}"
            ][:]

            cipher_group = src[
                f"metadata/ciphers/ciphers_{original_idx}"
            ]

            key = cipher_group["key"][:]
            plaintexts = cipher_group["plaintexts"][:]

            # Validate input
            if trace.ndim != 1:
                raise ValueError(
                    f"{trace_name} is not one-dimensional: {trace.shape}"
                )

            # if pinpoints.ndim != 2 or pinpoints.shape[1] != 2:
            #     raise ValueError(
            #         f"Unexpected pinpoint shape for {trace_name}: "
            #         f"{pinpoints.shape}. Expected (N, 2)."
            #     )

            if pinpoints.dtype.names != ("start", "end"):
                raise ValueError(
                    f"Unexpected pinpoint dtype for {trace_name}: "
                    f"{pinpoints.dtype}"
                )

            n_cos = len(pinpoints)

            if len(plaintexts) != n_cos:
                raise ValueError(
                    f"Mismatch in {trace_name}: "
                    f"{n_cos} COs but {len(plaintexts)} plaintexts."
                )

            trace_length = len(trace)

            # Process each CO independently
            for co_idx in range(n_cos):

                #s_i, e_i = map(int, pinpoints[co_idx])
                s_i = int(pinpoints[co_idx]["start"])
                e_i = int(pinpoints[co_idx]["end"])

                # Determine extraction boundaries
                # a_i = (
                #     0 if co_idx == 0
                #     else int(pinpoints[co_idx - 1][1]) + 1
                # )

                # b_i = (
                #     trace_length - 1 if co_idx == n_cos - 1
                #     else int(pinpoints[co_idx + 1][0]) - 1
                # )

                # a_i = (
                #     0 if co_idx == 0
                #     else int(pinpoints[co_idx - 1]["end"]) + 1
                # )
                

                # b_i = (
                #     trace_length - 1 if co_idx == n_cos - 1
                #     else int(pinpoints[co_idx + 1]["start"]) - 1
                # )
                a_i = (
                    0 if co_idx == 0
                    else int(pinpoints[co_idx - 1]["end"])
                )
                b_i = (
                    trace_length if co_idx == n_cos - 1
                    else int(pinpoints[co_idx + 1]["start"])
                )               

                # Validate boundaries
                # if not (0 <= a_i <= s_i <= e_i <= b_i < trace_length):
                #     raise ValueError(
                #         f"Invalid boundaries in {trace_name}, CO {co_idx}: "
                #         f"a={a_i}, s={s_i}, e={e_i}, b={b_i}"
                #     )
                if not (0 <= a_i <= s_i < e_i <= b_i <= trace_length):
                    raise ValueError(
                        f"Invalid boundaries in {trace_name}, CO {co_idx}: "
                        f"a={a_i}, s={s_i}, e={e_i}, b={b_i}"
                    )

                # Extract subtrace (inclusive endpoints)
                subtrace = trace[a_i:b_i]

                # Update pinpoint coordinates relative to subtrace
                # new_pinpoint = np.array(
                #     [[s_i - a_i, e_i - a_i]],
                #     dtype=pinpoints.dtype
                # )
                new_pinpoint = np.array([(s_i - a_i, e_i - a_i)], dtype=pinpoints.dtype)

                # Create transformed datasets
                new_trace_name = f"trace_{new_idx}"
                new_pinpoint_name = f"pinpoints_{new_idx}"
                new_cipher_name = f"ciphers_{new_idx}"

                traces_group.create_dataset(
                    new_trace_name,
                    data=subtrace,
                    dtype=trace.dtype
                )

                pinpoints_group.create_dataset(
                    new_pinpoint_name,
                    data=new_pinpoint,
                    dtype=pinpoints.dtype
                )

                # Create corresponding cipher metadata
                new_cipher_group = ciphers_group.create_group(
                    new_cipher_name
                )

                new_cipher_group.create_dataset(
                    "key",
                    data=key
                )

                new_cipher_group.create_dataset(
                    "plaintexts",
                    data=plaintexts[co_idx:co_idx + 1]
                )

                new_idx += 1

            total_subtraces += n_cos

        print(f"Original traces: {len(trace_names)}")
        print(f"Generated subtraces: {total_subtraces}")
        print(f"Output: {output_path}")

# def validate_transformed_chunk(output_path):

#     with h5py.File(output_path, "r") as f:

#         traces = f["data/traces"]
#         pinpoints = f["metadata/pinpoints"]
#         ciphers = f["metadata/ciphers"]

#         n_traces = len(traces)

#         assert n_traces == len(pinpoints)
#         assert n_traces == len(ciphers)

#         for i in range(n_traces):

#             trace = traces[f"trace_{i}"]
#             pp = pinpoints[f"pinpoints_{i}"]
#             cipher = ciphers[f"ciphers_{i}"]

#             assert trace.ndim == 1
#             # assert pp.shape == (1, 2)
#             # assert len(cipher["plaintexts"]) == 1

#             # start, end = pp[0]

#             # assert 0 <= start <= end < len(trace)
#             assert pp.shape == (1,)
#             assert pp.dtype.names == ("start", "end")

#             start = pp[0]["start"]
#             end = pp[0]["end"]

#             assert 0 <= start <= end < len(trace)

#         print(f"Validation successful: {n_traces} subtraces.")

def validate_transformed_chunk(output_path, txt_output_dir):

    from pathlib import Path

    txt_output_dir = Path(txt_output_dir)
    txt_output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "r") as f:

        traces = f["data/traces"]
        pinpoints = f["metadata/pinpoints"]
        ciphers = f["metadata/ciphers"]

        n_traces = len(traces)

        assert n_traces == len(pinpoints)
        assert n_traces == len(ciphers)

        for i in tqdm(range(n_traces), desc="Validating traces"):

            trace_name = f"trace_{i}"
            pinpoint_name = f"pinpoints_{i}"
            cipher_name = f"ciphers_{i}"

            trace = traces[trace_name]
            pp = pinpoints[pinpoint_name]
            cipher = ciphers[cipher_name]

            # Basic validation
            assert trace.ndim == 1
            assert pp.shape == (1,)
            assert pp.dtype.names == ("start", "end")
            assert len(cipher["plaintexts"]) == 1

            start = int(pp[0]["start"])
            end = int(pp[0]["end"])

            #assert 0 <= start <= end < len(trace)
            assert 0 <= start < end <= len(trace)
            # Export pinpoint to TXT
            txt_file = txt_output_dir / f"pinpoints_{trace_name}.txt"

            with open(txt_file, "w") as out:
                out.write(f"{start}\t{end}\n")

        print(f"Validation successful: {n_traces} subtraces.")
        print(f"Pinpoints exported to: {txt_output_dir}")
        

if __name__ == "__main__":

    transform_chameleon_chunk(CHAMELEON_FILE, OUTPUT_FILE)  # Trace Dataset transformed 
    validate_transformed_chunk(OUTPUT_FILE, "pinpoints_txt/")                 # Validation 

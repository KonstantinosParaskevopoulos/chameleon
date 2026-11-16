from pathlib import Path


def write_boundaries(output_dir, trace_index, start, end):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"trace_{trace_index}.txt"
    with output_file.open("w", encoding="utf-8") as file:
        file.write(f"{start}\t{end}\n")

    return output_file

import h5py
import numpy as np


def _numeric_key(name):
    """Sort names such as trace_2 before trace_10."""
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return name


class HDF5Dataset:
    def __init__(self, filename, traces_path, pinpoints_path,
                 trace_prefix="trace_", pinpoint_prefix="pinpoints_"):
        self.file = h5py.File(filename, "r")
        self.traces = self.file[traces_path]
        self.pinpoints = self.file[pinpoints_path]
        self.trace_prefix = trace_prefix
        self.pinpoint_prefix = pinpoint_prefix

        self.trace_names = sorted(
            [k for k in self.traces.keys() if k.startswith(trace_prefix)],
            key=_numeric_key
        )

    def __len__(self):
        return len(self.trace_names)

    def trace(self, index):
        name = self.trace_names[index]
        return np.asarray(self.traces[name])

    def pinpoint(self, index, pinpoint_index=0):
        name = f"{self.pinpoint_prefix}{index}"
        points = self.pinpoints[name]
        point = points[pinpoint_index]

        # Supports the structured dtype with fields "start" and "end".
        if points.dtype.names:
            return int(point["start"]), int(point["end"])

        # Also supports a simple Nx2 array.
        return int(point[0]), int(point[1])

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

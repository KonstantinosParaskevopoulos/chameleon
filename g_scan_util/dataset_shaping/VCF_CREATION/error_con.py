import numpy as np

# Example int16 trace
trace = np.arange(-2048, 2048, dtype=np.int16)

offset = 0
normalization_factor = 1.0

# float32 implementation
trace_f32 = trace.astype(np.float32)

norm_f32 = (((trace_f32 + offset) - trace_f32.min()) /
            (trace_f32.max() - trace_f32.min())) * normalization_factor


# float64 reference
trace_f64 = trace.astype(np.float64)

norm_f64 = (((trace_f64 + offset) - trace_f64.min()) /
            (trace_f64.max() - trace_f64.min())) * normalization_factor


# Error
error = norm_f64 - norm_f32.astype(np.float64)

print("Maximum absolute error:", np.max(np.abs(error)))
print("Mean error:", np.mean(np.abs(error)))
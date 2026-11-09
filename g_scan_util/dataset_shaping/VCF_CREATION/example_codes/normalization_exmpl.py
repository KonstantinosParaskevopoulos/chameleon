import numpy as np

# Example ADC inputs (int16)
adc_values = np.array([
    -2048,   # minimum
    -1024,
    0,       # mid-scale
    1024,
    2047     # maximum
], dtype=np.int16)

# Min–max normalization for 12-bit signed ADC
# x' = (x + 2048) / 4095


####################################################################
normalized = (adc_values.astype(np.float32) + 2048.0) / 4095.0
####################################################################
# Print results
print("ADC input  ->  Normalized [0,1]")
for x, x_norm in zip(adc_values, normalized):
    print(f"{x:6d}     ->  {x_norm:.2f}")

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Read intervals from file
# =========================
def read_intervals(filename):
    intervals = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()   # <-- key fix (handles any whitespace)

            if len(parts) < 2:
                continue  # skip malformed lines

            start, end = map(int, parts[:2])
            intervals.append((start, end))

    return intervals


# =========================
# 2. Compute overlap intervals (efficient)
# =========================
def compute_overlaps(A, B):
    overlaps = []
    for s1, e1 in A:
        for s2, e2 in B:
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                overlaps.append((start, end))
    return overlaps


# =========================
# 3. Build binary signals (for visualization)
# =========================
def build_signal(intervals, x):
    signal = np.zeros_like(x)
    for s, e in intervals:
        signal[(x >= s) & (x <= e)] = 1
    return signal


# =========================
# 4. Plot binary signals
# =========================
def plot_single(x, signal, title):
    plt.figure(figsize=(12, 3))
    plt.plot(x, signal)
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

# =========================
# 4b. Plot all signals in one figure with separate subplots
# =========================
def plot_all_signals(x, signal_A, signal_B, signal_overlap):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(x, signal_A, color='blue')
    axs[0].set_title("Signal A")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    axs[1].plot(x, signal_B, color='green')
    axs[1].set_title("Signal B")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    axs[2].plot(x, signal_overlap, color='red')
    axs[2].set_title("Overlap Signal")
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Value")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# =========================
# 5. Plot interval lines (clean visualization)
# =========================
def plot_intervals(A, B, overlaps):
    plt.figure(figsize=(12, 3))

    for s, e in A:
        plt.plot([s, e], [2, 2], linewidth=6, label="A" if s == A[0][0] else "")

    for s, e in B:
        plt.plot([s, e], [1, 1], linewidth=6, label="B" if s == B[0][0] else "")

    for s, e in overlaps:
        plt.plot([s, e], [0, 0], linewidth=6, label="Overlap" if s == overlaps[0][0] else "")

    plt.yticks([0, 1, 2], ["Overlap", "B", "A"])
    plt.legend()
    plt.title("Interval Visualization")
    plt.grid()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Input files
    file_A = "A.txt"
    file_B = "B.txt"

    # Read intervals
    A = read_intervals(file_A)
    B = read_intervals(file_B)

    print("A intervals:", A)
    print("B intervals:", B)

    # Compute overlaps
    overlaps = compute_overlaps(A, B)
    print("Overlaps:", overlaps)

    # Build axis (ONLY for plotting signals)
    xmin = min(min(s for s, _ in A), min(s for s, _ in B))
    xmax = max(max(e for _, e in A), max(e for _, e in B))
    #x = np.arange(xmin, xmax + 1)
    step = 1000  # or larger
    x = np.arange(xmin, xmax + 1)#, step)
    # Build signals
    signal_A = build_signal(A, x)
    signal_B = build_signal(B, x)
    signal_overlap = signal_A & signal_B

    # Plot results
    # plot_single(x, signal_A, "Signal A")
    # plot_single(x, signal_B, "Signal B")
    # plot_single(x, signal_overlap, "Overlap Signal")    
    #plot_intervals(A, B, overlaps)
    plot_all_signals(x, signal_A, signal_B, signal_overlap)
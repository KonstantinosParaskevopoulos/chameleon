#include <stdio.h>
#include <stdlib.h>

#define MAX_PREDS 10000
#define MAX_TRACE 100000

typedef struct {
    long pos;
    int cls;
} Prediction;

Prediction preds[MAX_PREDS];

// Core buffers
double waveform[MAX_TRACE];
double smooth[MAX_TRACE];
int binary[MAX_TRACE];

// Difference array for fast range updates
double diff[MAX_TRACE + 1];

int main() {
    FILE *in = fopen("../RAiSD_Report.FASTER-NN-PT-BINFRQPOS-SCAN-SWEEP.1", "r");
    FILE *out = fopen("../../RESULTS/segments.txt", "w");

    if (!in || !out) {
        printf("Error opening files\n");
        return 1;
    }

    int n = 0;
    while (fscanf(in, "%ld %d", &preds[n].pos, &preds[n].cls) == 2) {
        n++;
    }

    printf("Loaded %d predictions\n", n);

    // ===== PARAMETERS =====
    long trace_len = MAX_TRACE;   // total trace length (adjust!)
    int window_size = 100;    // approximate effective window
    int half_w = window_size / 2;

    int smooth_k = 5;           // smoothing radius
    double threshold = 0.5;     // detection threshold
    printf("STEP 1");
    // ===== STEP 1: Waveform (difference array) =====
    for (long i = 0; i <= trace_len; i++)
        diff[i] = 0.0;

    for (int i = 0; i < n; i++) {
        if (preds[i].cls == 0) { // ENCR only
            long start = preds[i].pos - half_w;
            long end   = preds[i].pos + half_w;

            if (start < 0) start = 0;
            if (end >= trace_len) end = trace_len - 1;

            diff[start] += 1.0;
            diff[end + 1] -= 1.0;
        }
    }
    
    // Prefix sum → waveform
    waveform[0] = diff[0];
    for (long i = 1; i < trace_len; i++) {
        waveform[i] = waveform[i - 1] + diff[i];
    }
    printf("STEP 2");
    // ===== STEP 2: Smoothing (moving average) =====
    for (long i = 0; i < trace_len; i++) {
        double sum = 0.0;
        int count = 0;

        for (int k = -smooth_k; k <= smooth_k; k++) {
            long idx = i + k;
            if (idx >= 0 && idx < trace_len) {
                sum += waveform[idx];
                count++;
            }
        }

        smooth[i] = sum / count;
    }
    printf("STEP 3");
    // ===== STEP 3: Thresholding =====
    for (long i = 0; i < trace_len; i++) {
        binary[i] = (smooth[i] > threshold) ? 1 : 0;
    }
    printf("STEP 4");
    // ===== STEP 4: Segmentation =====
    int in_segment = 0;
    long seg_start = 0;

    for (long i = 0; i < trace_len; i++) {
        if (binary[i] == 1) { // ENCR region
            if (!in_segment) {
                seg_start = i;
                in_segment = 1;
            }
        } else {
            if (in_segment) {
                fprintf(out, "%ld\t%ld\n", seg_start, i - 1);
                in_segment = 0;
            }
        }
    }

    if (in_segment) {
        fprintf(out, "%ld\t%ld\n", seg_start, trace_len - 1);
    }

    fclose(in);
    fclose(out);

    printf("Segmentation complete → segments.txt\n");

    return 0;
}
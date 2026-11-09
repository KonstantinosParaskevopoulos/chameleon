import os
import random


# SSD / NVMe
# buffer_mb = 8–32
# FLUSH_EVERY = 500k – 1M

# HDD / Network FS
# buffer_mb = 1–4
# FLUSH_EVERY = 50k – 200k


FLUSH_EVERY = 500_000    # tune: 100k–1M depending on SSD/HDD


def init_vcf(
    output_file,
    num_samples,
    chrom="1",
    buffer_mb=8
):
    f = open(
        output_file,
        mode="a",                      # append-only, no truncation
        encoding="utf-8",
        buffering=buffer_mb * 1024 * 1024
    )

    # Write header ONLY if file is empty
    if f.tell() == 0:
        f.write("##fileformat=VCFv4.2\n")
        f.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        )

        sample_headers = "\t".join(
            f"sample{i+1}" for i in range(num_samples)
        )

        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + sample_headers + "\n"
        )

        f.flush()
        os.fsync(f.fileno())   # header is now durable

    return f



def append_vcf_row(
    f,
    binary_row,
    pos,
    ref_alleles,
    alt_alleles,
    chrom="1",
):
    # Pre-allocate list (faster)
    row = [
        chrom,
        str(pos),
        ".",
        random.choice(ref_alleles),
        random.choice(alt_alleles),
        ".",
        ".",
        ".",
        "GT"
    ]

    row.extend(str(v) for v in binary_row)
 

    f.write("\t".join(row))
    f.write("\n")


def flush_buffer(f, counter):
    # Periodic durability checkpoint
    if counter % FLUSH_EVERY == 0:
        f.flush()
        os.fsync(f.fileno())

import os
import re

COUNTERMEASURE="BASE"

directory = f"/local/scratch/paraskevopoulosk/Datasets/{COUNTERMEASURE}/train_{COUNTERMEASURE}"
#directory = "/local/scratch/paraskevopoulosk/Datasets"
out_file = f"RANDOM_ONLY_cpl_{COUNTERMEASURE}.vcf"

# Match RAND1_..., RAND2_..., etc.
pattern = re.compile(r"^RAND_(\d+).*\.vcf$")

files = []

for f in os.listdir(directory):
    m = pattern.match(f)
    if m:
        rand_idx = int(m.group(1))
        files.append((rand_idx, f)) #Extract index from name

# Sort numerically: RAND1, RAND2, ..., RAND10
files.sort(key=lambda x: x[0])

header_written = False
pos_counter = 1
with open(out_file, "w") as out:

    for rand_idx, fname in files:
        
        with open(os.path.join(directory, fname), "r") as inp:

            for line in inp:

                if line.startswith("##"):
                    if not header_written:
                        out.write(line)
                    continue

                if line.startswith("#CHROM"):
                    if not header_written:
                        out.write(line)
                        header_written = True
                    continue

                if not line.strip():
                    continue

                cols = line.rstrip("\n").split("\t")

                # Set CHROM equal to RAND number
                cols[0] = str(rand_idx + 1) # CHROM Index starts @ 1
                cols[1] = str(pos_counter)
                pos_counter += 1
                out.write("\t".join(cols) + "\n")

print(f"Merged {len(files)} RAND VCF files into {out_file}")
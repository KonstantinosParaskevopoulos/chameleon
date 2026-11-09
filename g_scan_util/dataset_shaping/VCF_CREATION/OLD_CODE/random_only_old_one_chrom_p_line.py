import os
import re
COUNTERMEASURE="RD"

directory = f"/local/scratch/paraskevopoulosk/Datasets/{COUNTERMEASURE}/train_{COUNTERMEASURE}"
out_file = "merged.vcf"

# STRICT match: RAND + number + underscore
pattern = re.compile(r"^RAND_(\d+).*\.vcf$")

files = []

for f in os.listdir(directory):
    m = pattern.match(f)
    if not m:
        continue

    idx = int(m.group(1))
    files.append((idx, f))

# sort by RAND index
files.sort(key=lambda x: x[0])

global_chrom = 0
header_written = False

with open(out_file, "w") as out:

    for _, fname in files:
        path = os.path.join(directory, fname)

        with open(path) as inp:
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

                cols = line.split("\t")

                global_chrom += 1
                cols[0] = str(global_chrom)

                out.write("\t".join(cols))

print(f"Merged {len(files)} RAND files into {out_file}")
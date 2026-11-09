def write_vcf(
    binary_matrix,
    ref_alleles,
    alt_alleles,
    start_pos,
    chrom="1",
    output_file="binary_matrix.vcf"
):
    """
    binary_matrix: list of lists (variants x samples), values in {0,1,None}
    ref_alleles:   list of REF alleles per variant
    alt_alleles:   list of ALT alleles per variant
    start_pos:     integer starting genomic position
    chrom:         chromosome label
    """

    num_variants = len(binary_matrix)
    num_samples = len(binary_matrix[0])

    with open(output_file, "w") as f:
        # Header
        f.write("##fileformat=VCFv4.2\n")
        f.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        )

        # Column header
        sample_headers = "\t".join([f"sample{i+1}" for i in range(num_samples)])
        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + sample_headers + "\n"
        )

        # Data rows
        for i in range(num_variants):
            pos = start_pos + i

            row = [
                chrom,
                str(pos),
                ".",
                ref_alleles[i],
                alt_alleles[i],
                ".",
                ".",
                ".",
                "GT"
            ]

            for val in binary_matrix[i]:
                if val is None:
                    row.append(".")
                else:
                    row.append(str(val))

            f.write("\t".join(row) + "\n")

def main():
    binary_matrix = [
    [0,0,1,0,0,1,0,0,1,0],
    [0,0,0,1,0,0,1,0,1,0],
    [1,0,0,0,1,0,0,1,0,0],
    [0,0,0,0,1,0,0,1,1,0],
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,1,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,0,1],
    [0,0,1,0,0,1,0,0,1,0],
    [0,1,0,0,1,0,0,0,0,0],
    [1,0,0,0,0,1,0,0,1,1],
    ]

    ref = ["A","C","G","T","A","G","C","T","A","G"]
    alt = ["G","T","A","C","C","T","G","A","T","C"]

    write_vcf(
        binary_matrix=binary_matrix,
        ref_alleles=ref,
        alt_alleles=alt,
        start_pos=1,
        output_file="derived_binary.vcf"
    )

if __name__ == "__main__":
    main()
from Bio import Seq

def extract_flanked_region(chrom_seq_str: str, strand: str, start: int, stop: int, upstream: int, downstream: int) -> str:
    """
    Extracts a sequence from chrom_seq_str from (start - upstream) to (stop + downstream),
    using 1-based inclusive coordinates and orienting 5' to 3' relative to the strand.

    Parameters:
        chrom_seq_str (str): Chromosome sequence as a string.
        strand (str): '+' or '-' indicating the strand.
        start (int): Start position (1-based, inclusive).
        stop (int): Stop position (1-based, inclusive).
        upstream (int): Number of bases to include upstream of start.
        downstream (int): Number of bases to include downstream of stop.

    Returns:
        str: The resulting sequence as a string.
    """
    if strand not in ('+', '-'):
        raise ValueError("Strand must be '+' or '-'")
    if start < 1 or stop < 1:
        raise ValueError("Start and stop must be 1-based (i.e., ≥ 1)")

    chrom_seq = Seq.Seq(chrom_seq_str)

    # Convert to 0-based index
    start0 = start - 1
    stop0 = stop - 1

    if strand == '+':
        seq_start = max(0, start0 - upstream)
        seq_end = stop0 + downstream + 1  # +1 to include stop position
        return str(chrom_seq[seq_start:seq_end]).upper()
    
    else:  # strand == '-'
        seq_start = max(0, start0 - downstream)
        seq_end = stop0 + upstream + 1
        return str(chrom_seq[seq_start:seq_end].reverse_complement()).upper()
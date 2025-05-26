import logging
from Bio import SeqIO, Seq
import pandas as pd
from typing import List

# Configure logging to file and stream
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

def is_valid_mutation(start: int, end: int, seq_length: int) -> bool:
    """
    Checks if the mutation coordinates are valid within the given sequence length.
    """
    return (
        isinstance(start, int) and isinstance(end, int) and
        1 <= start <= seq_length and
        1 <= end <= seq_length and
        start <= end
    )

def extract_sequences(fasta_file: str, mutations_df: pd.DataFrame, window_size: int = 10) -> list:
    """
    Extracts flanking sequences around mutation positions from a FASTA file for each mutation in the DataFrame.
    """
    gene_sequences = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            gene_sequences[record.id] = str(record.seq).upper()
        logging.info(f"Loaded {len(gene_sequences)} sequences from {fasta_file}")
    except FileNotFoundError:
        logging.error(f"FASTA file not found: {fasta_file}")
        raise FileNotFoundError(f"FASTA file {fasta_file} not found.")
    except Exception as e:
        logging.error(f"Error parsing FASTA file: {e}")
        raise Exception(f"Error parsing FASTA file: {str(e)}")

    flanked_sequences = []

    for idx, row in mutations_df.iterrows():
        gene = row.get('Gene_name')
        start = row.get('Start_Position')
        end = row.get('End_Position')
        strand = row.get('Strand', '+')
        case_id = row.get('case_id', 'unknown')

        if not gene or gene not in gene_sequences:
            logging.warning(f"Case {case_id}: Gene {gene} not found in FASTA file")
            flanked_sequences.append('')
            continue

        try:
            start = int(start)
            end = int(end)
        except (ValueError, TypeError):
            logging.warning(f"Case {case_id}: Invalid position types for gene {gene}: start={start}, end={end}")
            flanked_sequences.append('')
            continue

        seq_str = gene_sequences[gene]
        seq_length = len(seq_str)

        if not is_valid_mutation(start, end, seq_length):
            logging.warning(f"Case {case_id}: Invalid mutation coordinates for gene {gene}: start={start}, end={end}, seq_len={seq_length}")
            flanked_sequences.append('')
            continue

        if strand not in ('+', '-'):
            logging.warning(f"Case {case_id}: Invalid strand '{strand}' for gene {gene}, defaulting to '+'")
            strand = '+'

        try:
            seq = extract_flanked_region(
                chrom_seq_str=seq_str,
                strand=strand,
                start=start,
                stop=end,
                upstream=window_size,
                downstream=window_size
            )
            expected_length = 2 * window_size + (end - start + 1)
            if len(seq) < expected_length:
                logging.warning(f"Case {case_id}: Extracted sequence too short: {len(seq)} vs expected {expected_length}")
        except Exception as e:
            logging.error(f"Case {case_id}: Error extracting sequence for gene {gene}: {str(e)}")
            seq = ''
        flanked_sequences.append(seq)

    valid_count = len([s for s in flanked_sequences if s])
    invalid_count = len(flanked_sequences) - valid_count
    logging.info(f"Extracted {valid_count} valid sequences, {invalid_count} invalid")
    return flanked_sequences

def extract_flanked_region(
    chrom_seq_str: str,
    strand: str,
    start: int,
    stop: int,
    upstream: int,
    downstream: int
) -> str:
    """
    Extracts a sequence from chrom_seq_str from (start - upstream) to (stop + downstream),
    using 1-based inclusive coordinates and orienting 5' to 3' relative to the strand.
    """
    if strand not in ('+', '-'):
        raise ValueError(f"Invalid strand: {strand}. Must be '+' or '-'")
    if start < 1 or stop < 1:
        raise ValueError(f"Invalid coordinates: start={start}, stop={stop}. Must be ≥ 1")
    if start > stop:
        raise ValueError(f"Start position {start} is greater than stop position {stop}")

    chrom_seq = Seq.Seq(chrom_seq_str)
    if not chrom_seq:
        raise ValueError("Empty sequence provided")

    start0, stop0 = start - 1, stop - 1

    if strand == '+':
        seq_start = max(0, start0 - upstream)
        seq_end = min(len(chrom_seq), stop0 + downstream + 1)
        if seq_end <= seq_start:
            raise ValueError(f"Invalid extraction range: start={seq_start}, end={seq_end}")
        return str(chrom_seq[seq_start:seq_end]).upper()
    else:
        seq_start = max(0, start0 - downstream)
        seq_end = min(len(chrom_seq), stop0 + upstream + 1)
        if seq_end <= seq_start:
            raise ValueError(f"Invalid extraction range: start={seq_start}, end={seq_end}")
        return str(chrom_seq[seq_start:seq_end].reverse_complement()).upper()

if __name__ == "__main__":
    # Adjust these paths to your files
    fasta_path = "genes.fasta"
    mutations_csv_path = "train_muts_data.csv"
    output_csv_path = "flanked_sequences.csv"

    # Read mutations data
    try:
        mutations_df = pd.read_csv(mutations_csv_path)
        logging.info(f"Loaded mutations data: {mutations_csv_path}")
    except Exception as e:
        logging.error(f"Failed to read mutations CSV: {e}")
        exit(1)

    # Extract sequences
    sequences = extract_sequences(fasta_path, mutations_df, window_size=10)

    # Save to CSV
    mutations_df['Flanked_Sequence'] = sequences
    mutations_df.to_csv(output_csv_path, index=False)
    logging.info(f"✅ Saved flanked sequences to {output_csv_path}")

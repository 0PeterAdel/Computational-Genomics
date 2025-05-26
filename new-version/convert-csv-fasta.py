import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Load CSV file
df = pd.read_csv("../DataSet/100_genes.csv") 

# Check for required columns
if 'gene' in df.columns and 'Sequence' in df.columns:
    records = [
        SeqRecord(Seq(row['Sequence']), id=str(row['gene']), description="")
        for _, row in df.iterrows()
    ]
    SeqIO.write(records, "../DataSet/genes.fasta", "fasta")
    print("✅ FASTA file created: genes.fasta")
else:
    print("❌ The CSV file does not contain the required columns: 'gene' and 'Sequence'")

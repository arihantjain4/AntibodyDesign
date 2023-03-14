from Bio import SeqIO
print("PRIMARY STRUCTURE: amino acid sequences")
for record in SeqIO.parse("PotD.fasta", "fasta"):
    print(record.seq)
# for record in SeqIO.parse("ANTIBODY-MUROMONAB-aaSeq.fasta", "fasta"):
#     print(record.seq)

# for record in SeqIO.parse("ANTIGEN-CD3-aaSeq.fasta", "fasta"):
#     print(record.seq)
#

# print("TERIARY STRUCTURE: pdb files")
# # for record in SeqIO.parse("ANTIBODY-MUROMONAB-3d.pdb", "pdb-atom"):
# #     print(record)
#
# for record in SeqIO.parse("ANTIGEN-CD3-3d.pdb", "pdb-atom"):
#     print(record)
#
# print("QUATERNARY STRUCTURE: 3d with annotations")

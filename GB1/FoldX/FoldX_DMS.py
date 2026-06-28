#!/usr/bin/env python
# coding: utf-8

# # Create FoldX files with all mutations in DMS.
# 
# Will generate one file for position

# In[1]:


import Bio
import pandas
from Bio import SeqIO
import os


# In[2]:


pwd


# In[3]:


aa_list = list('ACDEFGHIKLMNPQRSTVWY')
aa_list


# In[4]:


len(aa_list)


# In[5]:


input_file = '1fcc_C.fasta'
fasta_sequences = SeqIO.parse(open(input_file),'fasta')


# In[6]:


for seq in fasta_sequences:
    print(seq.seq)


# In[7]:


for i,aa in enumerate(seq):
    out = open(f'individual_list_{i+1}.txt','w')
    sub = [r for r in aa_list if r != aa]
    for s in sub:
        out.write(f'{aa}C{i+1}{s};\n')
    out.close()
    


# In[8]:


os.chdir(f'/Users/ignaciaecheverria/Dropbox/UCSF/DMS/GB1/FoldX/')
for i,aa in enumerate(seq):
    sub = [r for r in aa_list if r != aa]
    os.system(f'~/SOFTW/foldx5MacC11/foldx_20241231 --command=BuildModel --pdb=1fcc_AC.pdb --mutant-file=individual_list_{i+1}.txt')
    if not os.path.exists(f'pos_{i+1}'):
        os.system(f'mkdir pos_{i+1}')
    print('SUB', sub)
    for j, s in enumerate(sub):
        os.system(f'mv 1fcc_AC_{j+1}.pdb pos_{i+1}/1fcc_AC_{i+1}_{j+1}_{s}.pdb')
        os.system(f'mv WT_1fcc_AC_{j+1}.pdb pos_{i+1}/1fcc_AC_{i+1}_{j+1}_{s}.pdb')
    os.system(f'mv Dif_1fcc_AC.fxout pos_{i+1}/')
    os.system(f'mv Raw_1fcc_AC.fxout pos_{i+1}/')
    os.system(f'mv Average_1fcc_AC.fxout pos_{i+1}/')


# In[ ]:


for i,aa in enumerate(seq):
    sub = [r for r in aa_list if r != aa]
    os.chdir(f'/Users/ignaciaecheverria/Dropbox/UCSF/DMS/GB1/FoldX/pos_{i+1}')
    for j, s in enumerate(sub):
        print(f'1fcc_AC_{i+1}_{j+1}_{s}.pdb')
        os.system(f'~/SOFTW/foldx5MacC11/foldx_20241231 --command=AnalyseComplex --pdb=1fcc_AC_{i+1}_{j+1}_{s}.pdb --analyseComplexChains=A,C --complexWithDNA=false')


# In[ ]:


#os.chdir(f'/Users/ignaciaecheverria/Dropbox/UCSF/DMS/GB1/FoldX/pos_1')
#os.system(f'~/SOFTW/foldx5MacC11/foldx_20241231 --command=AnalyseComplex --pdb=1fcc_AC_1_10_M.pdb --analyseComplexChains=A,C --complexWithDNA=false')


# In[ ]:


pwd


# In[ ]:


#~/SOFTW/foldx5MacC11/foldx_20241231 --command=AnalyseComplex --pdb=1fcc_AC.pdb --analyseComplexChains=A,C --complexWithDNA=false


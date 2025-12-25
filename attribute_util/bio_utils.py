#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import argparse
from collections import defaultdict
import numpy as np

codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    'PPP':'#',
}

int_table = {
    'I':0,  'M':1,  'T':2,  'N':3,  'K':4,
    'S':5,  'R':6,  'L':7,  'P':8,  'H':9,
    'Q':10, 'V':11, 'A':12, 'D':13, 'E':14,
    'G':15, 'F':16, 'Y':17, 'C':18, 'W':19,
    '*':20, '#':21,
}

int_inv = {
    0:'I',  1:'M',  2:'T',  3:'N',  4:'K',
    5:'S',  6:'R',  7:'L',  8:'P',  9:'H',
    10:'Q', 11:'V', 12:'A', 13:'D', 14:'E',
    15:'G', 16:'F', 17:'Y', 18:'C', 19:'W',
    20:'*', 21:'#'
}

aa_table = defaultdict(list)
for key,value in codon_table.items():
    aa_table[value].append(key)

# def readFasta(filename):
#     try:
#         f = file(filename)
#     except IOError:
#         print("The file, %s, does not exist" % filename)
#         return

#     order = []
#     sequences = {}
#     for line in f:
#         if line.startswith('>'):
#             name = line[1:].rstrip('\n')
#             name = name.replace('_', ' ')
#             order.append(name)
#             sequences[name] = ''
#         else:
#             sequences[name] += line.rstrip('\n').rstrip('*')
#     print("%d sequences found" % len(order))
#     return order, sequences

def geneToProtein(dna_seqs, verbose=True):
    global codon_table
    p_seqs = []
    for dna_seq in dna_seqs:
        p_seq = "M"
        if dna_seq[0:3] != 'ATG':
            if verbose: print("Not valid gene (no ATG)")
            continue
        for i in range(3, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            try:
                aa = codon_table[codon]
                p_seq += aa
                if aa == '*': break
            except:
                if verbose: print("Error! Invalid Codon {} in {}".format(codon, dna_seq))
                break
        if len(p_seq) <= 2: #needs to have stop codon and be of length greater than 2
            if verbose: print("Error! Protein too short.")
        elif p_seq[-1] != '*':
            if verbose: print("Error! No valid stop codon.")
        else:
            p_seqs += [p_seq]
    return p_seqs

def proteinToInt(peps, verbose=True):#transform only one seq
    global int_table
    i_seqs = []
    peps = peps.replace('\n','')
    for pep in peps:
        try :
            int_num = int_table[pep]
            i_seqs.append(int_num)
        except :
            if verbose: print('ERROR!invalid condon {} in {}'.format(pep,peps))
    return i_seqs

def intToProtein(ints, verbose=True):
    global int_inv
    peps = ''
    for num in ints:
        try :
            pep = int_inv[num]
            peps += pep
        except :
            if verbose: print('ERROR!invalid condon {} in {}'.format(pep,peps))
    return peps


def countCorrectP(dna_seqs, verbose=False):
    global codon_table
    total = 0.0
    correct = 0.0
    for dna_seq in dna_seqs:
        p_seq = ""
        total += 1
        if dna_seq[0:3] != 'ATG':
            if verbose: print("Not valid gene (no ATG)")
            continue
        for i in range(3, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            try:
                aa = codon_table[codon]
                p_seq += aa
                if aa == '_': 
                    break
            except:
                if verbose: print("Error! Invalid Codon {} in {}".format(codon, dna_seq))
                break
        if len(p_seq) <= 2: #needs to have stop codon and be of length greater than 2
            if verbose: print("Error! Protein too short.")
        elif p_seq[-1] != '_':
            if verbose: print("Error! No valid stop codon.")
        else:
            correct+=1
    return correct/total

def proteinToDNA(protein_seqs):
    global aa_table
    stop_codons = ['TAA', 'TAG', 'TGA']
    dna_seqs = []
    for p_seq in protein_seqs:
        dna_seq_list = [np.random.choice(aa_table[aa]) for aa in p_seq]
        # stop_codon = np.random.choice(stop_codons)
        dna_seq = ""
        for seq in dna_seq_list:
            dna_seq += seq
        # dna_seqs += ['ATG' + "".join(dna_seq)+ stop_codon]
        dna_seqs.append(dna_seq)
    # print(len(dna_seqs[0]))
    return dna_seqs

def main():
    parser = argparse.ArgumentParser(description='protein to dna.')
    parser.add_argument("--dataset", default="random", help="Dataset to load (else random)")
    args = parser.parse_args()
    outfile = './samples/' + args.dataset + '_dna_seqs.fa'
    with open(args.dataset,'rb') as f:
        dna_seqs = f.readlines()
    p_seqs = geneToProtein(dna_seqs)

if __name__ == '__main__':
    main()

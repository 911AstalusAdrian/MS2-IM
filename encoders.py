import itertools
from abc import ABC, abstractmethod

import torch


# Encoder interface
class Encoder(ABC):
    @abstractmethod
    def encode(self, sequence, seq_length, num_features):
        pass


class OneHotEncoder(Encoder):
    def encode(self, sequence, seq_length, num_features):
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded_sequence = torch.zeros(seq_length, num_features)
        for i, base in enumerate(sequence):
            if base in base_to_idx and base != '0':
                encoded_sequence[i][base_to_idx[base]] = 1
        return encoded_sequence


class KMerEncoder(Encoder):
    def __init__(self):
        self.k = 4
        self.kmer_to_idx = self.generate_kmer_indices()

    def generate_kmer_indices(self):
        bases = ['A', 'C', 'G', 'T']
        kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        return {kmer: i for i, kmer in enumerate(kmers)}

    def encode(self, sequence, seq_length, num_features):
        encoded = torch.zeros((len(sequence) - self.k + 1, len(self.kmer_to_idx)))
        for i in range(len(sequence) - self.k + 1):
            kmer = ''.join(sequence[i:i + self.k])
            if kmer in self.kmer_to_idx:
                encoded[i][self.kmer_to_idx[kmer]] = 1
        return encoded

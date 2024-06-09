import itertools
from abc import ABC, abstractmethod
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer


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


class Autoencoder(Model):
    pass
#     def __init__(self, latent_dim, shape):
#         print("Into Autoencoder class")
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.shape = shape
#         # self.encoder = tf.keras.Sequential([
#         #     layers.Flatten(),
#         #     layers.Dense(latent_dim, activation='relu'),
#         # ])
#         # self.decoder = tf.keras.Sequential([
#         #     layers.Dense(np.prod(shape), activation='sigmoid'),
#         #     layers.Reshape(shape)
#         # ])
#
#         self.encoder = tf.keras.Sequential([
#             layers.Flatten(),
#             layers.Dense(256, activation='relu'),
#             layers.Dropout(0.2),  # Adding dropout for regularization
#             layers.Dense(latent_dim, activation='relu'),
#         ])
#
#         # Decoder
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(256, activation='relu'),
#             layers.Dropout(0.2),  # Adding dropout for regularization
#             layers.Dense(np.prod(shape), activation='sigmoid'),
#             layers.Reshape(shape)
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#     def encode(self, sequence, seq_length, num_features):
#         return self.encoder(sequence)

class DNAAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(DNAAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Sigmoid to get the output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def encode_sequence(self, seq):
        numerical_data = self.one_hot_encode(seq)
        with torch.no_grad():
            return self.encoder(torch.tensor(numerical_data, dtype=torch.float32))

    def decode(self, x):
        with torch.no_grad():
            return self.decoder(x)

    def train_autoencoder(self, data, num_epochs=1000, learning_rate=0.001, batch_size=32):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # Shuffle the data
            np.random.shuffle(data)

            # Mini-batch training
            for i in range(0, len(data), batch_size):
                batch_sequences = data[i:i + batch_size]

                # One-hot encode the batch sequences
                batch_encoded = []
                for seq in batch_sequences:
                    encoded_seq = DNAAutoencoder.one_hot_encode(seq)
                    batch_encoded.append(encoded_seq)
                batch_data = torch.tensor(batch_encoded, dtype=torch.float32)

                # Forward pass
                output = self.forward(batch_data)
                loss = criterion(output, batch_data)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    @staticmethod
    def one_hot_encode(sequence):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '0': 0}
        one_hot = np.zeros((len(sequence), 4))
        for i, nucleotide in enumerate(sequence):
            # print("Nucleotide:", nucleotide)
            # print("Mapping:", mapping)
            if nucleotide not in mapping.keys():
                one_hot[i, 0] = 1
            else:
                one_hot[i, mapping[nucleotide]] = 1
        return one_hot.flatten()

    @staticmethod
    def one_hot_decode(encoded_seq):
        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        decoded_seq = []
        for i in range(0, len(encoded_seq), 4):
            one_hot = encoded_seq[i:i + 4]
            decoded_seq.append(mapping[np.argmax(one_hot)])
        return ''.join(decoded_seq)

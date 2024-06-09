import os

import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from encoders import Encoder, Autoencoder, DNAAutoencoder
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np


# Load and preprocess FASTQ data
def load_data():
    sequences_modern = []
    sequences_ancient = []

    ancient_dir = 'Data/ancient/'
    modern_dir = 'Data/modern/'

    for filename in os.listdir(ancient_dir):
        if filename.endswith(".fastq"):
            with open(os.path.join(ancient_dir, filename), "r") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    sequences_ancient.append(str(record.seq))

    for filename in os.listdir(modern_dir):
        if filename.endswith(".fastq"):
            with open(os.path.join(modern_dir, filename), "r") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    sequences_modern.append(str(record.seq))

    return sequences_ancient[:10000], sequences_modern[:10000]


def load_autoencoder_data():
    sequences_modern = []
    sequences_ancient = []

    ancient_dir = 'Data/ancient/'
    modern_dir = 'Data/modern/'

    for filename in os.listdir(ancient_dir):
        if filename.endswith(".fastq"):
            with open(os.path.join(ancient_dir, filename), "r") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    sequences_ancient.append(str(record.seq))

    for filename in os.listdir(modern_dir):
        if filename.endswith(".fastq"):
            with open(os.path.join(modern_dir, filename), "r") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    sequences_modern.append(str(record.seq))

    sequences = sequences_ancient[:100] + sequences_modern[:100]
    max_length = max(len(seq) for seq in sequences)
    sequences = process_sequences(sequences, max_length)
    return sequences


# Pad or truncate sequences to the maximum length
def process_sequences(sequences, max_length):
    processed_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            seq += '0' * (max_length - len(seq))
        elif len(seq) > max_length:
            seq = seq[:max_length]
        processed_sequences.append(seq)
    return processed_sequences


def get_max_len():
    # Load the data
    sequences = load_autoencoder_data()

    # Determine the maximum sequence length
    max_length = max(len(seq) for seq in sequences)
    return max_length


def load_and_preprocess_data(encoder: Encoder):
    # Load the data
    sequences_ancient, sequences_modern = load_data()

    # Determine the maximum sequence length
    max_length = max(max(len(seq) for seq in sequences_modern), max(len(seq) for seq in sequences_ancient))

    # Process sequences
    sequences_modern = process_sequences(sequences_modern, max_length)
    sequences_ancient = process_sequences(sequences_ancient, max_length)

    if isinstance(encoder, DNAAutoencoder):
        print('Encoding using Autoencoder')
        sequences_modern = [encoder.encode_sequence(seq) for seq in sequences_modern]
        sequences_ancient = [encoder.encode_sequence(seq) for seq in sequences_ancient]
    else:
        sequences_modern = [encoder.encode(seq, max_length, 4) for seq in sequences_modern]
        sequences_ancient = [encoder.encode(seq, max_length, 4) for seq in sequences_ancient]

    # Convert lists of tensors to a single tensor
    sequences_modern = torch.stack(sequences_modern, dim=0)
    sequences_ancient = torch.stack(sequences_ancient, dim=0)

    # Create labels for ancient (0) and modern (1) sequences
    labels_modern = torch.ones(sequences_modern.size(0), dtype=torch.long)
    labels_ancient = torch.zeros(sequences_ancient.size(0), dtype=torch.long)

    # Combine sequences and labels
    data_x = torch.cat((sequences_modern, sequences_ancient), dim=0)
    data_y = torch.cat((labels_modern, labels_ancient), dim=0)

    return data_x, data_y, max_length

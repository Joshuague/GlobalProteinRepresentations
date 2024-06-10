import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='This script trains a resnet bottleneck model using an input fasta and embedding file.')

# Define command-line arguments
parser.add_argument('--emb', type=str, help='Specify the path to the input embedding h5.')
parser.add_argument('--outpath', type=str, default = "light_attention.pth", help='Specify the path to the output file.')
parser.add_argument('--bs', type=int, default = 64, help='Specify the batch size.')
parser.add_argument('--num_epochs', type=int, default = 300, help='Specify the number of epochs to train for.')
parser.add_argument('--lr', type=float, default = 0.001, help='Specify the learning rate.')
parser.add_argument("--dropout", type = float, default = 0.5, help = "Specify the dropout.")
parser.add_argument("--filter", type = int, default = 512, help = "Specify the maximum seq length to keep.")
parser.add_argument("--num_layers", type = int, default = 2, help = "Specify the number of decoder layers.")
parser.add_argument("--heads", type = int, default = 16, help = "Specify the number of heads for multihead attention.")
parser.add_argument("--patience", type = float, default = 10, help = "Specify patience.")
parser.add_argument("--log_file", type = str, default = "la_training.log", help = "Specify the path to the logging file.")

args = parser.parse_args()

log_file = args.log_file
logging.basicConfig(filename=log_file, level=logging.INFO)

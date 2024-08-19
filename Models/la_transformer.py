import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
import optuna
import numpy as np
from copy import deepcopy
import math
import h5py
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""This is the same model as the la_embedder.py. But this was the original script taking an embedding and a sequence
file as input to train the model."""

aa_dic = {"X": 0, "<SOS>": 1, "<EOS>": 2, "A": 3, "R": 4, "N": 5, "D": 6, "C": 7, "Q": 8, "E": 9, "G": 10,
              "H": 11, "I": 12, "L": 13, "K": 14, "M": 15, "F": 16, "P": 17, "S": 18, "T": 19, "W": 20, "Y": 21,
              "V": 22}



def create_padding_mask(sequences):
    """
    Create a padding mask for a batch of sequences.

    Args:
    - sequences (torch.Tensor): Batch of sequences with zero-padding.

    Returns:
    - torch.Tensor: Padding mask with False for padding elements and True for actual elements.
    """
    padding_mask = torch.zeros_like(sequences[:, 0, :], dtype=torch.bool)
    padding_mask[sequences[:, 0, :] != 0] = True
    return padding_mask

def generate_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def tokenize_seq(sequence):
    output = [1] #SOS token
    seq_int = [aa_dic[ch] for ch in sequence]
    output.extend(seq_int)
    output.append(2) #EOS token
    return output

class CustomDataset(Dataset):
    def __init__(self, sequences, context_vectors):
        self.sequences = sequences
        self.context_vectors = context_vectors

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        context_vector = self.context_vectors[index]

        # Convert sequence and context_vector to PyTorch tensors if not already
        sequence = torch.tensor(sequence, dtype=torch.long)
        context_vector = torch.tensor(context_vector, dtype=torch.float32)

        return sequence, context_vector
    def state_dict(self):
        return {"sequences": self.sequences, "context_vectors": self.context_vectors}

    def load_state_dict(self, state_dict):
        self.sequences = state_dict["sequences"]
        self.context_vectors = state_dict["context_vectors"]


def custom_collate(batch):
    sequences, context_vectors = zip(*batch)
    input_seqs = [seq[:-1] for seq in sequences]
    target_seqs = [seq[1:] for seq in sequences]
    # Pad sequences to the length of the longest sequence in the batch
    max_seq_length = max(len(seq) for seq in target_seqs)
    padded_targets = [torch.nn.functional.pad(seq, (0, max_seq_length - len(seq))) for seq in target_seqs]
    padded_inputs = [torch.nn.functional.pad(seq, (0, max_seq_length - len(seq))) for seq in input_seqs]
    # Pad context vectors to the length of the longest vector in the batch
    max_vec_length = max(vec.size(0) for vec in context_vectors)
    padded_context_vectors = [torch.nn.functional.pad(vec, (0, 0, 0, max_vec_length - vec.size(0))) for vec in context_vectors]

    # Stack the padded sequences and context vectors to create batch tensors
    batched_targets = torch.stack(padded_targets)
    batched_inputs = torch.stack(padded_inputs)
    batched_context_vectors = torch.stack(padded_context_vectors)

    return batched_inputs, batched_targets, batched_context_vectors

#code taken from: https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
class LightAttention(nn.Module):
    def __init__(self, d_model, embeddings_dim=1024, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.information_fnn = nn.Linear(embeddings_dim*2, d_model)
        self.gate_fnn = nn.Linear(embeddings_dim*2, d_model)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        mask = create_padding_mask(x)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]

        gate = self.gate_fnn(o)
        gate = self.sigmoid(gate)

        i = self.information_fnn(o)

        o = i * gate

        return o


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_layer, dropout):
        super(DecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x,target_mask):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



#
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout, prot_emb_size, num_layers, max_seq_len):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = dropout, max_len = max_seq_len*2)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_layer, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = dropout

        self.dropout1  = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, prot_emb_size)
        #self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, per_prot_context):
        x = self.dropout1(self.embedding(x))
        x = self.pos_encoder(x)
        x = torch.cat([per_prot_context.unsqueeze(0), x], dim=0)
        for transformer_block in self.transformer_blocks:
            tgt_mask = generate_mask(x.size(0))
            tgt_mask = tgt_mask.to(x.device)
            x = transformer_block(x,tgt_mask)
        output = self.dropout2(self.linear(x))
        return output


class Normi(nn.Module):
    def __init__(self, light_attention_model, decoder_model):
        super(Normi, self).__init__()
        self.light_attention_model = light_attention_model
        self.decoder_model = decoder_model

    def forward(self, embs, seqs, **kwargs):
        # Forward pass through the Light Attention model
        light_attention_output = self.light_attention_model(embs, **kwargs)

        # Forward pass through the existing decoder model
        decoder_output = self.decoder_model(seqs, light_attention_output)

        return decoder_output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='This script trains a transformer model using an input fasta and embedding file.')

    # Define command-line arguments
    parser.add_argument('--fasta', type=str, help='Specify the path to the input fasta file.')
    parser.add_argument('--emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--outpath', type=str, default = "light_attention.pth", help='Specify the path to the output file.')
    parser.add_argument('--bs', type=int, default = 64, help='Specify the batch size.')
    parser.add_argument('--num_epochs', type=int, default = 300, help='Specify the number of epochs to train for.')
    parser.add_argument('--lr', type=float, default = 0.001, help='Specify the learning rate.')
    parser.add_argument("--dropout", type = float, default = 0.5, help = "Specify the dropout.")
    parser.add_argument("--filter", type = int, default = 512, help = "Specify the maximum seq length to keep.")
    parser.add_argument("--num_layers", type = int, default = 2, help = "Specify the number of decoder layers.")
    parser.add_argument("--heads", type = int, default = 16, help = "Specify the number of heads for multihead attention.")
    parser.add_argument("--patience", type = float, default = 5, help = "Specify patience.")
    parser.add_argument("--meta", type = str, help = "Specify path to the metadata for the embedding file.")
    parser.add_argument("--log_file", type = str, default = "la_training.log", help = "Specify the path to the logging file.")
    # Parse the command-line arguments
    args = parser.parse_args()

    log_file = args.log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)





    fasta_sequences = SeqIO.parse(args.fasta,'fasta')
    seqs = []
    fids = []

    for f in fasta_sequences:
        seq = str(f.seq)
        seqs.append(seq)
        fids.append(f.id)

    with h5py.File(args.emb, 'r') as h5_file:
        ids = list(h5_file.keys())
        embs = [np.array(h5_file[protein_id]) for protein_id in ids]

    seq_dict = dict(zip(fids, seqs))
    seqs = [seq_dict[id_emb] for id_emb in ids]

    invalid_characters = ["X", "U", "B", "Z", "O"]

    filtered_data = [
        (emb, ''.join(['X' if char in invalid_characters else char for char in seq]))
        for emb, seq in zip(embs, seqs)
        if len(seq) <= args.filter
    ]

    embs, seqs = zip(*filtered_data)
    seqs = [tokenize_seq(seq) for seq in seqs]





    custom_dataset = CustomDataset(seqs, embs)

    dataset_size = len(custom_dataset)
    train_size = int(0.75 * dataset_size)
    val_size = dataset_size - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
    logging.info(f"Training data of size: {train_size}")
    # Define batch size
    batch_size = args.bs

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)





    def objective(trial):
        num_layers = trial.suggest_categorical("num_layers", [2, 4, 8, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        vocab_size = len(aa_dic)
        d_model = 512
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16, 32])
        ff_hidden_layer = 2 * d_model
        esm_emb_size = len(custom_dataset[0][1][0])
        num_epochs = args.num_epochs
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5*1e-3)
        scheduler_patience = trial.suggest_categorical("scheduler_patience", [5, 10, 30, 10000]) #last one basically disables the scheduler
        scheduler_reduction_factor = trial.suggest_float("scheduler_reduction_factor", 0.2, 1)


        la = LightAttention(d_model = d_model, embeddings_dim= esm_emb_size, kernel_size=9, conv_dropout= dropout)
        dec = Decoder(vocab_size = vocab_size, d_model = d_model, num_heads = num_heads, ff_hidden_layer = ff_hidden_layer,
                      dropout = dropout, prot_emb_size = esm_emb_size, num_layers = num_layers, max_seq_len=args.filter)
        model = Normi(la, dec)



        model.to(device)
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=scheduler_patience, factor=scheduler_reduction_factor)

        best_val_loss = float('inf')
        patience = args.patience
        counter = 0


        for epoch in range(num_epochs):
            # Training
            model.train()
            total_train_loss = 0.0
            for batch in train_dataloader:
                # Unpack the batch
                inputs, _, context_vectors = batch
                inputs, context_vectors = (inputs.to(device), context_vectors.to(device))
                # Pass the sequences through the model
                output = model(context_vectors.permute(0, 2, 1), inputs.permute(1, 0))
                output = output[1:-1].permute(1, 0, 2)  # remove prediction for stop token and right shift
                targets = deepcopy(context_vectors)
                mask = targets != 0
                masked_outputs = output[mask]
                masked_targets = targets[mask]
                loss = criterion(masked_outputs, masked_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            average_train_loss = total_train_loss / len(train_dataloader)

            # Validation
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_inputs, val_targets, val_context_vectors = val_batch
                    val_inputs, val_targets, val_context_vectors = (val_inputs.to(device), val_targets.to(device), val_context_vectors.to(device))
                    val_output = model(val_context_vectors.permute(0, 2, 1), val_inputs.permute(1, 0))
                    val_output = val_output[1:-1].permute(1, 0, 2)
                    targets = deepcopy(val_context_vectors)
                    mask = targets != 0
                    masked_outputs = val_output[mask]
                    masked_targets = targets[mask]
                    val_loss = criterion(masked_outputs, masked_targets)
                    total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_dataloader)
            scheduler.step(average_val_loss)

            #save model in case of crashes


            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_state = deepcopy(model.state_dict())
                best_optim = deepcopy(optimizer.state_dict)
                best_epoch = epoch
                counter = 0  # Reset the counter
            else:
                counter += 1  # No improvement, increment the counter
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'optimizer_state_dict': best_optim,
                    }, args.outpath)

                # Check if training should be stopped
            if counter >= patience:
                break
        return best_val_loss
    def log_trial(study, trial):
        logging.info(f'Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}')
        logging.info(f'Best value so far: {study.best_value} with parameters: {study.best_params}')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, callbacks=[log_trial])


    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")

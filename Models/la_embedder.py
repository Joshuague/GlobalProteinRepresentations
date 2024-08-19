import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
import gzip
from transformers import T5EncoderModel, T5Tokenizer, set_seed
from copy import deepcopy
import math
import h5py
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

class CustomDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences  # Raw sequences, not tokenized

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return sequence  # Return the raw sequence as a string

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

        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]

        #attention on attention to reduce noise and project down to the size used in the decoder

        gate = self.gate_fnn(o) # [batchsize, d_model]
        gate = self.sigmoid(gate) # [batchsize, d_model]

        i = self.information_fnn(o)

        o = i * gate

        return o


"""relatively simple decoder block. Multihead Attention followed by up and downsampling the attention output"""
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



"""positional encoding for the decoder"""
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



"""The Decoder itself. The input to the forward pass is the per protein embedding from the LA module 
concatenated with the input sequence. The output is the logits for the sequence itself"""
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
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, per_prot_context):
        x = self.dropout1(self.embedding(x))
        x = self.pos_encoder(x)
        x = torch.cat([per_prot_context.unsqueeze(0), x], dim=0)
        for transformer_block in self.transformer_blocks:
            tgt_mask = generate_mask(x.size(0))
            tgt_mask = tgt_mask.to(x.device)
            x = transformer_block(x,tgt_mask)
        e_out = self.dropout2(self.linear(x))
        return e_out


"""Normi combines both the decoder and the LA model into one"""
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
    parser = argparse.ArgumentParser(description='This script trains a transformer model using an input fasta file.')

    # Define command-line arguments
    parser.add_argument('--fasta', type=str, help='Specify the path to the input fasta file.')
    parser.add_argument('--outpath', type=str, default = "light_attention.pth", help='Specify where to save the weights.')
    parser.add_argument('--bs', type=int, default = 3, help='Specify the batch size.')
    parser.add_argument('--num_epochs', type=int, default = 300, help='Specify the number of epochs to train for.')
    parser.add_argument('--lr', type=float, default = 0.00066, help='Specify the learning rate.')
    parser.add_argument("--dropout", type = float, default = 0.135, help = "Specify the dropout.")
    parser.add_argument("--filter", type = int, default = 512, help = "Specify the maximum seq length to keep.")
    parser.add_argument("--num_layers", type = int, default = 4, help = "Specify the number of decoder layers.")
    parser.add_argument("--heads", type = int, default = 16, help = "Specify the number of heads for multihead attention.")
    parser.add_argument("--patience", type = float, default = 10, help = "Specify patience.")
    parser.add_argument("--scheduler", type = bool, default = False, help = "If set to true uses the ReduceLROnPlateau scheduler.")
    parser.add_argument("--log_file", type = str, default = "la_training.log", help = "Specify the path to the logging file used during training.")
    # Parse the command-line arguments
    args = parser.parse_args()

    log_file = args.log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)

    file_path = args.fasta

    # Determine the file extension
    if file_path.endswith('.gz'):
        # Handle gzipped FASTA file
        with gzip.open(file_path, "rt") as handle:
            fasta_sequences = SeqIO.parse(handle, 'fasta')
            seqs = []
            fids = []
            for f in fasta_sequences:
                seq = str(f.seq)
                seqs.append(seq)
                fids.append(f.id)
    else:
        # Handle plain FASTA file
        with open(file_path, "r") as handle:
            fasta_sequences = SeqIO.parse(handle, 'fasta')
            seqs = []
            fids = []
            for f in fasta_sequences:
                seq = str(f.seq)
                seqs.append(seq)
                fids.append(f.id)


    # Filter out sequences with invalid characters and those that are too long
    invalid_characters = ["X", "U", "B", "Z", "O"]

    filtered_seqs = [
        ' '.join(['X' if char in invalid_characters else char for char in seq])
        for seq in seqs
        if len(seq) <= args.filter
    ]


    #loading in prott5 to embed the sequences during training
    checkpoint = "Rostlab/prot_t5_xl_half_uniref50-enc"
    embedder = T5EncoderModel.from_pretrained(checkpoint)
    embedder.to(device)
    embedder.eval()
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    custom_dataset = CustomDataset(filtered_seqs)


    dataset_size = len(custom_dataset)
    train_size = int(0.75 * dataset_size)
    val_size = dataset_size - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
    logging.info(f"Training data of size: {train_size}")
    # Define batch size
    batch_size = args.bs

    # Create DataLoaders for training and validation, no collate function since padding is handled by the T5 tokenizer later on
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    #specify hyperparameters
    num_layers = args.num_layers
    dropout = args.dropout
    vocab_size = len(aa_dic)
    d_model = 512 #hidden size of the decoder and output size of the LA
    num_heads = args.heads
    ff_hidden_layer = 2 * d_model #for upsampling
    prott5_emb_dim = 1024
    num_epochs = args.num_epochs
    learning_rate = args.lr

    logging.info("defining the model")

    la = LightAttention(d_model = d_model, embeddings_dim= prott5_emb_dim, kernel_size=9, conv_dropout= dropout)
    dec = Decoder(vocab_size = vocab_size, d_model = d_model, num_heads = num_heads, ff_hidden_layer = ff_hidden_layer,
                  dropout = dropout, prot_emb_size = prott5_emb_dim, num_layers = num_layers, max_seq_len = args.filter)
    model = Normi(la, dec)



    model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction='mean') #ignore index used to ignore padding tokens
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.9)

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    logging.info("beginning the training")


    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for batch in train_dataloader:
            # Unpack the batch
            inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_masks = inputs['attention_mask'].to(device)
            with torch.no_grad():
                context_vectors = embedder(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state

            # Pass the sequences through the model
            #dimensions of my decoder were a bit scrambled, never got to fixing it. Thats the reason for the permutes

            emb_output = model(context_vectors.permute(0, 2, 1), input_ids.permute(1, 0))
            output = emb_output[:-1].permute(1, 2, 0)  # remove prediction for stop token
            loss = criterion(output, input_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_masks = inputs['attention_mask'].to(device)
                with torch.no_grad():
                    context_vectors = embedder(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state

                # Pass the sequences through the model
                emb_output = model(context_vectors.permute(0, 2, 1), input_ids.permute(1, 0))
                output = emb_output[:-1].permute(1, 2, 0)  # remove prediction for stop token
                loss = criterion(output, input_ids)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        if args.scheduler:
            scheduler.step(average_val_loss)
            logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")



        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = deepcopy(model.state_dict())
            best_optim = deepcopy(optimizer.state_dict)
            best_epoch = epoch
            counter = 0  # Reset the counter
        else:
            counter += 1  # No improvement, increment the counter
        if epoch % 10 == 0:
            # save model in case of crashes
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'optimizer_state_dict': best_optim,
                }, args.outpath)

            # Check if training should be stopped
        if counter >= patience:
            stop_str = f'Early stopping after {patience} epochs with no improvement.'
            logging.info(stop_str)
            break
        # Print training and validation loss for the current epoch
        info_str = f"Epoch: [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}"
        logging.info(info_str)


    torch.save({"model_state_dict": best_state, "optimizer_state_dict": best_optim}, args.outpath)

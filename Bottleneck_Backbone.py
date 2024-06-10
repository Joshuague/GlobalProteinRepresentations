import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from Bio import SeqIO
import argparse
import logging
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

def get_activation_fn(name: str):
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")



class ProteinResNetBlock(nn.Module):

    def __init__(self, hidden_size, act = "relu"):
        super().__init__()
        self.conv1 = nn.Conv1d(
            hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        #self.bn1 = ProteinResNetLayerNorm(config)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        #self.bn2 = ProteinResNetLayerNorm(config)
        self.activation_fn = get_activation_fn(act)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation_fn(out)

        return out

class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class ProteinResNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
       Here i dont know what they mean by token_type embeddings. But the input is only the tokenized sequence
       so i dont think they had an additional seperate embedding.
       That still doesnt explain the 8k vocab size they use though
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_dropout_prob, use_prot_emb = False):
        super().__init__()
        embed_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        inverse_frequency = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer('inverse_frequency', inverse_frequency)
        self.projection_layer = nn.Linear(1024, hidden_size)
        self.layer_norm = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.use_prot_emb = use_prot_emb

    def forward(self, input_ids):
        if self.use_prot_emb:
            words_embeddings = self.projection_layer(input_ids) #in this case the input ids are already the prott5 emb
        else:
            words_embeddings = self.word_embeddings(input_ids)
            print(words_embeddings.shape)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length - 1, -1, -1.0,
            dtype=words_embeddings.dtype,
            device=words_embeddings.device)
        sinusoidal_input = torch.ger(position_ids, self.inverse_frequency)
        position_embeddings = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos()], -1)
        position_embeddings = position_embeddings.unsqueeze(0)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



#the pooler is not needed for the bottleneck approach

class ProteinResNetPooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        attention_scores = self.attention_weights(hidden_states)
        attention_weights = torch.softmax(attention_scores, -1)
        weighted_mean_embedding = torch.matmul(
            hidden_states.transpose(1, 2), attention_weights).squeeze(2)
        pooled_output = self.dense(weighted_mean_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output




class ResNetEncoder(nn.Module):

    def __init__(self, hidden_size, num_hidden_layers):
        super().__init__()
        self.layer = nn.ModuleList(
            [ProteinResNetBlock(hidden_size) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        outputs = (hidden_states,)
        return outputs
    

class ProteinResNetModel(nn.Module):

    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_dropout_prob, num_hidden_layers, max_size, use_prot_emb = False):
        super(ProteinResNetModel, self).__init__()
        self.use_prot_emb = use_prot_emb

        self.embeddings = ProteinResNetEmbeddings(hidden_size, vocab_size, layer_norm_eps, hidden_dropout_prob, use_prot_emb)
        self.encoder = ResNetEncoder(hidden_size, num_hidden_layers)
        #self.pooler = ProteinResNetPooler(hidden_size)

        self.linear1 = nn.Linear(max_size * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, max_size *hidden_size)

        self.mlmhead = MLMHead(hidden_size, vocab_size, hidden_act = "relu")

    def forward(self, input_ids, targets = None):
        embedding_output = self.embeddings(input_ids)
        embedding_output = embedding_output.transpose(1, 2)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output.transpose(1, 2).contiguous()
        pre_shape = sequence_output.shape
        embeddings = self.linear1(sequence_output.reshape(sequence_output.shape[0], -1))
        sequence_output = self.linear2(embeddings).reshape(*pre_shape)

        if not self.use_prot_emb:
            loss = self.mlmhead(sequence_output, input_ids)
        elif self.use_prot_emb and targets is None:
            return sequence_output, embeddings
        else:
            loss = self.mlmhead(sequence_output, targets)

        return sequence_output,embeddings, loss  #reconstructed sequence_output, per_prot_emb, loss

def tokenize_and_pad(seq, max_len, is_emb = False):
    aa_dic = {"X": 0, "<SOS>": 1, "<EOS>": 2, "A": 3, "R": 4, "N": 5, "D": 6, "C": 7, "Q": 8, "E": 9, "G": 10,
              "H": 11, "I": 12, "L": 13, "K": 14, "M": 15, "F": 16, "P": 17, "S": 18, "T": 19, "W": 20, "Y": 21,
              "V": 22}
    seq_l = len(seq)
    if max_len < seq_l:
        seq = seq[:max_len]
        seq_l = max_len
        if not is_emb:
            seq = [aa_dic[aa] for aa in seq]
    else:
        pad_len = max_len - seq_l
        if is_emb:
            emb_dim = seq.shape[1]
            seq = np.append(seq, np.zeros([pad_len, emb_dim])).reshape([-1, emb_dim])
        else:
            seq = [aa_dic[aa] for aa in seq]
            seq.extend([0]*pad_len)
    return seq,seq_l


class PredictionHeadTransform(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 hidden_act: 'relu',
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = get_activation_fn(hidden_act)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLMHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: 'relu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -100): #changed this from -100 to 0 since i use 0 for padding and special tokens
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        outputs = (hidden_states,)
        if targets is not None:
            #since targets are always my padded sequences, i exchange the 0 padding with -100 which is
            #the ignore index of the loss function
            targets[targets == 0] = -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.reshape(-1, self.vocab_size), targets.reshape(-1))
            #metrics = {'perplexity': torch.exp(masked_lm_loss)}
            #loss_and_metrics = (masked_lm_loss, metrics)
            #outputs = (masked_lm_loss,) + outputs
        return masked_lm_loss #loss

def plot_loss(loss_dic, outpath):
    matplotlib.use('Agg')
    plt.plot(loss_dic["Validation_Loss"], label='Validation Loss')
    plt.plot(loss_dic["Training_Loss"], label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outpath)
    plt.clf()

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(t_total - current_step) / float(max(1, t_total - warmup_steps))
            )

        super(WarmupLinearSchedule, self).__init__(optimizer, lr_lambda, last_epoch)



if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='This script trains a resnet bottleneck model using an input fasta and embedding file.')

    # Define command-line arguments
    parser.add_argument('--fasta', type=str, help='Specify the path to the input fasta.')
    parser.add_argument('--plot_out', type=str, default=None, help='Specify the path to the input fasta.')
    parser.add_argument('--vocab_size', type=int, default=23, help='Specify the number of tokens in your alphabet')
    parser.add_argument('--hidden_size', type=int, default=512, help='Specify the size of the hidden layer.')
    parser.add_argument('--emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--outpath', type=str, default = "resnet_bottleneck.pth", help='Specify the path to which the model weights are saved.')
    parser.add_argument('--bs', type=int, default = 16, help='Specify the batch size.')
    parser.add_argument('--num_epochs', type=int, default = 150, help='Specify the number of epochs to train for.')
    parser.add_argument('--lr', type=float, default = 0.001, help='Specify the learning rate.')
    parser.add_argument("--dropout", type = float, default = 0.5, help = "Specify the dropout.")
    parser.add_argument("--warmup", type = int, default = 20, help = "Specify the number of epochs to warm up for.")
    parser.add_argument("--filter", type = int, default = 512, help = "Specify the maximum seq length to keep.")
    parser.add_argument("--num_layers", type = int, default = 2, help = "Specify the number of decoder layers.")
    parser.add_argument("--patience", type = float, default = 20, help = "Specify patience.")
    parser.add_argument("--log_file", type = str, default = "la_training.log", help = "Specify the path to the logging file.")

    args = parser.parse_args()

    max_len = args.filter
    fasta_sequences = SeqIO.parse(args.fasta,'fasta')
    seqs = []
    ids = []
    for f in fasta_sequences:
        seqs.append(str(f.seq))
        ids.append(f.id)

    with h5py.File(args.emb, "r") as f:
        embs = [np.array(f[key]) for key in ids]



    invalid_characters = ["X", "U", "B", "Z", "O"]
    filtered_data = [
        ''.join('X' if char in invalid_characters else char for char in s)
        for s in seqs
    ]
    tokenized_fastas = [tokenize_and_pad(s, max_len) for s in filtered_data]
    padded_embs = [tokenize_and_pad(e, max_len, is_emb=True)[0] for e in embs]

    lengths = torch.Tensor([s[1] for s in tokenized_fastas]).type(torch.LongTensor)
    tokenized_fastas = [s[0] for s in tokenized_fastas]

    X_tensor = torch.Tensor(tokenized_fastas).type(torch.LongTensor)
    X_embedding_tensor = torch.Tensor(np.array(padded_embs)).type(torch.FloatTensor)
    dataset = TensorDataset(X_tensor,  X_embedding_tensor, lengths)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size = args.bs)
    val_dataloader = DataLoader(val_dataset, batch_size = args.bs)
    model = ProteinResNetModel(num_hidden_layers = args.num_layers, hidden_size= args.hidden_size, vocab_size = args.vocab_size, layer_norm_eps=0.001, hidden_dropout_prob=args.dropout, max_size = max_len, use_prot_emb=True)


    num_epochs = args.num_epochs
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.warmup != None:
        warmup_steps = len(dataloader)*args.warmup
        t_total = len(dataloader)*num_epochs - warmup_steps
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps, t_total)


    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    model.to(device)
    all_losses = {"Validation_Loss":[], "Training_Loss":[]}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_val_loss = 0
        for batch, emb_batch, batch_len in dataloader:
            batch, emb_batch, batch_len = (batch.to(device), emb_batch.to(device), batch_len.to(device))
            sequence_output,embeddings, loss = model(emb_batch, batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup != None:
                scheduler.step()
        model.eval()
        for batch, emb_batch, batch_len in val_dataloader:
            batch, emb_batch, batch_len = (batch.to(device), emb_batch.to(device), batch_len.to(device))
            sequence_output,embeddings, loss = model(emb_batch, batch)
            epoch_val_loss += loss.item()

        all_losses["Validation_Loss"].append(epoch_val_loss/len(val_dataloader))
        all_losses["Training_Loss"].append(epoch_loss/len(dataloader))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            counter = 0  # Reset the counter
        else:
            counter += 1  # No improvement, increment the counter

            # Check if training should be stopped
        if counter >= patience:
            stop_str = f'Early stopping after {patience} epochs with no improvement.'
            logging.info(stop_str)
            break
        if (epoch+1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, args.outpath)
            if not args.plot_out == None:
                plot_loss(all_losses, args.plot_out)


        print(f"Epoch {epoch}, Training Loss: [{epoch_loss/len(dataloader)}], Validation Loss: [{epoch_val_loss/len(val_dataloader)}]")

    torch.save({"model_state_dict": best_state}, args.outpath)
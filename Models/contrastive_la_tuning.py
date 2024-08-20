from Models.la_transformer import LightAttention, Decoder, Normi
from copy import deepcopy
import torch.nn as nn
import torch
import h5py
from torch.utils.data import DataLoader, Dataset
import argparse
import logging

"""This script was used to tune the la_transformer decoder on the contrastive loss. It didnt really work. This 
script requires an embedding file path and weights for the Normi model from the la_transformer script. The default 
parameters are not tuned but rather the hyperparameters for the LA model. The hyperparmeter tuning results were lost on the 
cluster."""

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_file):
        # Load all embeddings into memory
        with h5py.File(embedding_file, 'r') as h5_file:
            self.embeddings = [torch.tensor(h5_file[key][()]).to(torch.float) for key in h5_file.keys()]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


"""Just a wrapper for the LA model."""
class ContrastiveLearningModel(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        x = x.permute(0,2,1)
        return self.encoder(x)


"""My implementation of the loss proposed in: https://arxiv.org/abs/2104.08821 paper. Doing this in a vectorized manner instead of looping speeds it up quite a bit,
but makes it numerically unstable for low temperatures ~ < 0.01"""
def nt_xent_loss(emb, z_emb, temp=0.5):
    """
    Parameters:
    emb (torch.Tensor): Tensor of embeddings (batch_size x embedding_dim)
    z_emb (torch.Tensor): Tensor of corresponding embeddings (batch_size x embedding_dim)
    temp (float): Temperature parameter for scaling.

    Returns:
    torch.Tensor: Computed loss.
    """
    batch_size = emb.shape[0]

    emb = torch.nn.functional.normalize(emb, dim=1)
    z_emb = torch.nn.functional.normalize(z_emb, dim=1)
    sim_matrix = torch.exp(torch.matmul(emb, z_emb.T)/temp)
    # Create a mask to zero out the diagonal (self-similarities)
    mask = torch.eye(batch_size, dtype=torch.bool).to(sim_matrix.device)

    # Select the positive pairs
    positives = sim_matrix[mask].view(batch_size, -1)


    # Select the negative pairs and compute the denominator (commented out is for when we want to exclude self similarity)

    #negatives = sim_matrix[~mask].view(batch_size, batch_size - 1)
    #denom = negatives.sum(dim=1, keepdim=True)

    denom = sim_matrix.sum(dim=1, keepdim=True)
    # Compute the loss
    loss = -torch.log(positives / denom).mean()

    return loss


def collate_fn(batch):
    # Pad the sequences to the same length
    embeddings = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return embeddings

def enable_dropout(model):
    """Enable the dropout layers during evaluation, important since we need dropout to calculate a meaningful loss"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


if __name__ == "__main__":



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='This script trains a transformer model using an input fasta and embedding file.')

    # Define command-line arguments
    parser.add_argument('--emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--outpath', type=str, default="contrastive_results/contrastive.pth",
                        help='Specify where to save the weights.')
    parser.add_argument('--bs', type=int, default=2, help='Specify the batch size.')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Specify the number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.001, help='Specify the learning rate.')
    parser.add_argument('--temperature', type=float, default=0.05, help='Specify the temperature for the loss.')
    parser.add_argument("--dropout", type=float, default=0.135, help="Specify the dropout.")
    parser.add_argument("--filter", type=int, default=512, help="Specify the maximum seq length to keep.")
    parser.add_argument("--num_layers", type=int, default=4, help="Specify the number of decoder layers.")
    parser.add_argument("--heads", type=int, default=32, help="Specify the number of heads for multihead attention.")
    parser.add_argument("--patience", type=float, default=10, help="Specify patience.")
    parser.add_argument("--meta", type=str, help="Specify path to the metadata for the embedding file.")
    parser.add_argument("--log_file", type=str, default="la_training.log", help="Specify the path to the logging file.")
    parser.add_argument("--pretrained", type=str, default="la")
    parser.add_argument("--weights", type=str, default="Weights/la_embedder.pth")
    parser.add_argument("--accumulation_steps", type=int, default = 1)

    # Parse the command-line arguments
    args = parser.parse_args()


    log_file = args.log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)
    batch_size = args.bs
    emb_path = args.emb
    vocab_size = 23
    hidden_size = 512
    num_heads = args.heads
    ff_hidden_layer = hidden_size * 2
    dropout = args.dropout
    prot_emb_size = 1024
    num_layers = args.num_layers
    grad_accum = args.accumulation_steps
    learning_rate = args.lr
    temp = args.temperature
    dataset = EmbeddingDataset(emb_path)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dat, val_dat = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dat, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_dat, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dec = Decoder(vocab_size, hidden_size, num_heads, ff_hidden_layer, dropout, prot_emb_size, num_layers,
                  args.filter)
    la = LightAttention(d_model=hidden_size, embeddings_dim=prot_emb_size, kernel_size=9, conv_dropout=dropout)
    _ = Normi(la, dec)
    checkpoint = torch.load(args.weights)

    """strict is set to false here to allow the loading in of slightly modified Normi versions I tested out. 
    The LA module stayed consistent throughout them"""

    _.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = ContrastiveLearningModel(la)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    logging.info("starting the training")

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        for i, batch in enumerate(train_loader):
            embeddings = batch.to(device)
            emb = model(embeddings)
            z_emb = model(embeddings)
            loss = nt_xent_loss(emb, z_emb, temp=temp)
            train_loss += loss.item()
            loss.backward()
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        enable_dropout(model)

        for batch in val_loader:
            embeddings = batch.to(device)
            emb = model(embeddings)
            z_emb = model(embeddings)
            loss = nt_xent_loss(emb, z_emb, temp=temp)
            val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
        else:
            counter += 1
        if counter >= patience:
            stop_str = f'Early stopping after {patience} epochs with no improvement.'
            logging.info(stop_str)
            break
        if (epoch + 1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, args.outpath)
        info_str = f"Epoch: [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss * 1000:.4f}, Validation Loss: {val_loss * 1000:.4f}"
        logging.info(info_str)
    torch.save({"model_state_dict": best_state}, args.outpath)






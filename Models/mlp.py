import torch.nn as nn
import torch
from typing import Union
from pathlib import Path
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
import lmdb
import h5py
import numpy as np
import argparse
from copy import deepcopy
from scipy.stats import spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import logging
import optuna
import re
import pandas as pd

def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()
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


"""Different LA as in the la_embedder.py script. Instead of adding the feature convoluted mean embeddings to the attention,
I add the mean. Was slightly better in one case, never got to test it properly though as running the LA pooler is relatively
computationally expensive compared to running this script with already pooled embeddings"""
class LightAttention(nn.Module):
    def __init__(self, d_model, embeddings_dim=1024, kernel_size=9, conv_dropout: float = 0.25, lin_dropout = 0.1):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(lin_dropout)

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
        skip_connect = x.mean(dim=2)

        mask = create_padding_mask(x)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, skip_connect], dim=-1)  # [batchsize, 2*embeddings_dim]

        gate = self.gate_fnn(o)
        gate = self.sigmoid(gate)

        i = self.dropout2(self.information_fnn(o))

        o = i * gate
    #torch.cat([o, skip_connect], dim = -1)
        return o


"""The classifier and regression head were taken from the 
https://github.com/MachineLearningLifeScience/meaningful-protein-representations git. As was the function to 
generate the LMDB datasets"""
class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            torch.nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            torch.nn.utils.parametrizations.weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)

class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0., pooler: str = "None"):
        super().__init__()
        self.pooler = pooler
        if pooler == "Mean+SD":
            self.value_prediction = SimpleMLP(hidden_size*2, 512, 1, dropout)
        else:
            self.value_prediction = SimpleMLP(hidden_size, 512, 1, dropout)
        self.la = LightAttention(hidden_size, conv_dropout=dropout)

    def forward(self, embeddings, targets=None):
        if self.pooler == "None":
            pooled_output = embeddings
        elif self.pooler == "Mean":
            pooled_output = embeddings.mean(dim = 1)
        elif self.pooler == "PCA":
            embeddings = embeddings.permute(0,2,1)
            _, _, v = torch.pca_lowrank(embeddings)
            pooled_output = torch.matmul(embeddings, v[:,:,:1])
            pooled_output = pooled_output.squeeze(2)
        elif self.pooler == "LA":
            pooled_output = self.la(embeddings.permute(0,2,1))
        elif self.pooler == "Mean+SD":
            mean = embeddings.mean(dim=1)
            std = embeddings.std(dim=1)
            pooled_output = torch.cat((mean, std), dim=1)
        else:
            raise ValueError(f"Unsupported pooler: {self.pooler}, supported options are: None, Mean, PCA, LA, Mean+SD")
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            loss_and_metrics = (value_pred_loss, None) #this is so the shape is equal to the Sequence ClassificationHead
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss, None), value_prediction


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0., pooler: str = "None"):
        super().__init__()
        self.pooler = pooler
        if pooler == "Mean+SD":
            self.classify = SimpleMLP(hidden_size*2, 512, num_labels, dropout)
        else:
            self.classify = SimpleMLP(hidden_size, 512, num_labels, dropout)
        self.la = LightAttention(hidden_size, conv_dropout=dropout)

    def forward(self, embeddings, targets=None):
        if self.pooler == "None":
            pooled_output = embeddings
        elif self.pooler == "Mean":
            pooled_output = embeddings.mean(dim = 1)
        elif self.pooler == "PCA":
            embeddings = embeddings.permute(0, 2, 1)
            _, _, v = torch.pca_lowrank(embeddings)
            pooled_output = torch.matmul(embeddings, v[:, :, :1])
            pooled_output = pooled_output.squeeze(2)
        elif self.pooler == "LA":
            pooled_output = self.la(embeddings.permute(0,2,1))
        elif self.pooler == "Mean+SD":
            mean = embeddings.mean(dim=1)
            std = embeddings.std(dim=1)
            pooled_output = torch.cat((mean, std), dim=1)
        else:
            raise ValueError(f"Unsupported pooler: {self.pooler}")

        logits = self.classify(pooled_output)
        outputs = (logits,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs


        return outputs

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item

"""The following data stuff is only used with the meltome and localization datasets."""
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create custom collate function
def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    labels = torch.stack(labels)
    return embeddings, labels


"""Reading in my processed fasta files for the localization and meltome dataset"""
def parse_and_save_sequences(file_path):
    # Initialize the sets
    train_set = []
    validation_set = []
    test_set = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        seq_id = None
        sequence = []
        target = None
        set_type = None
        validation = None

        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # If there's an existing sequence, save it to the appropriate set
                if seq_id and sequence:
                    seq_str = ''.join(sequence)
                    if set_type == "train" and not validation:
                        train_set.append((seq_id, target, seq_str))
                    elif set_type == "train" and validation:
                        validation_set.append((seq_id, target, seq_str))
                    elif set_type == "test":
                        test_set.append((seq_id, target, seq_str))

                # Extract the relevant parts from the new header line
                parts = line.split()
                seq_id = parts[0][1:]
                target = parts[1].split('=')[1]
                set_type = parts[2].split('=')[1]
                validation = parts[3].split('=')[1] == "True"
                sequence = []  # Reset sequence for the new entry
            else:
                # Accumulate sequence lines
                sequence.append(line)

        # Don't forget to save the last sequence in the file
        if seq_id and sequence:
            seq_str = ''.join(sequence)
            if set_type == "train" and not validation:
                train_set.append((seq_id, target, seq_str))
            elif set_type == "train" and validation:
                validation_set.append((seq_id, target, seq_str))
            elif set_type == "test":
                test_set.append((seq_id, target, seq_str))

    return train_set, validation_set, test_set



"""this may seem too complicated, that is because it is. I did not embed all my sequences, as i removed sequences longer
than 2000 AA. So not every id  has an entry in the h5 file. Conversion to float 16 is due to memory issues with the 
relatively large meltome dataset."""
def get_embeddings(hdf5_file, ids, labels):
    embeddings = []
    matched_labels = []

    with h5py.File(hdf5_file, 'r') as f:
        for i, (id_, label) in enumerate(zip(ids, labels)):
            if id_ in f:
                embeddings.append(np.array(f[id_][:], dtype=np.float16))  # Convert to numpy array
                matched_labels.append(float(label))

    return embeddings, matched_labels


"""Small bootstrapping function"""
def bootstrap(targets, predictions, metric = "spearman", n_bootstrap=10000):
    n = len(targets)
    metrics = []

    for _ in range(n_bootstrap):
        if metric == "spearman":
            # Generate bootstrap sample
            indices = np.random.choice(np.arange(n), size=n, replace=True)
            bootstrap_targets = targets[indices]
            bootstrap_predictions = predictions[indices]

            # Calculate Spearman correlation
            spearman_corr, _ = spearmanr(bootstrap_targets, bootstrap_predictions)
            metrics.append(spearman_corr)
        elif metric == "accuracy":
            predictions_2 = predictions.argmax(-1)
            # Generate bootstrap sample
            indices = np.random.choice(np.arange(n), size=n, replace=True)
            bootstrap_targets = targets[indices]
            bootstrap_predictions = predictions_2[indices]


            correct = (bootstrap_predictions == bootstrap_targets)
            metrics.append(correct.sum()/len(bootstrap_targets))
    return metrics


#mean+ sd
#learning the embeddings instead of tokens

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='This script trains a Classifier.')

    """If the input data is just a fasta and a h5 file and not three lmdb directories. Only the train_lmdb for the fasta
    and train_emb for the embedding is used. Out_name is the prefix of the png file the regression's scatter plot gets saved in
    and the column name for the bootstrapped values to be saved in."""
    # Define command-line arguments
    parser.add_argument('--train_lmdb', type=str, help='Specify the path to the input lmdb directory.')
    parser.add_argument('--test_lmdb', type=str, help='Specify the path to the input lmdb directory.')
    parser.add_argument('--val_lmdb', type=str, help='Specify the path to the input lmdb directory.')
    parser.add_argument('--out_name', type=str, default="meltome_contrastive_concat_mean", help='Specify the path to the input fasta.')
    parser.add_argument('--vocab_size', type=int, default=23, help='Specify the number of tokens in your alphabet')
    parser.add_argument('--train_emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--test_emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--val_emb', type=str, help='Specify the path to the input embedding h5.')
    parser.add_argument('--outpath', type=str, default=None,
                        help='Specify the path to which the model weights are saved.')
    parser.add_argument('--pooler', type=str, default="Mean",
                        help='Specify the pooler for the per residue embeddings'
                             'options are: [Mean, LA, PCA, Mean+SD].')
    parser.add_argument('--bs', type=int, default=6, help='Specify the batch size.')
    parser.add_argument('--num_epochs', type=int, default=5000, help='Specify the number of epochs to train for.')
    parser.add_argument('--num_labels', type=int, default=1195, help='Specify the number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Specify the learning rate.')
    parser.add_argument("--dropout", type=float, default=0.4, help="Specify the dropout.")
    parser.add_argument("--patience", type=float, default=5, help="Specify patience.")
    parser.add_argument("--log_file", type=str, default="classifier.log",
                        help="Specify the path to the logging file.")
    parser.add_argument("--tune_hyperparams", type=bool, default=False)
    parser.add_argument("--accum", type=int, default=1)

    args = parser.parse_args()
    log_file = args.log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)

    #read in data  training set:

    data_type = args.train_lmdb.split("\\")[1]


    if data_type == "Stability":
        train_data = LMDBDataset(data_file=args.train_lmdb)
        train_label = [entry["stability_score"][0] for entry in train_data]
        num_labels = 1 #regression task

        with h5py.File(args.train_emb, 'r') as h5_file:
            ids = list(range(len(train_data)))  #this is done to preserve the embedding order
            ids = [str(i) for i in ids]
            train_embs = [np.array(h5_file[protein_id]) for protein_id in ids]

        val_data = LMDBDataset(data_file=args.val_lmdb)
        val_label = [entry["stability_score"][0] for entry in val_data]

        with h5py.File(args.val_emb, 'r') as h5_file:
            ids = list(range(len(val_data)))  #this is done to preserve the embedding order
            ids = [str(i) for i in ids]
            val_embs = [np.array(h5_file[protein_id]) for protein_id in ids]

        test_data = LMDBDataset(data_file=args.test_lmdb)
        test_label = [entry["stability_score"][0] for entry in test_data]

        with h5py.File(args.test_emb, 'r') as h5_file:
            ids = list(range(len(test_data)))  #this is done to preserve the embedding order
            ids = [str(i) for i in ids]
            test_embs = [np.array(h5_file[protein_id]) for protein_id in ids]


    elif data_type == "meltome":
        num_labels = 1
        train_set, validation_set, test_set = parse_and_save_sequences(args.train_lmdb)

        train_ids, train_label, train_seqs = zip(*[(item[0], item[1], item[2]) for item in train_set])
        val_ids, val_label, val_seqs = zip(*[(item[0], item[1], item[2]) for item in validation_set])
        test_ids, test_label, test_seqs = zip(*[(item[0], item[1], item[2]) for item in test_set])

        train_embs, train_label = get_embeddings(args.train_emb, train_ids, train_label)
        test_embs, test_label = get_embeddings(args.train_emb, test_ids, test_label)
        val_embs, val_label = get_embeddings(args.train_emb, val_ids, val_label)


    elif data_type == "localization":
        num_labels = 10
        train_set, validation_set, test_set = parse_and_save_sequences(args.train_lmdb)

        train_ids, train_label, train_seqs = zip(*[(item[0], item[1], item[2]) for item in train_set])
        val_ids, val_label, val_seqs = zip(*[(item[0], item[1], item[2]) for item in validation_set])
        test_ids, test_label, test_seqs = zip(*[(item[0], item[1], item[2]) for item in test_set])

        train_embs, train_label = get_embeddings(args.train_emb, train_ids, train_label)
        test_embs, test_label = get_embeddings(args.train_emb, test_ids, test_label)
        val_embs, val_label = get_embeddings(args.train_emb, val_ids, val_label)


    # Convert lists to tensors with float dtype
    if num_labels == 1:
        train_labels_tensor = torch.tensor(np.array(train_label), dtype=torch.float32).unsqueeze(1)
        val_labels_tensor = torch.tensor(np.array(val_label), dtype=torch.float32).unsqueeze(1)
        test_labels_tensor = torch.tensor(np.array(test_label), dtype=torch.float32).unsqueeze(1)
    else:
        train_labels_tensor = torch.tensor(np.array(train_label), dtype=torch.long)
        val_labels_tensor = torch.tensor(np.array(val_label), dtype=torch.long)
        test_labels_tensor = torch.tensor(np.array(test_label), dtype=torch.long)


    # Assuming train_embs, val_embs, test_embs are lists of numpy arrays with shape (seq_length, hidden_size)
    train_embs_tensor = [torch.tensor(e, dtype=torch.float32) for e in train_embs]
    val_embs_tensor = [torch.tensor(e, dtype=torch.float32) for e in val_embs]
    test_embs_tensor = [torch.tensor(e, dtype=torch.float32) for e in test_embs]

    # Create TensorDataset objects
    train_dataset = EmbeddingDataset(train_embs_tensor, train_labels_tensor)
    val_dataset = EmbeddingDataset(val_embs_tensor, val_labels_tensor)
    test_dataset = EmbeddingDataset(test_embs_tensor, test_labels_tensor)
    num_epochs = args.num_epochs
    patience = args.patience
    batch_size = args.bs

    def objective(trial):
        # Suggest values for the learning rate and dropout
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.6)
        accumulation_steps = trial.suggest_categorical("accumulation_steps", [1, 2, 4, 8])

        # Run the original training loop with the suggested hyperparameters
        if len(val_embs_tensor[0].shape) == 2:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            if num_labels == 1:  # regression task
                classifier = ValuePredictionHead(hidden_size=len(val_embs_tensor[0][0]), dropout=dropout,
                                                 pooler=args.pooler)
            else:
                classifier = SequenceClassificationHead(hidden_size=len(val_embs_tensor[0][0]), num_labels=num_labels,
                                                        dropout=dropout, pooler=args.pooler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if num_labels == 1:  # regression task
                classifier = ValuePredictionHead(hidden_size=len(val_embs_tensor[0]), dropout=dropout, pooler="None")
            else:
                classifier = SequenceClassificationHead(hidden_size=len(val_embs_tensor[0]), num_labels=num_labels,
                                                        dropout=dropout, pooler="None")

        classifier.to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            classifier.train()
            epoch_loss = 0
            epoch_val_loss = 0
            for i, (input, target) in enumerate(train_loader):
                input, target = input.to(device), target.to(device)
                (loss, _), _ = classifier(input, target)
                epoch_loss += loss.item()
                loss = loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            classifier.eval()
            for input, target in val_loader:
                input, target = input.to(device), target.to(device)
                (loss, _), _ = classifier(input, target)
                epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

        return best_val_loss


    # Create the Optuna study


    if args.tune_hyperparams:
        logging.info("Beginning the Trials")
        def logging_callback(study, trial):
            logging.info(f'Trial {trial.number}: Value={trial.value}, Params={trial.params}, State={trial.state}')
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, callbacks=[logging_callback])

        # Save the best hyperparameters
        trial = study.best_trial
        logging.info(f"Best hyperparameters: {trial.params}")
        logging.info(f"Best validation loss: {trial.value}")

        # Retrain the model with the best hyperparameters and save the final model
        learning_rate = trial.params['learning_rate']
        dropout = trial.params['dropout']
        accumulation_steps = trial.params["accumulation_steps"]
        print(f"starting training with dropout: {dropout}, learning rate: {learning_rate} and effective batch size of {args.bs*accumulation_steps}")
    else:
        learning_rate = args.lr
        dropout = args.dropout
        accumulation_steps = args.accum




    # Run the original training loop with the best hyperparameters
    if len(val_embs_tensor[0].shape) == 2:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        if num_labels == 1:  # regression task
            classifier = ValuePredictionHead(hidden_size=len(val_embs_tensor[0][0]), dropout=dropout,
                                             pooler=args.pooler)
        else:
            classifier = SequenceClassificationHead(hidden_size=len(val_embs_tensor[0][0]), num_labels=num_labels,
                                                    dropout=dropout, pooler=args.pooler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if num_labels == 1:  # regression task
            classifier = ValuePredictionHead(hidden_size=len(val_embs_tensor[0]), dropout=dropout, pooler="None")
        else:
            classifier = SequenceClassificationHead(hidden_size=len(val_embs_tensor[0]), num_labels=num_labels,
                                                    dropout=dropout, pooler="None")

    classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    counter = 0
    logging.info(f"Starting the Training for {data_type} prediction using the {classifier.pooler} pooler:")

    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0
        epoch_val_loss = 0
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            (loss, _), _ = classifier(input, target)
            epoch_loss += loss.item()
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        classifier.eval()
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            (loss, _), _ = classifier(input, target)
            epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch: {epoch}, Training Loss: {loss / len(train_loader)}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(classifier.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break


    if not args.outpath == None:
        torch.save({"model_state_dict": best_state}, args.outpath)

    classifier.load_state_dict(best_state)
    classifier.eval()
    targets = []
    predictions = []
    accuracies = []
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            (_, batch_accuracy), prediction = classifier(input, target)
            if num_labels == 1:
                targets.extend(target.squeeze(1).cpu().numpy())  # Move targets to CPU and convert to numpy
                predictions.extend(prediction.squeeze(1).cpu().numpy())
            else:
                targets.extend(target.cpu().numpy())
                predictions.extend(prediction.cpu().numpy())
                accuracies.append(batch_accuracy["accuracy"].cpu().numpy())



    if num_labels == 1:
        spearman_corrs = bootstrap(np.array(targets), np.array(predictions), metric = "spearman")

        df = pd.DataFrame({args.out_name: spearman_corrs})
        csv_path = 'bootstrapped_stability.csv'

        # Check if the CSV file exists
        try:
            existing_df = pd.read_csv(csv_path)
            # Concatenate new data with existing data
            combined_df = pd.concat([existing_df, df], axis=1)
        except FileNotFoundError:
            # If the file does not exist, create it with the new data
            combined_df = df

        combined_df.to_csv(csv_path, index=False)

        # Create scatter plot with point density coloring
        xy = np.vstack([targets, predictions])


        z = gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(targets, predictions, c=-z, cmap='viridis', s=50)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Scatter Plot of Actual vs Predicted Values with Point Density Coloring')

        # Add regression line
        z_fit = np.polyfit(targets, predictions, 1)
        p = np.poly1d(z_fit)
        plt.plot(targets, p(targets), "r--", label='Regression Line')

        # Add Spearman correlation text
        original_spearman_corr, _ = spearmanr(targets, predictions)
        plt.text(min(targets), max(predictions) * 0.97, f'Spearman Correlation: {original_spearman_corr:.4f}',
                 fontsize=10, color='red')

        plt.savefig(f"{args.out_name}.png")

    else:
        bootstrapped_accuracies  = bootstrap(np.array(targets), np.array(predictions), metric = "accuracy")
        logging.info(f"Accuracy: {np.mean(accuracies):.4f}")
        print(f"Accuracy: {np.mean(accuracies):.4f}")
        df = pd.DataFrame({args.out_name: bootstrapped_accuracies})
        csv_path = 'bootstrapped_localization.csv'

        try:
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df], axis=1)
        except FileNotFoundError:
            combined_df = df

        combined_df.to_csv(csv_path, index=False)






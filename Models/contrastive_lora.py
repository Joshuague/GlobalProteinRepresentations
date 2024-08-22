import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import re
import numpy as np
import copy
import random
import argparse
from Bio import SeqIO

from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers import T5EncoderModel, T5Tokenizer, set_seed

from peft import inject_adapter_in_model, LoraConfig

"""I only ran this script on a small set locally to see if it runs. I can run it with a batch size of 4 using my 
12 gb nvidea gpu. It returns the model state dic. Alot of the code here is taken from the 
 https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/PT5_LoRA_Finetuning_per_prot.ipynb github and modified
 to fit to my specific task. Output of the script (the model state dict as best_checkpoint.pt file and the loss history 
 as history.csv is saved to the --outpath directory."""


"""I've decided to first try to just tune lora and not the light attention head. But the helper functions of the 
LA are already in here (e.g. create_padding_mask). It should be pretty simple to just add the LA into the 
T5EncoderLAHead class and run the entire thing again."""

class LAConfig:
    def __init__(self, dropout=0.2, out_dim=1024, kernel_size=9):
        self.dropout_rate = dropout
        self.out_dim = out_dim
        self.kernel_size = kernel_size


def create_padding_mask(sequences):
    padding_mask = torch.zeros_like(sequences[:, 0, :], dtype=torch.bool)
    padding_mask[sequences[:, 0, :] != 0] = True
    return padding_mask


class T5EncoderLAHead(nn.Module):
    def __init__(self, config, la_config):
        super(T5EncoderLAHead, self).__init__()
        self.dense = nn.Linear(2 * config.d_model, la_config.out_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        pooled_output = torch.cat((mean, std), dim=1)
        o = self.dense(pooled_output)
        return o


"""My implementation of the loss proposed in: https://arxiv.org/abs/2104.08821 paper"""
def nt_xent_loss(emb, z_emb, temp=0.05):
    batch_size = emb.shape[0]
    emb = torch.nn.functional.normalize(emb, dim=1)
    z_emb = torch.nn.functional.normalize(z_emb, dim=1)
    sim_matrix = torch.exp(torch.matmul(emb, z_emb.T) / temp)
    mask = torch.eye(batch_size, dtype=torch.bool).to(sim_matrix.device)
    positives = sim_matrix[mask].view(batch_size, -1)
    denom = sim_matrix.sum(dim=1, keepdim=True)
    loss = -torch.log(positives / denom).mean()
    return loss

"""this creates the two embedding sets and calculates the loss."""
class T5EncoderForSimpleContrastiveLearning(T5PreTrainedModel):
    def __init__(self, config: T5Config, la_config):
        super().__init__(config)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.dropout = nn.Dropout(la_config.dropout_rate)
        self.pooler = T5EncoderLAHead(config, la_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        temperature=0.05,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs2 = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states1 = outputs1[0]
        hidden_states2 = outputs2[0]

        per_prot_1 = self.pooler(hidden_states1)
        per_prot_2 = self.pooler(hidden_states2)

        logits = (per_prot_1 + per_prot_2) / 2

        loss = nt_xent_loss(per_prot_1, per_prot_2, temperature)
        return loss, logits


"""loads in the prott5 model and injects the peft adapter"""
def load_T5_model(checkpoint, out_dim, dropout, t5_dropout=0.1):
    model = T5EncoderModel.from_pretrained(checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model.config.dropout_rate = t5_dropout
    class_config = LAConfig(out_dim=out_dim, dropout=dropout)
    class_model = T5EncoderForSimpleContrastiveLearning(model.config, class_config)
    class_model.shared = model.shared
    class_model.encoder = model.encoder
    model = class_model
    del class_model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_Classfier\nTrainable Parameter: " + str(params))
    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["q", "k", "v", "o"]
    )
    model = inject_adapter_in_model(peft_config, model)
    for (param_name, param) in model.pooler.named_parameters():
        param.requires_grad = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_LoRA_Classfier\nTrainable Parameter: " + str(params) + "\n")
    return model, tokenizer



def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, tokenizer):
        tokenized = tokenizer(seqs, padding=True)
        self.input_ids = torch.tensor(tokenized['input_ids'])
        self.attention_masks = torch.tensor(tokenized['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx]



"""need to enable dropout after setting the model to evaluation mode to properly calculate validation loss"""
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

"""helper function to save history on the lrz cluster"""
def save_history_to_csv(train_history, output_csv_path):
    keys = train_history.keys()
    with open(output_csv_path, mode='w') as file:
        header = ','.join(keys) + '\n'
        file.write(header)
        num_rows = len(next(iter(train_history.values())))
        for i in range(num_rows):
            row = ','.join(str(train_history[key][i]) for key in keys) + '\n'
            file.write(row)

def train_per_protein(
        checkpoint,
        train_fasta,
        outpath,
        dropout=0.25,
        patience=10,
        batch=4,
        val_batch=16,
        epochs=10,
        lr=3e-4,
        temperature=0.05,
        seed=42,
        max_len=20):

    # Set all random seeds
    set_seeds(seed)

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_T5_model(checkpoint, out_dim=1024, dropout=dropout)
    model.to(device)
    print("loading in sequences")

    sequences = []
    with open(train_fasta, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))
    special_chars = re.compile('[OBUZJ]')
    sequences = [" ".join(special_chars.sub('X', s))[:max_len] for s in sequences]

    random.shuffle(sequences)
    train_sz = int(len(sequences) * 0.75)
    train_data = sequences[:train_sz]
    val_data = sequences[train_sz:]

    train_data = CustomDataset(tokenizer=tokenizer, seqs=train_data)
    val_data = CustomDataset(tokenizer=tokenizer, seqs=val_data)
    train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_history = {'loss': [], 'epoch': [], 'eval_loss': []}
    best_val_loss = 1e9
    counter = 0

    print("starting the training")
    for epoch in range(epochs):
        train_history['epoch'].append(epoch)
        model.train()
        train_epoch_loss = 0
        val_epoch_loss = 0
        for input_ids, attention_masks in train_dataloader:
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            loss, _ = model(input_ids=input_ids, attention_mask=attention_masks, temperature=temperature)
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_epoch_loss = train_epoch_loss / len(train_dataloader)
        train_history['loss'].append(train_epoch_loss)

        model.eval()
        enable_dropout(model)
        for input_ids, attention_masks in val_dataloader:
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            with torch.no_grad():
                loss, _ = model(input_ids=input_ids, attention_mask=attention_masks, temperature=temperature)
            val_epoch_loss += loss.item()
        val_epoch_loss = val_epoch_loss / len(val_dataloader)
        train_history['eval_loss'].append(val_epoch_loss)
        if val_epoch_loss < best_val_loss:
            print(f"epoch {epoch}, new best eval loss: {val_epoch_loss}")
            best_val_loss = val_epoch_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(outpath, "best_checkpoint.pt"))
        else:
            counter += 1
            if counter == patience:
                print(f'Early Stopping at epoch {epoch}')
                break
        print(f"epoch {epoch}, train loss: {train_epoch_loss}, eval loss: {val_epoch_loss}")

    save_history_to_csv(train_history, os.path.join(outpath, "history.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--val_batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()
    train_per_protein(
        args.checkpoint,
        args.fasta,
        args.outpath,
        dropout=args.dropout,
        patience=args.patience,
        batch=args.batch,
        val_batch=args.val_batch,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
        max_len=args.max_len)

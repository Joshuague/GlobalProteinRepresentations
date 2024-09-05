# GlobalProteinRepresentations

A quick summary of the work i did during my Fortgeschrittenen-Praktikum at the RostLab. The Model folder contains the code to train the light attention (LA) decoder, the contrastive LoRA model and the MLP used for the downstream prediction tasks utilizing the precomputed embeddings.

### Datasets

The different datasets used to compare the embeddings on per protein prediction tasks were: <br />

**Meltome**: The mixed split which can be found at: [Meltome data](https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome) <br />

**Stability**: LMDB datasets provided in: [Stability data](https://github.com/songlab-cal/tape?tab=readme-ov-file#lmdb-data) <br /> 

**Localization**: The original dataset is the five split DeepLoc 2.0 data, found on their page: [DeepLoc 2.0 data](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) 
              Of that dataset I used the first four splits for training and the fifth for validation. As the test set, I use the setHard data
              proposed here: [setHard data](https://github.com/HannesStark/protein-localization/tree/master/data_files). The Localization data was also processed by me to be in the same format
              as the Meltome dataset (See utilities.ipynb). <br />

### Creation of new Embeddings

There are two different models in the "Models" directory: the LA decoder and a contrastive LoRA model, both models were trained on a 50 percent redundancy reduced SwissProt. The LA model is similar to the architecture proposed in the 2021 paper [**"Light Attention Predicts Protein Location from the Language of Life"**](https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1) by Hannes Stärk, Christian Dallago, Michael Heinzinger, and Burkhard Rost. Combining the LA pooler with a decoder-only transformer to translate per-protein embeddings back to the original sequence showed some promising results. The idea is that the resulting intermediate LA embeddings capture additional sequential context that is lost when using mean embeddings.

To create the new fixed-size embeddings, I input the per-amino-acid embeddings into the trained model and extracted the output of the LA pooler as a new per protein embedding. This embedding is then appended the ProtT5 mean embeddings, to enhance the mean embeddings sequential information. The code for converting ProtT5 per-residue embeddings to the new per-protein embeddings can be found in the Utilities notebook.

Contrastive learning is a widely used approach to create sentence embeddings from word embeddings in natural language modeling. I applied the loss function proposed in the 2021 paper [**"SimCSE: Simple Contrastive Learning of Sentence Embeddings"**](https://arxiv.org/abs/2104.08821) by Tianyu Gao, Xingcheng Yao, and Danqi Chen, using a LoRA-injected ProtT5 model to generate contrastively enhanced per-protein Mean+SD embeddings. 

Similarly to the LA decoder, the final contrastive LoRA imbeddings showed the most promising results when used in combination with the standart mean embedding. 





### Results and Comparison to State-of-the-Art Fine-Tuning Method: LoRA

In my work, I compared my results against the state-of-the-art fine-tuning method, LoRA. The numeric values used to simulate the data for the boxplots can be found in Table 6 of the supplementary material from the 2024 paper titled [**"Fine-tuning Protein Language Models Boosts Predictions Across Diverse Tasks"**](https://www.nature.com/articles/s41467-024-51844-2) by Robert Schmirler, Michael Heinzinger, and Burkhard Rost.

It is important to note that the results for localization cannot be directly compared. This is because I used the DeepLoc 2.0 dataset, while the referenced paper employed the original DeepLoc dataset for training. The test set is the same setHard dataset for both.

Although my approach does not yet surpass the fine-tuned embeddings, I have demonstrated improvements over mean embeddings by incorporating additional information through LA embeddings or contrastive learning. The hyperparameters for the Multi-Layer Perceptron were optimized separately for each embedding dataset using the [Optuna package](https://optuna.readthedocs.io/en/stable/) and the boxplots were created using 10,000 bootstrapped values, with the value for the regression tasks (meltome and stability) being the spearman correlation and the value for the classification task (subcellular localization) being the accuracy. 

![Boxplots_summary](https://github.com/user-attachments/assets/e5328e33-7f36-4758-98f7-6db400546afb)


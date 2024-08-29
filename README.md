# GlobalProteinRepresentations

A quick summary of the work i did during my Fortgeschrittenen-Praktikum at the RostLab. The Model folder contains the code to train the light attention decoder, the contrastive LoRA model and the MLP used for the downstream prediction tasks utilizing the precomputed embeddings. The decoder was trained on a 50 percent redundancy reduced SwissProt, and the contrastive LoRA model was sadly not trained at all. <br />

The different datasets used to compare the embeddings on per protein prediction tasks were: <br />

**Meltome**: The mixed split which can be found at: [Meltome data](https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome) <br />

**Stability**: LMDB datasets provided in: [Stability data](https://github.com/songlab-cal/tape?tab=readme-ov-file#lmdb-data) <br /> 

**Localization**: The original dataset is the five split DeepLoc 2.0 data, found on their page: [DeepLoc 2.0 data](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) 
              Of that dataset I used the first four splits for training and the fifth for validation. As the test set, I use the setHard data
              proposed here: [setHard data](https://github.com/HannesStark/protein-localization/tree/master/data_files). The Localization data was also processed by me to be in the same format
              as the Meltome dataset (See utilities.ipynb). <br />



### Results and Comparison to State-of-the-Art Fine-Tuning Method: LoRA

In this study, I compared my results against the state-of-the-art fine-tuning method, LoRA. The simulation data used for the boxplots can be found in Table 6 of the supplementary material from the 2024 paper titled **"Fine-tuning Protein Language Models Boosts Predictions Across Diverse Tasks"** by Robert Schmirler, Michael Heinzinger, and Burkhard Rost.

It is important to note that the results for localization cannot be directly compared. This is because I used the DeepLoc 2.0 dataset, while the referenced paper employed the original DeepLoc dataset for training. The test set is the same setHard dataset for both.

Although my approach does not yet surpass the fine-tuned embeddings, I have demonstrated improvements over mean embeddings by incorporating additional information through LA embeddings. The hyperparameters for the Multi-Layer Perceptron (MLP) were optimized separately for both the mean embeddings and the concatenation of mean and LA embeddings using the [Optuna package](https://optuna.readthedocs.io/en/stable/).

![Boxplots Summary](https://github.com/user-attachments/assets/528ac9f1-d305-43f9-8436-dd744526a986)


# GlobalProteinRepresentations

A quick summary of the work i did during my Fortgeschrittenen-Praktikum at the RostLab. The Model folder contains the code to train the light attention decoder, the contrastive LoRA model and the MLP used for the downstream prediction tasks utilizing the precomputed embeddings. The decoder was trained on a 50 percent redundancy reduced SwissProt, and the contrastive LoRA model was sadly not trained at all. 

The different datasets used to compare the embeddings on per protein prediction tasks were:

Meltome: The mixed split which can be found at: https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome 

Stability: LMDB datasets provided in: https://github.com/songlab-cal/tape?tab=readme-ov-file#lmdb-data 

Localization: The original dataset is the five split DeepLoc 2.0 data, found on their page: https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ 
              Of that dataset I used the first four splits for training and the fifth for validation. As the test set, I use the setHard data
              proposed here: https://github.com/HannesStark/protein-localization/tree/master/data_files. The Localization data was also processed by me to be in the same format
              as the Meltome dataset.

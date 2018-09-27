# Recursive-Neural-Structural-Correspondence-Network

This is an instruction file for successfully running the code provided.

The code is an implementation of the following paper:

************************************************************************************************
"@InProceedings{P18-1202,
  author = 	"Wang, Wenya and Pan, Sinno Jialin",
  title = 	"Recursive Neural Structural Correspondence Network for Cross-domain Aspect and Opinion Co-Extraction",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2171--2181",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1202"
}"
***********************************************************************************************

Please follow the following steps:
1. Go to 'util' folder to produce intermediate files:
   - Download Stanford dependency tree parser
   - use '10depParse.py' to generate dependency trees
   - use '20dtreeLabel_cross_split.py' to build data structures and split data to training and testing
   - use '30word_embedding.py' to store pre-trained word embeddings

2. Go to the main folder to conduct training
   - run 'train_depnn_cross.py' to pre-train recursive neural network first
   - run 'train_joint_cross.py' to train the joint model 


Note: When the digital device dataset is used as the source domain, we remove the sentences without any aspect words to train the model.

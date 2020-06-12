This repository contains code for the paper:

Field, Anjalie and Yulia Tsvetkov, "Unsupervised Discovery of Implicit Gender Bias", https://arxiv.org/abs/2004.08361


The file src/run_from_scratch.sh contains run commands for the primary pipeline, to create data splits and train models from scratch. This pipeleine requires separately downloading the RtGender dataset, which is available here: https://nlp.stanford.edu/robvoigt/rtgender/

The file src/run_saved_models.sh contains code for loading the pre-trained models used in the paper and running them over test data. This script requires separately downloading the models and training data (for vocabulary construction), available here: https://drive.google.com/file/d/1FvVTk-FIW__oEl3Nz2ruifhpfIQ_JCsO/view?usp=sharing (Note that these files are large, 7.7G compresssed).

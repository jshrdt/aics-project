# aics-project
## Synthetic vs. genuine image detection

An exploration of ML for the detection of AI-generated images which imitate photographic realism based on Bird & Lotfi (2024) and the CIFAKE (Bird, 2023) dataset.

Required datasets, to be placed in code/:  
CIFAKE https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data  
CIFAR100: https://www.kaggle.com/datasets/fedesoriano/cifar100/data  
ai-generated-images-vs-real-images/test: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images (only the test dir!)

___

#### Quickstart:
* pretrained model files are stored in models/
* go to code/CIFAKE_classification.ipynb or code/func_difficulty.ipynb for training/testing on CIFAKE  
* then to transfer_cifar100.ipynb or transfer_mixed.ipynb to load & apply models on CIFAR100/AGIRI
  

Models can also be trained/tested from the cmd line, but this is not necessarily for the notebooks/analysis; they be stored in & loaded from models/

For the cmd line scripts train.py & test.py (described in below under code/), from aics-project/code, call e.g.:
> $ python3 train.py -ep1 -mf dummymodel  
> $ python3 test.py -mf dummymodel

___

a general description of the project which should include:  

(i) what information can be found in other folders  
(ii) instructions how to run your system, what other componenets/datasets are required and where they can be obtained  
(iii) general work plan of the projects and a place to leave our comments. This is also the first file we will look at.  

___

## Repos breakdown:

code/: all code should go in this folder. For coding we recommend using Jupyter notebooks. All steps required to run the code should be well documented, as well as individual parts of the code should be explained with comments. One should be able to run the code and replicate the process. 

### Notebooks

#### CIFAKE_classification.ipynb
* Training & testing of the base model and model modified with SRM channel attention

#### func_difficulty
* Exploration of CIFAKE task difficulty; examine training progress of first epoch after X batches

#### transfer_cifar100.ipynb
* Testing the transfer performance of base, mini, and attention model on cifar100 (unseen, real images)
* Viewing performance per super- and atomic class

#### transfer_mixed.ipynb
* Testing the transfer performance of base, mini, and attention model on dataset of ai/real imgs from various origins/styles (Zhang)

### Scripts

#### classes.py  
OOP classes for...  
* CI_loader: data batching, transformation, encoding; used for CIFAKE & CIFAR100
* CIFAKE_CNN: Model architecture based on cifar10_tutorial.ipynb, modified for binary classification
* SRMLayer: Style-recalibration module channel attention layer (https://blog.paperspace.com/srm-channel-attention/)

#### train.py
* Script to train models on CIFAKE
* cmd line args: batch size (bs; 32), epochs (ep; 5), modelfile (mf; only save model if passed; format: '../models/{args.modelfile}.pth'  
Example call:  
> $ python3 train.py -ep1 -mf dummymodel

#### test.py
* Script to test & evaluate model on CIFAKE, CIFAR100 test data, includes eval functions used in notebooks
* cmd line args: modelfile (mf; model to test from '../models/{args.modelfile}.pth', default: base_model), decision_threshold (thr; 0.5)  
Example call:  
> $ python3 test.py -thr 0.4

#### config.json
* simple json file to keep track of some filenames; I.e. CIFAKE dir, base/attention/mini model paths

____

data/: use this folder for datasets that you have created. Those that can be downloaded from elsewhere can be referenced in README.md and files that are created by your code should be in your folder on MLTGPU. If you need to upload large files to Github, you can use their Git Large File Storage.

* (CIFAKE should be placed in here)
* (CIFAR100 should be placed in here)

____

library/: a place for any supporting files that you have used, for example required system configurations and other supporting files that cannot be obtained from open repositories.

* (not used)
____

models/: place to store trained models for simple loading  

* shared params: batch size 32, learn rate 0.001, momentum 0.9

#### base_model.pth
* base CIFAKE_CNN, trained on CIFAKE train data, epochs 10

#### mini_model.pth
* base CIFAKE_CNN, trained on CIFAKE train data for 1 epoch to examine task difficulty

#### attn_model.pth
* CIFAKE_CNN with SRM channel attention layer before 2nd conv2d layer, trained on CIFAKE train data (same params as base)

____

notes/: To show the work that you have done (especially important if you get a negative result) you should keep a lab log as you work. This will also allow you to write the report faster. Use this folder with markdown text. files as a wiki to document your experiment and take notes as you go along. These files can also be edited and viewed on Github.

#### lognotes.md
* work progress diary (messy, personal use)

____

paper/: a place for your course report and presentation

#### LT2318_AICS_project_pitch.pdf
* original project pitch; proposed project steps
* updated on canvas discussion thread

#### aics_report_v01.pdf
* project paper

#### aics_ppp.pptx
* presentation slides (draft)



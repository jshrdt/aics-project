# aics-project
## Synthetic vs. genuine image detection

An exploration of ML for the detection of AI-generated images which imitate photographic realism based on Bird & Lotfi (2024) and the CIFAKE (Bird, 2023) dataset.

Required datasets:
CIFAKE https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data

___

#### Quickstart:
* pretrained model files are stored in models/
* go to code/aics0.ipynb or code/aics_mini.ipynb to view performances

* train & test own models with cmd line scripts train.py & test.py (described below)  
Ex., from aics-project/code, call:
> $ python3 train.py -ep1 -mf dummymodel
> $ python3 test.py -mf dummymodel
* to quickly train & test your own model which will be stored in & loaded from models/

___

a general description of the project which should include:  

(i) what information can be found in other folders  
(ii) instructions how to run your system, what other componenets/datasets are required and where they can be obtained  
(iii) general work plan of the projects and a place to leave our comments. This is also the first file we will look at.  

___

## Repos breakdown:

code/: all code should go in this folder. For coding we recommend using Jupyter notebooks. All steps required to run the code should be well documented, as well as individual parts of the code should be explained with comments. One should be able to run the code and replicate the process. 

### Notebooks

#### aics0.ipynb
* ...

#### aics_mini.ipynb
* Exploration of CIFAKE task difficulty; examine training progress of first epoch after X batches

### Scripts

#### classes.py  
OOP classes for...  
* CIFAKE_loader: data batching, transformation, encoding
* CIFAKE_CNN: Model architecture based on cifar10_tutorial.ipynb, modified for binary classification
* SRMLayer: Style-recalibration module channel attention layer (https://blog.paperspace.com/srm-channel-attention/)

#### train.py
* Script to train models on CIFAKE
* cmd line args: batch size (bs; 32), epochs (ep; 5), modelfile (mf; only save model if passed; format: '../models/{args.modelfile}.pth'  
Example call:  
> $ python3 train.py -ep1 -mf dummymodel

#### test.py
* Script to test & evaluate model on CIFAKE test data, includes eval functions used in notebooks
* cmd line args: modelfile (mf; model to test from '../models/{args.modelfile}.pth', default: base_model), decision_threshold (thr; 0.5)  
Example call:  
> $ python3 test.py -thr 0.4

#### config.json
* simple json file to keep track of some filenames; I.e. CIFAKE dir, base/attention/mini model paths

____

data/: use this folder for datasets that you have created. Those that can be downloaded from elsewhere can be referenced in README.md and files that are created by your code should be in your folder on MLTGPU. If you need to upload large files to Github, you can use their Git Large File Storage.

* (CIFAKE should be placed in here)

____

library/: a place for any supporting files that you have used, for example required system configurations and other supporting files that cannot be obtained from open repositories.

* (not used)
____

models/: place to store trained models for simple loading  

#### base_model.pth
* base CIFAKE_CNN, trained on CIFAKE train data
* epochs 10, batch size 32, learn rate 0.001, momentum 0.9
* Performance (thr 0.5): Accuracy: 91.39%, Precison: 88.43%, Recall: 95.23%, F1-Score: 91.70%
* Performance peaks per threshold: Accuracy 92.26% (0.7), Precision 96.76% (0.9), Recall 99.15% (0.1), F1-Score 92.22% (0.6)

#### mini_model.pth
* base CIFAKE_CNN, trained on 95% of CIFAKE train data to examine task difficulty
* epoch 1, batch size 32, learn rate 0.001, momentum 0.9
* Performance (thr 0.5): Accuracy: 77.46%, Precison: 74.42%, Recall: 83.69%, F1-Score: 78.78%

#### attn_model.pth
* CIFAKE_CNN with SRM channel attention layer, trained on CIFAE train data
* ...
* ...

____

notes/: To show the work that you have done (especially important if you get a negative result) you should keep a lab log as you work. This will also allow you to write the report faster. Use this folder with markdown text. files as a wiki to document your experiment and take notes as you go along. These files can also be edited and viewed on Github.

#### lognotes.md
* work progress diary

____

paper/: a place for your course report and presentation

#### LT2318_AICS_project_pitch.pdf
* original project pitch

#### aics_report_draft.pages
* project paper (draft)

#### aics_ppp.pptx
* presentation slides (draft)



# Work notes

### Dec11:  
GitHub setup; report template/basic notes, dataset references. 

- get/load dataset (Class CIFAKE; sep train/test data, shuffle), batcher  
- start setting up code from tutorial  

TBD: input transform; label encode? - done

### Dec12:  
- iterator for batches, filename to normalised tensor transform, label 0/1 encoding. 
- load, transform, train, test, evaluate functions; first complete run + scores
- split up code into main notebook, script for classes, script for functions

Img tensor range   
? (-1.9132, (2.3761) from https://pytorch.org/vision/0.8/transforms.html (transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)). 
? alt: course tutorial normalise to (-1, 1). 
labels: 0 - FAKE, 1 - REAL (in accordance with CIFAKE paper)

- add seeded shuffling: once all data in test/train; then once inside batch during iter/transform   
- unroll CNN structure from cifar10_tutorial.ipynb to visualise/understand structure 
- modify CNN structure from class for binary classification (add sigmoid, change to BCELoss)  
- adapt testing for binary task (round to 0/1 with simple 0.5 thresh)  
- typed up some documentation, pulled classes/train/test functions apart

#### V01.01  
- first run with these minimal changes in place    
- normalise to 0.5 across the board  
- batch size 18, 2 epochs, BCELoss, added Sigmoid/last linear layer to dim 1)    
- avg loss starts at 0.687 (25%) -> ep0:  0.544; end of ep1: 0.341 (slight tapering in ep1 visible already)

Performance  
Accuracy: 90.24%  
Precision: 92.14%  
Recall: 87.98%  
F1-Score: 90.01%  


#### CIFAKE Paper
- CNN+linears, Relu; eval with acc/prec/recall/f1  
- kernels?; stride=1  
- pooling  
- sigmoid  
- score (tbd across modalities), but overall (p.6) 'all feature extractors scored relatively well without the need for dense layers to process feature maps' avg. classificaiton accuracy of 91.79%  

- XAI: Gradient Class Activation Mapping (Grad-CAM)  

TBD: 
- run more epochs on GPU (GitHub sync); compare architecture with paper (p6ff)  
- look at params  
- torch seed!
- add threshold keyword for test evaluation 

Performance seeded
Performance  
Accuracy: 90.24%  
Precision: 92.14%  
Recall: 87.98%  
F1-Score: 90.01%  

Varying threshold for 0/1 decision (previously 0.5), 'fairly good' performance is easy to achieve, so task does not seem to be particularly difficult?

Performance with thresh 0.1  
Accuracy: 74.09%  
Precison: 66.24%  
Recall: 98.22%  
F1-Score: 79.12%  
Could make a plot from this! - done!

#### Next:
- architecture experiments?
- data bias?
- attention? Cf repos Maria 

#### Dec17  
Added SRM (Style reduction module) layer after first conv/pool pair (channels 6)   
Slower to train (in terms of loss progression), but 5 epochs already reaching 89%; peak closer to thresh 0.6

- 30 epochs (loss still decreasing): epoch: 29	total loss: 429.3308508070186	avg loss: 0.13738587225824594; thresh:0.7, acc	92.364236, prec	92.683664, rec	91.988398, f1	92.334722; img: SRM01e30DEC18

[paper writing, formulating plan]

#### Dec20
- looking for other possible real vs fake img datasets

#### Jan7
- start on paper
- considerations which of proposed directions to explore (cifar100, other fake images, attn heat maps, zero/few shot; dataset bias)

#### Jan8
- rewrite loader to transform data + send to device only once on first epoch
- plan: mini, normal, v attention on CIFAKE
- per-batch learning for mini: difficulty of task in CIFAKE?
- normal: og attempt
- attention: improvement on paper/extension

- added mini_model: train only 1 epoch, logging losses/accuracy every X batches to test task difficulty
- closer look at the other ai vs real image datasets collected
- code organisation

then:
- apply to other fake data
- apply to CIFAR100 unseen categories

#### Jan9
- documentation; cmd line compatibility of scripts





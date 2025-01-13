
# LT2318: AICS project outline
## Real or really fake? – Synthetic vs. genuine image classification

Thread: https://canvas.gu.se/courses/80310/discussion_topics/634296

Rough idea: (Re)-implement a model for image classification on real vs. synthetic image data, cf. Bird
& Lotfi (2024, https://ieeexplore.ieee.org/abstract/document/10409290)

#### Main task
* Test the capabilities of CNNs to distinguish between 'real' and 'fake' image data
* Extend Bird & Lotfi (2024) with channel attention
* Apply models to real, similar images (CIFAR-100) & real-fake mixed, dissimilar ones (AGIRI, https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data)

* Apply to CIFAR-100: False negative risk? Explore possible content-prediction interaction

### Research angles:

A) CIFAKE
* Performance of base vs. attention model: denoising successful?  (/code/CIFAKE_classification.ipynb)  
* Train mini version of base model (1 epoch): learning curve? Performance? Task difficulty of CIFAKE? (/code/func_difficulty.ipynb)

B) CIFAR-100
* /code/transfer_cifar100.ipynb
* Was real v. Fake distinction learned? Chance of false negatives on similar data
* Possible texture/content influences? – compare performance per content class labels in CIFAR100

C) AGIRI
* /code/transfer_mixed.ipynb
* difficult transfer task, but most realistic (varied data in terms of content, styles, sources)
* CIFAKE suitable as training data for applied was? (Suggested by Bird & Lotfi 2024)
* comparison with mini model; good performance on CIFAKE -> good performance AGIRI?


#### Resources
Required datasets, to be placed in code/:
CIFAKE https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data  
CIFAR100: https://www.kaggle.com/datasets/fedesoriano/cifar100/data  
ai-generated-images-vs-real-images/test: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images   

Base code for CNN: https://canvas.gu.se/files/9275057/  (supervised/cifar10_tutorial.ipynb)


# ProtoNets-TensorFlow
Implement `ProtoNets` from [Snell et al., prototypical networks for few-shot learning](https://arxiv.org/abs/1703.05175) in TensorFlow 2, and perform experiments on the [COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).

The embedding architecture of `ProtoNets` is composed of 4 convolutional blocks. Each block consists of a 64-filter 3 by 3 convolution, batch normalization, ReLU and a 2 by 2 max pooling layer. Moreover, `ProtoNets` uses Euclidean distance.

See [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/ProtoNets/ProtoNets.py) for the implementation of `ProtoNets` and [here](https://github.com/DrMMZ/ProtoNets-TensorFlow/blob/master/Experiments/COVIDx.ipynb) for the notebook demonstration on the COVIDx dataset experiments.


### Requirements
`numpy 1.18.5`, `tensorflow-gpu 2.1.0`, `matplotlib 3.2.2`


### Experiments
The COVIDx dataset consists of chest X-ray images for COVID-19. There are 3 classes in 13918/1579 train/test images: 489/100 COVID-19, 7966/885 normal and 5463/594 pneumonia. We further split the train set to train/val 400/89 images per class. All images are resized to 84 by 84 by 3 and rescaled. Moreover each training image is augmented with 90, 180 and 270 degree rotations, horizontal flip and central crop. Augmented images are added to different classes. It follows that the train set becomes comprising of 18 classes. 

The embedded images are in 1600-dimension. All models were trained for 120 epochs on a GPU, each epoch contains N steps where N is defined by the size of the train set divided by the number of classes ("way") multiplied by the sum of the number of support ("shot") and query examples. We used Adam optimizer with the initial learning rate 0.001, halved at 25, 50 and 75 and 100 epochs.

We trained `ProtoNets` on {3,5,10,15}-way, {1,5,10}-shot and 15-query, and validated on 3-way, {5,10,20,50}-shot and 15-query. The final model `{10-way, 5-shot, 15-query}-ProtoNet` is the one trained with 10-way, 5-shot and 15-query. 

#### Results
We trained the `{10-way, 5-shot, 15-query}-ProtoNet` 4 times with randomly sampled images. The results were computed on the test set on 3-way, 50-shot and 15-query, and provided by positive predictive value (PPV) and true positive rate (TPR) averaged over 901 steps.

|1.|Normal|Pneumonia|COVID-19|
|---|---|---|---|
|TPR (%)|87.1|80.9|85.5|
|PPV (%)|83.8|81.6|90.1|

|2.|Normal|Pneumonia|COVID-19|
|---|---|---|---|
|TPR (%)|84.5|78.0|83.8|
|PPV (%)|84.0|76.7|87.9|

|3.|Normal|Pneumonia|COVID-19|
|---|---|---|---|
|TPR (%)|85.5|80.5|76.3|
|PPV (%)|82.6|76.5|86.3|

|4.|Normal|Pneumonia|COVID-19|
|---|---|---|---|
|TPR (%)|86.9|79.8|84.5|
|PPV (%)|84.9|79.9|88.5|

The results do not represent the state-of-art on this dataset. See [Wang et al., COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images](https://arxiv.org/abs/2003.09871). 

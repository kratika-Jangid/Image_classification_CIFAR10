﻿# Image_classification_CIFAR10
INTRODUCTION :
Deep learning is a sub-field of machine learning — which, in turn, is a sub-field of AI — that deals 
with training models and artificial neural networks (ANNs) capable of replicating the working of a 
human brain. 
Deep learning is, right now, an ambitious field of research that has shown promising applications for 
transforming the world. 
Examples of deep learning include Google’s DeepDream and self-driving cars. As such, it is 
becoming a lucrative field to learn and earn in the 21st century.
 DEEP DREAM SELF DRIVING CARS
Image classification is a popular application of deep learning that is highly popular among deep 
learning engineers. Deep learning is revolutionising how images are being identified and classified 
automatically.
 
This project aims to create an image classification program using the popular CIFAR-10 
dataset. The dataset contains 60,000 images that belong to 10 different classes. 
Moreover, the dataset also has 50,000 training images along with 10,000 test images. Although the 
dataset is effectively solved, it can be used as the basis for learning and practising how to develop, 
evaluate, and use convolutional deep learning neural networks for image classification.
CIFAR is an acronym that stands for the Canadian Institute For Advanced Research and the CIFAR10 dataset was developed along with the CIFAR-100 dataset by researchers at the CIFAR institute.
The dataset comprises 60,000 32×32 pixel colour photographs of objects from 10 classes, such as 
frogs, birds, cats, ships, etc. The class labels and their standard associated integer values are listed 
below.
● 0: aeroplane
● 1: automobile
● 2: bird
● 3: cat
● 4: deer
● 5: dog
● 6: frog
● 7: horse
● 8: ship
● 9: truck
These are very small images, much smaller than a typical photograph, and the dataset was intended 
for computer vision research.
[https://machinelearningmastery.com/what-is-deep-learning/]


LITERATURE SURVEY :
Visual object recognition is of great importance for humans to interact with each other and the natural 
world. We possess a huge ability of visual recognition as we can almost effortlessly recognize objects 
encountered in our life such as animals, faces, and food. Especially, humans can easily recognize an 
object even though it may vary in position, scale, pose, and illumination. Such ability is called core 
object recognition, and is carried out through the ventral stream in the human visual system [1].
Visual object recognition has long been considered as a privilege of biological systems. In the field of 
computer vision, many studies have tried to build systems able to imitate humans’ object recognition 
ability. In spite of several decades of effort, machine visual recognition was far from human 
performance. Yet, since the past few years, machine performance has been dramatically improved 
thanks to the reemergence of convolutional neural networks (CNN) and deep learning [2, 3, 4, 5, 6], 
and thus even surpasses human performance [7, 8]. Like traditional neural networks, which are 
inspired by biological neural systems, the architecture of CNNs for object recognition is feedforward 
and consists of several layers in a hierarchical manner. Particularly, some works reveal hierarchical 
correspondence between CNN layers and those in the human object recognition system [9].
Recently, many deep neural networks with various architectures have been proposed for computer 
vision, especially for visual recognition [5, 8, 19, 20, 21, 22]; although the first and basic 
convolutional neural network (LeNet) was presented several years before to recognize handwritten 
characters [23]. In this section, we will focus only on recent CNNs that dramatically improve object 
recognition accuracy. Generally, a CNN consists of several layers stacked upon each other; they are 
convolution (conv), Rectified Linear Unit (ReLU), Batch Normalisation (BN) [24], pooling, dropout 
[25], and Fully-Connected layer (FC). Some basic layers can be grouped together to create a block. 
FC layers are often used towards the end of a network. A loss function between target (ground truth) 
and network’s output is minimised to find the parameters or weights of the neural network. Gradient 
Descent with back propagation is almost exclusively used for optimization. For efficient convergence, 
some initialization and weight updating techniques have been proposed [7, 26].
Deep neural networks for visual object recognition
Sno. ARCHITECTURE 
NAME
STRUCTURE ACCURACY
1. Alex Net ● It has 8 layers with learnable 
parameters.
● The input to the Model is RGB images.
● It has 5 convolution layers with a 
combination of max-pooling layers.
● Then it has 3 fully connected layers.
● The activation function used in all layers 
is Relu.
● It used two Dropout layers.
● The activation function used in the 
AlexNet in 
PyTorch 
CIFAR10 
Class(74.74% 
Test Accuracy)
output layer is Softmax.
● The total number of parameters in this 
architecture is 62.3 million.
https://www.analyticsvidhya.com/blog/2021/03/introduction-to-thearchitecture-ofalexnet/#:~:text=The%20Alexnet%20has%20eight%20layers,layers%20exc
ept%20the%20output%20layer.
2. Google Net The architectural details of auxiliary classifiers as 
follows:
● An average pooling layer of filter size 5×5 and 
stride 3.
● A 1×1 convolution with 128 filters for dimension 
reduction and ReLU activation.
● A fully connected layer with 1025 outputs and 
ReLU activation
● Dropout Regularisation with dropout ratio = 0.7
● A softmax classifier with 1000 classes output 
similar to the main softmax classifier.
https://towardsdatascience.com/deep-learning-googlenet-explainedde8861c82765
GoogleNet in 
PyTorch 
CIFAR10 
Class(92.57% 
Test Accuracy)
3. VGG 13 Architecture walkthrough:
● The first two layers are convolutional layers with 
3*3 filters, and first two layers use 64 filters that 
results in 224*224*64 volumes with the same 
convolutions are used. The filters are always 3*3 
with stride of 1
● After this, pooling layer was used with max-pool 
of 2*2 size and stride 2 which reduces height and 
width of a volume from 224*224*64 to 
112*112*64.
● This is followed by 2 more convolution layers 
with 128 filters. This results in the new dimension 
of 112*112*128.
● After pooling layer is used, volume is reduced to 
56*56*128.
● Two more convolution layers are added with 256 
filters each followed by down sampling layer that 
reduces the size to 28*28*256.
● Two more stacks each with 3 convolution layers 
are separated by a max-pool layer.
VGG in 
PyTorch 
CIFAR10 
Class(90.17% 
Test Accuracy)
● After the final pooling layer, 7*7*512 volume is 
flattened into Fully Connected (FC) layer with 
4096 channels and softmax output of 1000 
classes.
https://medium.com/analytics-vidhya/vggnet-architecture-explainede5c7318aa5b6
1. VGG
VGG is one of the most well-known CNNs by introducing a new architecture and boosting 
recognition performance on the ImageNet dataset [27]. The particularity of VGG is that all 
convolutional kernels are of size 3×3. All conv layers are followed by ReLU, while max-pooling is 
applied after some of them. VGG utilises three FC layers at the end, meanwhile dropout is absent 
from the network. When feature size is halved, width (i.e. number of feature maps or number of 
filters) is doubled to keep an equivalent complexity per layer. Another interpretation of such structure 
is as follows. After pooling to decrease feature size, the receptive field size increases and hence there 
are more variations in this image region, which in turn require more filters to represent. VGG is the 
first to show that a deep network (19 layers or more) is possible.
2. Inception/GoogLeNet 
GoogLeNet (22 layers with parameters) is the implementation of the Inception architecture for 
ImageNet; it contains several Inception modules [28]. Each Inception module has four branches: three 
conv branches with kernel size of 1×1, 3×3, and 5×5, and one branch for max-pooling. These multibranches represent multi-scale processing. Conv layers are always used with ReLU; dropout is near 
the network’s output. Besides, 1×1 conv layer is also utilised to reduce dimension (before 3×3 or 5×5 
conv layers) or to match dimension (before addition). In fact, the 1×1 conv layer was first proposed in 
the Network-in-Network architecture [29]. When combined with a traditional conv layer, it can 
enhance representational power while keeping the same receptive field size. Figure 1: Example of 
ResNet’s architecture [21]. The dashed shortcuts represent subsampling. The network has 18 
weighted layers. As in the VGG architecture, towards the network’s output, feature size decreases and 
width increases. 
3) Alexnet
AlexNet was primarily designed by Alex Krizhevsky. It is a Convolutional Neural Network or CNN. 
After competing in ImageNet Large Scale Visual Recognition Challenge, AlexNet shot to fame. It 
achieved a top-5 error of 15.3%. This was 10.8% lower than that of the runner up. 
The primary result of the original paper was that the depth of the model was absolutely required for its 
high performance. This was quite expensive computationally but was made feasible due to GPUs or 
Graphical Processing Units, during training. AlexNet architecture consists of 5 convolutional layers, 
3 max-pooling layers, 2 normalisation layers, 2 fully connected layers, and 1 softmax layer. Each 
convolutional layer consists of convolutional filters and a nonlinear activation function ReLU. The 
pooling layers are used to perform max pooling. Input size is fixed due to the presence of fully 
connected layers. The input size is mentioned at most of the places as 224x224x3 but due to some 
padding which happens to be 227x227x3 .AlexNet overall has 60 million parameters.


DATASET
CIFAR10 is a dataset of natural colour images, which is widely used to evaluate recognition 
capability of deep neural networks [8, 19, 20, 21, 30]. This dataset contains 60000 small images of 
32×32 pixels from ten categories (or classes): aeroplane, automobile, bird, cat, deer, dog, frog, horse, 
ship, and truck. It is divided into two sets: a training set of 50000 images and a test set of 10000 
images (1000 images per category). The images are taken under varying conditions in position, size, 
pose, and illumination. Many objects are partially occluded. The following figure shows the first 9 
images of the dataset.
REQUIREMENTS :
 LIBRARIES USED :
1. Pytorch
2. Torchvision
3. Matplotlib
4. Numpy
5. Joblib
METHODOLOGY : 
STEP 1 : LOADING THE DATASET
The CIFAR-10 dataset is loaded having classes: plane,car,bird,cat,deer,'dog', 'frog', 'horse', 'ship', 
'truck'.The images are normalised to 0.5 along all the axes. A training and testing set is created using 
torchvision.dataset.CIFAR10().
The data is loaded to the train loader and test loader with batch size of 4. Using 
torch.utils.data.DataLoader().
STEP 2 : DEFINING A CONVOLUTIONAL NEURAL NETWORK
CNN is a feed-forward network. During the training process, the network will process the input 
through all the layers, compute the loss to understand how far the predicted label of the image is 
falling from the correct one, and propagate the gradients back into the network to update the weights 
of the layers. By iterating over a huge dataset of inputs, the network will “learn” to set its weights to 
achieve the best results. A forward function computes the value of the loss function, and the backward 
function computes the gradients of the learnable parameters.
To build a neural network with pytorch, we’ll use the torch.nn package. This package contains 
modules, extensible classes and all the required components to build a neural network. A CNN is a 
class of neural networks, defined as multilayered neural networks designed to detect complex features 
in data.
In the code 3 CNN layers have been defined, in which all layers follow a hierarchical sequence of cnn, 
leakyRelu and maxpooling i.e.
Conv2d is used with a kernel size of 5 and maxpool with size 2.
We'll use nn.Sequential to chain the layers and activation functions into a single network 
architecture.
The following layers are involved in our network:
● Convolutional Layer is the first layer that is used to extract the various features from the input 
images. In this layer, the mathematical operation of convolution is performed between the 
input image and a filter of a particular size MxM. By sliding the filter over the input image, 
the dot product is taken between the filter and the parts of the input image with respect to the 
size of the filter (MxM).
The output is termed as the Feature map which gives us information about the image such as 
the corners and edges. Later, this feature map is fed to other layers to learn several other 
features of the input image.
https://www.upgrad.com/blog/basic-cnn-architecture/
● The ReLU layer is an activation function to define all incoming features to be 0 or greater. 
When you apply this layer, any number less than 0 is changed to zero, while others are kept 
the same.
● The MaxPool layer will help us to ensure that the location of an object in an image will not 
affect the ability of the neural network to detect its specific features.
Dropout: Dropout is a machine learning technique where you remove (or "drop out") units in a neural 
net to simulate training large numbers of architectures simultaneously. Importantly, dropout can 
drastically reduce the chance of overfitting during training. 
Impact of Using Dropout in PyTorch
● An unregularized network quickly overfits on the training dataset. The validation loss for 
without-dropout run diverges a lot after just a few epochs. This accounts for the higher 
generalisation error.
● Training with a dropout layer with a dropout probability of 25% prevents model from 
overfitting. However, this brings down the training accuracy, which means a regularised 
network has to be trained longer.
● Dropout improves the model generalisation. Even though the training accuracy is lower than 
the unregularized network, the overall validation accuracy has improved. This accounts for a 
lower generalisation error. 
https://wandb.ai/authors/ayusht/reports/Implementing-Dropout-in-PyTorch-With-Example--
VmlldzoxNTgwOTE
To achieve regularisation, so that you can Effectively prevent the problem of overfitting we use 
dropout over the layers.
The forward method is called when we use the neural network to make a prediction. Another term for 
"making a prediction" is running the forward pass, because information flows forward from the input 
through the hidden layers to the output. When we compute parameter updates, we run the backward 
pass by calling the function loss.backward(). During the backward pass, information about parameter 
changes flows backwards, from the output through the hidden layers to the input.
The forward method is called from the __call__ function of nn.Module, so that when we run 
model(input), the forward method is called.
https://www.cs.toronto.edu/~lczhang/360/lec/w03/nn.html

STEP 3 : DEFINING A LOSS FUNCTION
A loss function computes a value that estimates how far away the output is from the target. The main 
objective is to reduce the loss function's value by changing the weight vector values through 
backpropagation in neural networks.
Loss value is different from model accuracy. Loss function gives us the understanding of how well a 
model behaves after each iteration of optimization on the training set. The accuracy of the model is 
calculated on the test data and shows the percentage of the right prediction.
In PyTorch, the neural network package contains various loss functions that form the building blocks 
of deep neural networks. Here we will use a Classification loss function based on Cross-Entropy loss 
and an Adam Optimizer. Learning rate (lr) sets the control of how much you are adjusting the weights 
of our network with respect to the loss gradient. You will set it as 0.001. The lower it is, the slower 
the training will be.
In the above snippet lr represents the learning rate which is set at 0.001 and momentum at 0.9

STEP 4 : TRAINING THE DATA
The data is trained with epoch 4 having a batch size of 12500. Forward pass followed by backward 
propagation is done in each cycle . Later we perform Adam optimization,which is an optimization 
method , which can greatly improve the performance of the experiment.
We use an equal learning rate for all layers, which we adjust manually throughout training.
Now total accuracy is specified as the accuracy rate, that is, the predicted probability of the model in 
the test set.

STEP 5: TESTING THE DATA
As defined, the loss value will be printed every 1,000 batches of images for every 
iteration over the training set. Loss value is expected to decrease with every loop.
The accuracy of the model is also displayed after each iteration. Model accuracy is 
different from the loss value. Loss function gives us the understanding of how well a 
model behaves after each iteration of optimization on the training set. The accuracy 
of the model is calculated on the test data and shows the percentage of the right 
prediction. In our case it will tell us how many images from the 10,000-image test set 
our model was able to classify correctly after each training iteration.
RESULT ANALYSIS :
The following output is generated. The model stands at an accuracy of 70.62%
FUTURE SCOPE AND CONCLUSION:
Image processing applications can make it possible for machines to act as more self-sufficient and 
ensure the quality of products. Assuming processing systems work faster than humans, inline quality 
controls like 100% controls can be very quickly implemented. Damaged parts can be replaced or 
corrected, which would lead to more efficacies of production facilities. 
Key concerns in agriculture include quality of yield and water stress. Irrigation monitoring and 
providing information can be made possible by tracking satellite imaging of the fields. Processing of 
infrared images can act as an additional means to monitor and analyse irrigation. This analysis can 
then be utilised in pre-harvesting operations for deciding whether to harvest or not. Growth of weeds 
can also be detected by using a combination of machine learning and image processing algorithms and 
techniques. 
3D imaging is a process where a 2D image is converted into a 3D image by creating the optical 
illusion of depth. The next step is rendering where colours and textures are included in the 3D model 
to make it look realistic. With such 3D imaging and rendering, doctors can see extremely high quality 
3D images of organs that they couldn’t have seen otherwise. This, in turn, can help them carry out 
delicate surgeries and make accurate diagnoses.
In this paper we have used CNN for image classification using the CIFAR-10 dataset.In the following 
code snippet 3 CNN layers have been defined, in which all layers follow a hierarchical sequence of 
cnn, leakyRelu and maxpooling.CNNs can be useful for extracting features from images. They helped 
us to improve the accuracy of our previous neural network model from 65% to 71% – a significant 
upgrade.
You can play around with the hyperparameters of the CNN model and try to improve accuracy even 
further. Some of the hyperparameters to tune can be the number of convolutional layers, number of 
filters in each convolutional layer, number of epochs, number of dense layers, number of hidden units 
in each dense layer, etc.
REFERENCES :
[1] J. J. DiCarlo, D. Zoccolan, and N. C. Rust, "How does the brain solve visual object recognition?," 
Neuron, vol. 73, no. 3, pp. 415-434, 2012.
[2] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, pp. 436–444, 2015. 
[3] J. Schmidhuber, "Deep learning in neural networks: An overview," Neural Networks, vol. 61, pp. 
85-117, 2015. 
[4] Y. Bengio, A. Courville, and P. Vincent, "Representation Learning: A Review and New 
Perspectives," PAMI, vol. 35, no. 8, pp. 1798-1828, 2013. 
[5] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional 
neural networks," in NIPS, 2012, pp. 1097-1105. 
[6] Y. Bengio, "Learning Deep Architectures for AI," Foundations and Trends in Machine Learning, 
vol. 2, no. 1, pp. 1-127 , 2009. 
[7] K. He, X. Zhang, S. Ren, and J. Sun, "Delving Deep into Rectifiers: Surpassing Human-Level 
Performance on ImageNet Classification," in ICCV, 2015. 
[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in CVPR, 
2016. 
[9] R. M. Cichy, A. Khosla, D. Pantazis, A. Torralba, and A. Oliva, "Comparison of deep neural 
networks to spatio-temporal cortical dynamics of human visual object recognition reveals hierarchical 
correspondence," Scientific Reports, vol. 6, no. 27755, 2016.
[20] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, "Aggregated Residual Transformations for Deep 
Neural Networks," in CVPR, 2017. 
[21] T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks with 
Cutout," arXiv, 2017. 
[22] X. Gastaldi, "Shake-Shake regularization," arXiv, 2017. 
[23] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradientbased learning applied to document 
recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998. 
[24] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by 
Reducing Internal Covariate Shift," in ICML, 2015. 
[25] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: a simple 
way to prevent neural networks from overfitting," The Journal of Machine Learning Research, vol. 
15, no. 1, pp. 1929-1958, 2014. 
[26] D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in International 
Conference on Learning Representations (ICLR), 2015.
[27] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image 
Recognition," arXiv, 2014. 
[28] C. Szegedy et al., "Going deeper with convolutions," in CVPR, 2015. 
[29] M. Lin, Q. Chen, and S. Yan, "Network in network," CoRR, 2013. 
[30] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical report, 
2009. 
[31] A. Vedaldi and K. Lenc, "MatConvNet: Convolutional Neural Networks for MATLAB," in 
ACM international conference on Multimedia, 2015, pp. 689-692. [32] A. Nguyen, J. Yosinski, and J. 
Clune, "Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognisable 
Images," in Computer Vision and Pattern Recognition, 2015.
[33]https://magnimindacademy.com/blog/image-processing-and-its-future-implications/
Use PyTorch to train your image classification model | Microsoft Docs
https://github.com/IliasEletas/ImageClassificationWithCifar1

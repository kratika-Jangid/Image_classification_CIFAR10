# Image_classification_CIFAR10
INTRODUCTION :
Artificial neural networks (ANNs) and training models are the focus of the
discipline of deep learning (DL), a branch of machine learning, which is a branch
of artificial intelligence.
The globe is currently being revolutionised by the ambitious and astonishing
field of study known as "deep learning".
Emerging technologies that use deep learning technology include Google's
DeepDream and self-driving cars. As a result, it is developing into a lucrative
subject to learn and work in the twenty-first century.
DEEP DREAM SELF DRIVING CARS
Image categorization is a famous deep learning application that is quite popular
among deep learning scientists. Deep learning makes it possible for photos to
be automatically recognised and categorised.
The objective of this research is to develop an image classification algorithm
utilising the well-known CIFAR-10 dataset. 60,000 photos from ten different
classes make up the dataset.
The collection also includes 50,000 training images and 10,000 test images. The
dataset can still be used as a learning and training tool for convolutional deep
learning neural networks for picture categorization even after it has been
effectively solved.
The CIFAR-10 dataset and the CIFAR-100 collection were developed by
researchers at the Canadian Institute for Advanced Research, or CIFAR—the
term for Canadian Institute for Advanced Research.
The collection includes 60,000 3232 pixel colour images of objects divided into 10
categories, such as frogs, birds, cats, ships, etc. The class names and the
corresponding ordinal integer values are listed below.
● 0:aeroplane
● 1:automobile
● 2:bird
● 3:cat
● 4:deer
● 5:dog
● 6:frog
● 7:horse
● 8:ship
● 9:truck
The dataset, which consists of very tiny images, was created for computer vision
research.


LITERATURE SURVEY :
Recognition of objects visually is essential for humans to interact with one other
and their surrounding environment. Because we recognise faces, animals, and
food almost instantly in daily life, we are able to recognise objects visually. The
ventral stream of the human visual system carries out the ability of core object
recognition. [1].
It has long been assumed that biological systems are able to recognise visual
objects. Many computer vision experiments have made an effort to imitate
human item identification. The accuracy of machine visual identification
remained much lower than that of a person despite years of study. However,
deep learning and convolutional neural networks (CNN) have substantially
improved machine performance to the point that it now even surpasses that of
humans. [7, 8]. Deep learning-based object recognition enables the quick and
precise prediction of an object's location within an image. The object detector
uses deep learning, a potent machine learning technique, to automatically
acquire the visual characteristics necessary for detection tasks. Similar to
standard neural networks, which are modelled after the neural networks found
in biological systems, convolutional neural networks for object recognition have
a feedforward design and consist of many layers stacked on top of one another.
Numerous studies show how Convolutional Neural Networks and those
deployed in the human object recognition system are analogous.[9].
Recently, several designs for computer vision, particularly for visual identification
based on deep neural networks, have been presented. [5, 8, 19, 20, 21, 22];
although the first convolutional neural network (LeNet) was formulated to
recognize characters written by hand [23].Some of the layers that generally make
up a CNN are convolution (conv), rectified linear unit (ReLU), batch normalisation
(BN), pooling, dropout, and fully connected layer (FC). A block can be created by
combining a few fundamental levels. The FC layers are frequently employed at
the network's termination. A cost function between the target (actual true value)
and network output is reduced in order to find the parameters or weights of the
neural network. Optimization almost exclusively employs gradient descent with
back propagation. For efficient convergence, a few commencement and weight
update techniques have been devised.[7, 26].
Deep neural networks for visual object recognition
Sno. ARCHITECTURE
NAME


STRUCTURE ACCURACY
1. Alex Net ● There are eight learnable layers in it.
● RGB photos are used as the model's
input.
● It has a combination of max-pooling
layers on each of its five convolutional
layers.
● It then has three completely linked
layers.
● Relu is the activation function utilised
in all levels.
● Two Dropout layers were utilised.
● Softmax is the activation function
utilised in the output layer.
● There are 62.3 million parameters in
this architecture as a whole.[34]
AlexNet in
PyTorch
CIFAR10
Class(74.74%
Test Accuracy)
2. Google Net ● The auxiliary classifiers' architectural
features are as follows:
● a typical 5x5 filter with a three-stride
pooling layer.
● a 1-1 convolution with 128 filters for
dimension reduction and Rectified
linear activation.
● a layer with 1024+1 outputs and
Rectified linear activation that is fully
connected.
● regularisation of dropouts at a
dropout ratio of 0.73.
● a softmax classifier with 1000 classes
that produces results similar to the
basic softmax classifier.
● [35]
GoogleNet in
PyTorch
CIFAR10
Class(92.57%
Test Accuracy)
3. VGG 13 ● Three-by-three filters are used in the
first two convolutional layers. With
VGG in
PyTorch
the same convolutions applied to the
top two layers' 64 filters, 224*224*64
volumes are generated. The filters
have a 3*3 step at all times.
● The volume's height and breadth
were then decreased from
224*224*64 to 112*112*64 using a
pooling layer. This layer had a stride
of 2 and a max-pool of size 2*2. This
was finished after two more
convolution layers with 128 filters. As
a result, the new dimension is
112*112*128.
● When the pooling layer is used, the
volume is once more reduced to
56*56*128.
● The size is decreased to 28*28*256
after two further convolution layers
with 256 filters each are added.
● The 7*7*512 volume is flattened into a
Densely Integrated (FC) layer with
4096 channels and a softmax output
of 1000 classes after the final pooling
layer.
CIFAR10
Class(90.17%
Test Accuracy)
1. VGG
By suggesting a ground-breaking design and improving recognition
performance on the ImageNet dataset, VGG rose to become one of the most
well-known CNNs. VGG is unique since each convolutional kernel is three by
three in size. ReLU is applied to each convolutional layer, and some of them also
receive max-pooling. Since VGG uses three FC levels at the very end, the network
doesn't drop out. To maintain an equivalent complexity per layer when a
feature's size is decreased in half, width (i.e., the quantity of feature maps or
filters) must be doubled. Here is another way to interpret such a structure.
2. Inception/GoogLeNet
GooLeNet implements the Inception architecture for ImageNet and features 22
layers, a variety of parameters, and several Inception modules. These distinctive
branches signify different scales of processing. To match or minimise
dimensions, the 11 convolution layer is also employed prior to the 33 or 55
convolution layers (before addition). Actually, the Network-in-Network design
was what first proposed the 11 convolutional layers. It can increase
representational power while maintaining a consistent receptive field size when
used with a conventional convolutional layer.
3) Alexnet
Alex Krizhevsky served as AlexNet's chief architect. It is a CNN, or convolutional
neural network. AlexNet gained notoriety after competing in the ImageNet
Large Scale Visual Recognition Challenge. With a top-5 error of 15.3%, it was
successful. This is behind the winning entry by 10.8%. Five convolutional layers,
three max-pooling layers, two normalisation layers, two fully connected layers,
and one softmax layer make up the AlexNet architecture. Convolutional filters
and the ReLU nonlinear activation function make up each convolutional layer.
Max pooling is carried out using the pooling layers. Because all of the layers are
fully connected, the input size is fixed. Although padding makes the input size
appear to be 224x224x3, it is actually 227x227x3.
DATASET
Alex Krizhevsky served as AlexNet's chief architect. It is a CNN, or convolutional
neural network. AlexNet rose to prominence after partaking in the ImageNet
Large Scale Visual Recognition Challenge. With a top-5 error of 15.32%, it was
successful. This is behind the winning entry by 10.8%. 5 convolutional layers, 3
max-pooling layers, 2 normalisation layers, 2 fully connected layers, and 1
softmax layer make up the AlexNet architecture. Convolutional filters and the
ReLU nonlinear activation function make up each fully connected layer. Pooling
layers help us carry out the max pooling. Because all of the layers are fully
connected, the input size is fixed. Although padding renders the input size
seems to be 224x224x3, it is truly 227x227x3.


REQUIREMENTS :
LIBRARIES USED :
1. Pytorch
2. Torchvision
3. Matplotlib
4. Numpy
5. Joblib


METHODOLOGY :
STEP 1 : LOADING THE DATASET
The CIFAR-10 dataset is supplied with the following classes: truck, ship, horse,
frog, horse, bird, cat, deer, plane, vehicle, and cat. All of the axes of the photos
have been normalised to 0.5. Utilising the torchvision.dataset.CIFAR10, a training
and testing set is produced ().
With a batch size of 4, the data is put into the train loader and test loader. Using
DataLoader from torch.utils.data ().
STEP 2 : DEFINING A CONVOLUTIONAL NEURAL NETWORK
CNN is a feed-forward network.
Error , cost function, are being calculated in the training process, and back
propagation updates the weight of the layers.
All backward functions compute the gradients of the trainable parameters,
whereas the forward function computes the values of the cost function. To build
a neural network, utilise the PyTorch.nn package and the torch library. The
package consists of extensible classes, modules that are required in the path of
formation of a neural network CNN is a neural network which is designed for the
detection of complicated features in data.
In the implementation 3 CNN layers have been defined, in which all layers follow
a hierarchical sequence of cnn, leakyRelu and maxpooling i.e.
Conv2d is used with a kernel size of 5 and maxpool with size 2.
To chain the layers and the activation functions into a single network
architecture we are going to use nn.sequential.
The following are the layers that make up our network:
● Numerous features in the input image are extracted by our neural
network's first layer, which is one of a set of layers.. Between the layers
and a filter we perform convolution that has a specified size of NxN. Then
we go over the image we have put in and dot product is being taken
between the parts of our input image and the used filter with keeping the
size in respect(NxN). The output which we get so far gives us the
information about the image, the information regarding the edges and
the vertices of the image and is called a Featured map . Then we put this
featured map over other layers to explore the other features of the input
image
● The activation function here is ReLU which defines all features which got
through the layer as 0 or greater than (>0). Which means if we apply this
function the negative number becomes zero and the positive ones are
kept the same
● Location of an object can affect the ability of out cnn system to detect the
features specifically hence, Maxpool layer comes into the picture which
helps us ensure no features are being affected.
Impact of Using Dropout in PyTorch
● An unregularized network quickly overfits on the training dataset. The
validation loss for without-dropout run diverges a lot after just a few
epochs. This accounts for the higher generalisation error.
● Training with a dropout layer with a dropout probability of 25% prevents
model from overfitting. However, this brings down the training accuracy,
which means a regularised network has to be trained longer.
● Dropout improves the model generalisation. Even though the training
accuracy is lower than the unregularized network, the overall validation
accuracy has improved. This accounts for a lower generalisation error.
To achieve regularisation, so that you can Effectively avert the complication of
overfitting we use dropout over the layers.
In order to make a prediction via neural network the forward method comes into
the picture. information is flowing forward through the hidden layers starting
from the input to the output because of which the other word for “making
prediction” is running the forward pass. Now comes the backward pass which is
runned when we compute parameter updates this is called by calling
loss.backward(). The information flows from the input layers to the hidden layers
and finally to the output layers.
To call the forward method the __call__ function in the nn.Module is used, so
that when the forward method is called we run model(input).
STEP 3 : DEFINING A LOSS FUNCTION
Calculated by the loss function is the output's deviation from the target.
Through backpropagation in neural networks, the loss function's value is
intended to be decreased.
After each optimization iteration, the loss function informs us of the behaviour of
the model.
We determine model accuracy. It displays the percentage of correctly predicted
outcomes.
We use the Adam Optimizer of the PyTorch neural network package and a loss
function based on the Cross-Entropy loss. The degree to which our network's
weights are adjusted in relation to loss is controlled by learning rate (lr). It will be
set to 0.001. The training moves more slowly the lower the learning rate.
Momentum is set to 0.9 and lr is set at 0.001.
STEP 4 : TRAINING THE DATA
The data is trained with epoch 4 having a batch size of 12500. Forward pass
followed by backward propagation is done in each cycle . Later we perform
Adam optimization,which is an optimization method , which can greatly improve
the performance of the experiment.
All layers have an equal learning rate, which is manually adjusted as training
progresses.
The predictive performance is now defined as the estimated probability of the
model in the test set, or overall accuracy.
STEP 5: TESTING THE DATA
The loss value is printed for every thousand(1k) batches of images for every
iteration. Loss value decreases after every loop.Model correctness is distinct from
loss value. The test data are used to calculate the model's accuracy, which displays
the percentage of correct predictions made. Here, it will reveal how many of the
10,000 test photos our model was able to accurately predict after each iteration.


RESULT ANALYSIS :
The following output is generated. The model stands at an accuracy of 70.62%


FUTURE SCOPE AND CONCLUSION:
Computer vision applications can make machines more self-sufficient and
ensure product quality. If processing systems are quicker than humans, it should
be simple to incorporate inline quality controls, such as 100% controls. Damaged
components can be replaced or fixed, increasing the productivity of
manufacturing operations.
Two major issues in agriculture are water scarcity and yield quality. Tracking
satellite imaging of the fields can enable the monitoring and data sharing of
irrigation. An additional tool for gathering and reporting irrigation is infrared
image processing. The results of this research can then be used to inform
pre-harvesting decisions concerning whether to harvest or not.A mix of machine
learning and image processing algorithms and methodologies can be used to
detect the growth of weeds.
2D images are converted into 3D images with a sense of depth using 3D
imaging technology. The 3D model is then rendered and colours and textures
are added to make it look realistic. Thanks to such 3D imaging and rendering,
doctors can see 3D images of her organs in very high quality that have never
been seen before. You can use it to perform delicate procedures and make
accurate diagnoses.
In this study, we classified images using CNN and the CIFAR-10 dataset. The
hierarchical sequence of CNN, LeakyRelu, and Max-Pooling is followed by all
three of the CNN layers defined in the code sample below. CNNs are effective at
removing features from images. As a result, the neural network model's accuracy
increased dramatically, from 65% to 70.62%.
To further increase accuracy, you can experiment with CNN model
hyperparameters. The number of dense layers, the number of hidden units in
each dense layer, the number of epochs, the number of convolutional layers, the
number of filters in each convolutional layer, and other hyperparameters can all
be adjusted.


REFERENCES :
[1] J. J. DiCarlo, D. Zoccolan, and N. C. Rust, "How does the brain solve visual object
recognition?," Neuron, vol. 73, no. 3, pp. 415-434, 2012.
[2] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, pp.
436–444, 2015.
[3] J. Schmidhuber, "Deep learning in neural networks: An overview," Neural
Networks, vol. 61, pp. 85-117, 2015.
[4] Y. Bengio, A. Courville, and P. Vincent, "Representation Learning: A Review and
New Perspectives," PAMI, vol. 35, no. 8, pp. 1798-1828, 2013.
[5] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with
deep convolutional neural networks," in NIPS, 2012, pp. 1097-1105.
[6] Y. Bengio, "Learning Deep Architectures for AI," Foundations and Trends in
Machine Learning, vol. 2, no. 1, pp. 1-127 , 2009.
[7] K. He, X. Zhang, S. Ren, and J. Sun, "Delving Deep into Rectifiers: Surpassing
Human-Level Performance on ImageNet Classification," in ICCV, 2015.
[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image
Recognition," in CVPR, 2016.
[9] R. M. Cichy, A. Khosla, D. Pantazis, A. Torralba, and A. Oliva, "Comparison of
deep neural networks to spatio-temporal cortical dynamics of human visual object
recognition reveals hierarchical correspondence," Scientific Reports, vol. 6, no.
27755, 2016.
[20] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, "Aggregated Residual
Transformations for Deep Neural Networks," in CVPR, 2017.
[21] T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural
Networks with Cutout," arXiv, 2017.
[22] X. Gastaldi, "Shake-Shake regularization," arXiv, 2017.
[23] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradientbased learning applied
to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324,
1998.
[24] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift," in ICML, 2015.
[25] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov,
"Dropout: a simple way to prevent neural networks from overfitting," The Journal
of Machine Learning Research, vol. 15, no. 1, pp. 1929-1958, 2014.
[26] D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in
International Conference on Learning Representations (ICLR), 2015.
[27] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition," arXiv, 2014.
[28] C. Szegedy et al., "Going deeper with convolutions," in CVPR, 2015.
[29] M. Lin, Q. Chen, and S. Yan, "Network in network," CoRR, 2013.
[30] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images,"
Technical report, 2009.
[31] A. Vedaldi and K. Lenc, "MatConvNet: Convolutional Neural Networks for
MATLAB," in ACM international conference on Multimedia, 2015, pp. 689-692.
[32] A. Nguyen, J. Yosinski, and J. Clune, "Deep Neural Networks are Easily Fooled:
High Confidence Predictions for Unrecognisable Images," in Computer Vision and
Pattern Recognition, 2015.
[33]Image Processing And Its Future Implications
[34]Alex Krizhevsky University of Toronto kriz@cs.utoronto.ca Ilya Sutskever
University of Toronto ilya@cs.utoronto.ca Geoffrey E. Hinton University of
Toronto hinton@cs.utoronto.ca,”ImageNet Classification with Deep Convolutional
Neural Networks”, 2012
[35] Deep Learning: GoogLeNet Explained | by Richmond Alake | Towards Data
Science

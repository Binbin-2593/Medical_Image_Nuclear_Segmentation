![Test Badge](https://github.com/Binbin-2593/Medical_Image_Nuclear_Segmentation/blob/master/.github/workflows/python-app.yml/badge.svg) 

# Medical Image Nuclear Segmentation
Before I go over anything else, here's a link the website hosting the app:

[Segmentation Homepage](https://binbin-2593-medical-image-nuclear-segmentatio-srcdisplay-cnn9qf.streamlit.app)

Technologies used: 
* Python
* Pytorch
* Streamlit
* Docker

# Summary
This project involved using Deep Convolutional Neural network to create a machine learining application that could classify 250 bird species based on images. The model architecture is a [ResNet50](https://en.wikipedia.org/wiki/Residual_neural_network) that was initially trained on the [ImageNet Dataset](https://en.wikipedia.org/wiki/ImageNet). Transfer learning was utilized to fine tune the ImageNet model to learn how to classify birds. After training, the model correctly identified 97% of bird images held out from training. The trained model was then deployed in an interactive website to allow users to identify their own bird pictures.

# Dataset
The dataset used for this project was found [on Kaggle](https://www.kaggle.com/gpiosenka/100-bird-species). Someone else went through the hard work of compiling and cleaning bird images so that I didn't have to. The dataset included 250 species of birds with about 100 - 130 training images of each species. Although this class imbalance did exist in the training data, it did not substantially affect the model scores. The validation and test data each included 5 images of each species. 

In any given image, the bird was near the center of the image and took up at least 50% of the image. This made it great for training but not the best for use in real world inference. Having said that, each species of bird had a variety of different positions they would be in including flying, sitting, perched on trees, etc. Additionally, image augmentation was critical to a high scoring model. Although any model trained on this data would not likely be able to correctly identify a bird from very far away, it would be likely to correctly identify a bird regardless of what position the bird was in.
### Barn Swallow:
![](imgs/barn_swallow.jpg)

### Tree Swallow:
![](imgs/tree_swallow.jpg)

# Streamlit App

I created a publically hosted application using Streamlit to showcase this project and allow users to interact with the trained model with a no-code implimentation. Users can select from any of the images I used for training, validation, and testing or they can upload their own image and see how the model would classify it.

The app outputs a table of the top five predictions including confidence levels of each prediction and a link to the Wikipedia page of the bird species in case users want to learn more.
![](imgs/st_app_shot.jpeg)

# GANs - Generative Adversarial Networks
> Objective - Trained a GAN to produce a dataset of Cars of size 128x128  

## What are GANs? - A quick review  
GANs were introduced in [a paper](https://arxiv.org/abs/1406.2661) by Ian Goodfellow and other researchers at the University of Montreal, including Yoshua Bengio, in 2014.  
- GANs are DNNs comprised of two nets, pitted one against the other (thus adversarial)
- They are made of two distinct models, a generator and a discriminator. 
- The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator.
- GANs are neural networks that are trained in an adversarial manner to generate data mimicking some distribution. 

## What's in here? - DCGAN  
**Deep Convolutional Generative Adversarial Networks** explicitly use convolutional and convolutional-transpose layers in the discriminator and generator, respectively.  
In this example, the discriminator is made up of strided convolution layers, batch norm layers, and SeLU activations.   
The generator is comprised of convolutional-transpose layers, batch norm layers, and SeLU activations.   

## Dataset  
In this example, images from carvana.com have been taken. You can download these images from [here](https://github.com/Divya932/EVA4-Phase-2/blob/master/Session06%20-%20GANs/Dataset/cars_raw.csv). However, this particular GAN focuses only on Sedan cars. [Here](https://github.com/Divya932/EVA4-Phase-2/blob/master/Session06%20-%20GANs/Dataset/sedans.csv) the link to segregated sedans.

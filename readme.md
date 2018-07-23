# **Pokédex Imagenet**

## **Summary**
My first attempt in recreating the Pokédex experience with the help of Deep Learning algorithms and a smartphone. To be more precise, models used for this project were a smaller version of the VGGnet, the full version of the VGG16net and the MobilenetV2. The following will be a walkthrough of my experience, thoughts as well as the results of this project. Towards the end I included a How-To guide in using my scripts.

## Using:
* iPhone 6
* Keras 2.2.0 with Python 3.6 (for MobileNetv2)
* Keras 2.1.6 with Python 2.7 (for coremltools to convert to iOS compatible models)
* Training data classes (11): 
    * Background, Bulbasaur, Charizard, Charmander, Dragonite, Gengar, Gyrados, Pikachu, Snorlax, Squirtle, Zapdos

-----

## **Walkthrough**
### Background:

Since most of my projects usually end up being closely related to the video-game culture or my childhood oriented series, I decided it would be cool to recreate the Pokédex experience of the popular series Pokémon. 

In the series, the Pokédex is a device used to classify Pokémon encountered in the wild. This of course was a great opportunity to use my newly gained Machine Learning experience and make the Pokédex come to life!... or at least a simple prototype of which. Expecially convolutional neural networks (CNN) came to mind when approaching this since images would be my inputs and classifying these is its strong suit.

It was also due to the tremendous help of [Adrian Rosebrock's tutorials](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/) that I had a good footing for this project, which brings me to my first attempt.

&nbsp;
### **Collecting and pruning Pokemon:**

My first take on this was of course to follow his tutorial in doing something surprisingly similar in motivation. Since his model is made to classify 5 different Pokémon of which around 300 images of each Pokémon were gathered by him beforehand, I wanted go through the process from the beginning step by step.

In a previous tutorial, he expressed his pleasant experience with the [Bing Image Seach API](https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api) to quickly gather a huge amount of training data. This sounded great since I really did not want to go through the web an download each image individually. Sadly, the free version only allowed monthly transaction limit of 3000 and [Google's version](https://developers.google.com/image-search/) was announced deprecated a while back in 2011. This did not suffice for me since I would easily go over this threshold, especially if I would pursue similar projects. An alternative is required!

After searching around I managed to find one that goes through Google Images with the help of selenium to target more than 100 images per run. With this I managed to download 1000 images of 5 new Pokémon of my choice with relative ease. The only tedious part was the pruning step. An average of 200 images for each Pokémon were actually useful for training purposes. As a side note, I actually planned to include all 151 Pokémon of the first generation but I quickly realized the amount of manual labor it would require to prune them. So I was content with 5 additional ones. In total I had 10 unique Pokémon and 1 background class some background images of offices, which I needed when I'm not targeting Pokémon. After pruning, this amounted to around 3000 images.


&nbsp;
### **Preprocessing:**

Before feeding the images to my model, certain steps had to made beforehand so that the model can properly train. On the one hand, training data have to be **split** in to for training and validating. On the other hand, the data should be resized and and its raw pixel intensities normalized to be handled correctly. Automating the first step was my first goal so that I can simply take the output of my downloads and split the data in such a way that training and validation data had a ratio of 80/20 respectively while keeping the directory structure intact. The following is a rough depiction of the directory structure for both the training and validation data:

* data/
	* train/
  		* class 0/
			* image.jpg
			*	....
		* class 1/
		*  ....
		* class n/
    *  validation/
  		* class 0/
		* class 1/
		*  ....
		* class n/


&nbsp;
### **SmallerVGG:**

The one proposed by Adrian was a [VGGnet-like](https://arxiv.org/abs/1409.1556) ConvNet that only uses 5 Convolutional layers instead of the standard 16 or 19. In my case, results of this was subpar as can seen below:

![SmallerVGG Plot](https://github.com/CouchCat/Pokedex-Imagenet/blob/master/plots/plotSm11.png)


While the accuracy was converging towards 0.8 and the training loss continuously sank towards 0, the validation loss seemed to indicate an over-fitting problem due to it  being very high. After several more tries with differnt parameters with no luck, I decided to take a different approach.

&nbsp;
### **VGG16**

To counter the over-fitting problem, I could have simply gathered more data but that would be tedious. I instead implemented data generators that uses **augmentation** to mulitply the data at hand. By creating the same images but with slightly different translation, rotation and scale values, training data could be used more effectively. I also decided to use a pre-trained VGG16 model by Keras and **fine-tune** the last layer to my case so that I could use it as a ready-made feature extractor. All layers exept the top were also made frozen so that only weights of the top layer remain trainble.

![VGG16](https://github.com/CouchCat/Pokedex-Imagenet/blob/master/plots/plot-vgg-plot_v2.png)

Validation loss still seemed to make some problems here due to its high values. Since my dataset is completely different to the Imagenet weights the model was trained with, I trained it again though with all layers unfrozen this time. I finally received better results.

![VGG16](https://github.com/CouchCat/Pokedex-Imagenet/blob/master/plots/plot-vgg-plot_v3-11.png)

While the results were better, training and validation loss were till apparent since they converged at around 0.3 instead of the hoped for 0.1 area. Still, I decided to try it out with on my iPhone 6 to see its performance. By using [CoreML](https://developer.apple.com/documentation/coreml) Keras models could be easily converted to a model that iOS devices can use.

The experience with it, however, was not as hoped for since it could only classify images captured by the camera every few seconds at the time. It was very **slow**! A proper experience also requires speed!

&nbsp;
### **Mobilenetv2:**

To this end, I decided to use the Keras model [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) as this model is more suited for mobile applications. Augmentation was performed as well as letting all layers be trainable.

![Mobilenetv2](https://github.com/CouchCat/Pokedex-Imagenet/blob/master/plots/plot-mobile2_v4.png)


Results seemed much better now since accuracy converged to 1.0 and loss towards 0.1, depicting a healthy model. Trying it out on my iphone 6 I could also achieve good results and very reasonable speed.

While also comparing the actual file size of the converted models, it was clear why the VGG16 model performed so poorly. VGG16 took around 160 MB of space while the MobilenetV2 only 10 MB! Also for the sake of comparison, here are the results of MobilenetV2 with frozen layers.

![Mobilenetv2_frozen](https://github.com/CouchCat/Pokedex-Imagenet/blob/master/plots/plot-mobile2_v5_frozen.png)


### **Technical Notes:**

The MobileNet architecture uses **depthwise separable convolutions**, which significantly reduces the number of parameters when compared to the network with normal convolutions (VGGnet) with the same depth in the networks. This results in light weight deep neural networks.

This is favorable in mobile and embedded vision applications with **less compute power** since this reduces the total number of floating point multiplication operations. By using depthwise separable convolutions, there is some **sacrifice of accuracy** for low complexity deep neural network.

MobileNetV2 builds upon the ideas from MobileNetV1, however, V2 introduces two new features to the architecture: 
* 1) linear bottlenecks between the layers
* 2) shortcut connections between the bottlenecks

![MobileNetv2_v1_comparison](https://2.bp.blogspot.com/-E7CT0RHBWq4/WsKlTgEeX2I/AAAAAAAACh0/dp1B4yh6O2k4H1LuC7BA-EKzrL7W0L8iACLcBGAs/s1600/image2.png)

Overall, the MobileNetV2 models are faster for the same accuracy across the entire latency spectrum. In particular, the new models use 2x fewer operations, need 30% fewer parameters and are about 30-40% faster than MobileNetV1 models, all while achieving higher accuracy. Even using my relatively dated iPhone6, MobileNetV2 still manages to achieves good performance.



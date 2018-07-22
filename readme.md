# **Pokédex Imagenet**

## **Summary**
My first attempt in recreating the Pokédex experience with the help of Deep Learning algorithms and a smartphone. To be more precise, models used for this project were a smaller version of the VGGnet, the full version of the VGG16net and the MobilenetV2. The following will be a walkthrough of my experience, thoughts as well as the results of this project. Towards the end I included a How-To guide in using my scripts.

-----

## **Walkthrough**
&nbsp;
### Background:

Since most of my projects usually end up being closely related to the video-game culture or my childhood oriented series, I decided it would be cool to recreate the Pokédex experience of the popular series Pokémon. 

In the series, the Pokédex is a device used to classify Pokémon encountered in the wild. This of course was a great opportunity to use my newly gained Machine Learning experience and make the Pokédex come to life!... or at least a simple prototype of which. Expecially convolutional neural networks (CNN) came to mind when approaching this since images would be my inputs and classifying these is its strong suit.

It was also due to the tremendous help of [Adrian Rosebrock's tutorials](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/) that I had a good footing for this project, which brings me to my first attempt.

&nbsp;
### Collecting and pruning Pokemon:

My first take on this was of course to follow his tutorial in doing something surprisingly similar in motivation. Since his model is made to classify 5 different Pokémon of which around 500 images of each Pokémon were gathered by him beforehand, I wanted go through the process from the beginning step by step.

In a previous tutorial, he expressed his pleasant experience with the [Bing Image Seach API](https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api) to quickly gather a huge amount of training data. This sounded great since I really did not want to go through the web an download each image individually. Sadly, the free version only allowed monthly transaction limit of 3000 and [Google's version](https://developers.google.com/image-search/) was announced deprecated a while back in 2011. This did not suffice for me since I would easily go over this threshold, especially if I would pursue similar projects. An alternative is required!

After searching around I managed to find one that goes through Google Images with the help of selenium to target more than 100 images per run. With this I managed to download 1000 images of 5 new Pokémon of my choice with relative ease. The only hardship of this part was the pruning step. An average of 300 images for each Pokémon were actually useful for training purposes. I actually planned to include all 151 Pokémon of the first generation but quickly realized the amount of manual labor it would be to prune them, so I stuck with 5 additional ones. In total I had 10 unique Pokémon and 1 background class some background images of offices, which I needed when I'm not targeting Pokémon. After pruning, this amounted to around 3000 images.


&nbsp;
### Preprocessing:

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
### SmallerVGG:

The one proposed by Adrian was a [VGGnet-like](https://arxiv.org/abs/1409.1556) ConvNet that only uses 5 Convolutional layers instead of the standard 16 or 19. In my case, results of this was subpar as can seen below:






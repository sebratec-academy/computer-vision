#!/usr/bin/env python
# coding: utf-8

# ## Before we begin
# 
# This material assumes that you have an introductory level of neural networks. If you feel you need to reinforce your knowledge on these subjects, take a look into our [Deep learning foundation course materials](https://github.com/sebratec-academy/deep-learning-foundation), and consider enrolling into Sebratec academy! :D

# ## Convolutional neural networks
# 
# 

# Neural networks can be applied to a lot of different challenges, and image recognition is one of them. In this first part of the material, you will be introduced to a particular neural network architecture called convolutional neural network, or CNN, in short. This architecture is really good in exploring properties of image data.

# ## Properties of an image
# 
# Before we can explore how CNNs can take advantage of these properties of images, we need to talk about what an image is, and about its properties. Images are composed of pixels, made from one or more measurements of light, and these pixels are the smallest elements present in a screen.
# 
# 
# Consider a black and white image. It can be represented into a 2D array of numerical values. Each index of this array, in a black and white image, holds a value between 0 to 255, where 0 is black, 255 is white, and the numbers in between are shades of gray.
# 
# ![bw-pixels](https://user-images.githubusercontent.com/20716798/78840184-64507200-79fa-11ea-8117-6660e489a3e0.png)
# 
# 
# For RGB images, it is a little different. There, the pixels are arranged in a 3D array. The first two dimensions represent the position of the pixels that form the image, and the colors are stored inside the third dimension. The third dimension has 3 layers, each corresponding to an RGB color. So, each pixel has three values, one for the amount of red, one for the amount of green, and one for the amount of blue.
# 
# ![rgb-pixels](https://user-images.githubusercontent.com/20716798/78840721-d70e1d00-79fb-11ea-9b30-180b2be0f629.png)
# 
# 
# These layers are overlayed, and the final pixel is composed of the mix of these three color values. So, if you take for example a pixel with (255, 125, 80) RGB values and the final mix would look like this:
# 
# ![mixed colors](https://user-images.githubusercontent.com/20716798/77757571-ba540b80-7031-11ea-9017-56fa4059327d.png)

# ## Feeding an image to a neural network
# 
# So, in the end, an image is a large 3D array of numbers. If your image has a dimension of 512 * 512, for example, you would have 786.432 numbers inside it.
# 
# *to find this number, just multiply the dimensions. In this case, 512 * 512 * 3*
# 
# Each of these numbers is an individual feature of the image. Remember that the number of parameters of a neuron in a neural network is the number features and bias, so our neural net, in this case, would have 786.432 + 1 parameters, and that is a lot of parameters for a neural network, that not only is inefficient but also can lead to overfitting very quickly.
# 
# Let's say you are trying to recognize something in an image. For example, a car. The car can be located at any point between these 786.432 parameters, and every car image can be very different from one another. The car can be in different positions, in different scenarios, with or without noise in the image, from different angles, with different lighting, and if you analyze these images represented as data, they will be very different from each other, but a car will still be there.

# ## How CNNs help to address these concerns
# 
# To address these concerns, we will use a convolutional neural network, in short, CNN, which is a special kind of neural network that can find useful patterns in an image. As mentioned above, many neural networks are fed with individual inputs, and since each color value inside each pixel is a different input, it would take too much time to learn from these inputs, and they can also lead to overfitting.  
# 
# CNNs are useful in this case because instead of using every single input from our image, it uses a technique called parameter sharing, which applies a filter, also called a convolutional kernel, to groups of pixels in different areas of an image, instead of analyzing the whole image at the same time. 
# 
# Every CNN is composed of multiple layers, and these layers usually are convolutional, pooling and fully connected layers. Each produces an output that is used for the next one to produce your expected result.

# ## The convolutional layer
# 
# The first important layer of a convolutional neural network is called the convolutional layer. This layer is defined by the filter size, stride, depth and padding parameters, and is the layer inside a CNN responsible for applying the filter to the image. These filters are usually small grids of values that slide over an image, pixel by pixel and outputs a filtered image exposing the extracted feature. The resulting image will be about the same size as the original image.
# 
# Under the hood, the filter works by applying convolution to the image, pixel by pixel.
# 
# ![3D_Convolution_Animation](https://user-images.githubusercontent.com/20716798/78149745-a136ca80-7436-11ea-9097-fbb6b99e8cf1.gif)
# 
# The convolution works by:
# 
# - Multiplying the values in the kernel with their matching pixel value. So, the value in the top left of our filter (0), will be multiplied by the pixel value in that same corner in our image area (7).
# 
# - Sum all these multiplied pairs of values to get a new value, in this case, 9. This value will be the new pixel value in the filtered output image, at the same location as the selected center pixel.
# 
# 
# 
# \begin{equation}
#   \begin{bmatrix}
#       0  &  -1  &  0  \\
#       -1  &  5  &  -1  \\
#       0  &  -1  &  0 \\
#   \end{bmatrix}
#   .  
#   \begin{bmatrix}
#     7  &  7  &  6  \\
#     7  &  7  &  6  \\
#     6  &  6  &  4 \\
#   \end{bmatrix}
#   =
#   \sum
#   \begin{bmatrix}
#     0  &  -7  &  0  \\
#     -7  &  35  &  -6  \\
#     0  &  -6  &  0 \\
#   \end{bmatrix}
#   =
#   9
# \end{equation}
# 
# It is also important to note that these values inside the filters are called weights. The weights determine how important the pixel is when forming the output image. In our example, the center weight is five, meaning that the center pixel is the most important one in our filter.
# 
# 
# The filter has a size, which corresponds to how many inputs features in the width and height dimensions one neuron takes in. We do not split up the image by its depth (or the channels), only the width and height. So if we specify the filter size, the number of inputs that our filter will take is filter_width * filter_height * input_depth + 1.
# 
# 
# ![filter](https://user-images.githubusercontent.com/20716798/77867552-083a6080-7238-11ea-8ebb-b5318fb03d52.png)
# 
# And to apply this filter to an image that is bigger than itself, we must move it. We specify how many pixels the filter is going to move with the **stride** hyperparameter. When the stride is 4, for example, then the filters jump 4 pixels at a time as we slide them around. Having a larger stride will produce smaller feature maps.
# 
# 
# ![stride](https://user-images.githubusercontent.com/20716798/77869803-12f8f380-7240-11ea-8b36-965114ff669b.gif)
# 
# You can also use multiple filters instead of just one. The amount of these filters is the hyperparameter called **depth**. These filters produce filtered images based on the input image, extracting a feature from this image that we can use to achieve our desired result. Our filters can, for example, extract the edges from our original image, others can detect useful color patterns and so on.
# 

# In[ ]:





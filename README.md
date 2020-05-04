![sebratec logo](https://user-images.githubusercontent.com/20716798/74448368-1ea07e80-4e7b-11ea-9e73-5c29ad328fc0.png)

![quality assurance](https://github.com/sebratec-academy/computer-vision/workflows/quality%20assurance/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/5772d3acc11fdc3a9b4e/maintainability)](https://codeclimate.com/github/sebratec-academy/computer-vision/maintainability)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sebratec-academy/deep-learning-foundation/issues)

# Computer Vision

This repository contains the course materials of Computer Vision course, which is held at Sebratec academy, in Gothenburg, Sweden.

**Important:** This is still a work in progress. These materials may change until the final release.

## Course Abstract

The computer vision course is presential and will take place in Sebratec academy’s headquarters. The total duration is 5 weeks, and you will have 2 classes per week, which will be 2 hours long each. Expect to have a high volume of laboratory classes, because this course is intended to provide you as much hands-on experience as possible. You will discuss cutting edge research material and develop your own computer vision systems during our laboratory sessions.

## Running the materials

Enrolled students have the convenience of an environment that is ready to run these materials. You can find it at https://lab.sebratec.com.

You are also free to run these materials in your computer, if you like. You will need to install python 3, pip3 and the following packages:

- numpy
- pandas
- sklearn
- matplotlib
- keras
- tensorflow

## Outline 

| Schedule                                    | Topics                                           | Learning outcomes                                                                                                                                                                                                                                                                                                                                                                                                                                            | Assignment                                                                                                                                      |
|---------------------------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Week 1            | Welcome, introduction, and simple convolutional neural networks |In the first session, you will meet your peers and teacher, understand what computer vision is, the history of computer vision and how it is changing the world. You will also be introduced to the math behind convolutions, convolutional layers, pooling layers, and deconvolutions. In the second session, you will have a hands-on laboratory class implementing a convolutional neural network yourself.                             | Lab: Simple convolutional neural networks                                                                                      |
| Week 2            | Image classification and transfer learning                 | In the third session, you will learn about image classification, object detection and segmentation, inception layers and transfer learning. In the fourth session, you will have a hands-on laboratory class to put into practice what you have learned in session three. Here, you will build a convolutional neural network, apply transfer learning to it, and use it to classify images. | Lab: Build a convolutional neural network, apply transfer learning to it, and use it to classify images                      |
| Week 3           | Object segmentation, detection, and generative adversarial networks           | In the fifth session, you will have a hands-on laboratory class where you will perform object detection and segmentation. In the sixth session, you will learn about architectures designed for object detection and segmentation and generative adversarial networks.                                                                                            | Lab: Object detection and segmentation                                                                                                                          |
| Week 4           | Autoencoders and generative adversarial networks              | In the seventh session, you will have a hands-on laboratory class about autoencoders, and their applications. In the eighth session, you will have a laboratory session focused on generative adversarial networks and their applications.                                                                                                        | Lab: Autoencoders Lab: Generative adversarial networks |
| Week 5 | Final project                                    | This final week is dedicated for project reviews, office hours and orientation. You must submit your project before the deadline                                                                                                                                                                                                                                                                                                                             | Project submission                                                                                                                              |
| Week 6                    | Graduation                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                 |


## Contributing
Contributions are more than welcome! Please make a pull request whenever you feel like you can improve this material. Just have in mind that the requirements for a pull request to be considered are:

- Pull requests must pass the quality audit check;
- Pull requests must not lower the project maintainability score in codeclimate.

When adding a jupyter notebook to the material, you must also:

- Add it to the [.github/workflows/main.yml](https://github.com/sebratec/deep-learning-foundation/blob/master/.github/workflows/main.yml) file so the quality audit can be performed on it;
- Export it to python (.py) and add it to the [.codeclimate](https://github.com/sebratec/deep-learning-foundation/tree/master/.codeclimate) folder, so codeclimate can analyze it.

Contributions will only be reviewed if they meet these requirements.

# ![](assets/uncertify.gif)
[![Build Status](https://travis-ci.com/matthaeusheer/uncertify.svg?token=bUEVgaJ1xLP8HNxuqXvL&branch=master)](https://travis-ci.com/matthaeusheer/uncertify)

This repository holds the code for our paper [The OOD Detection Blindspot of Unsupervised Lesion Detection](https://openreview.net/forum?id=ZDD2TbZn7X1) which has been accepted for the [MIDL](https://2021.midl.io/) (Medical Imaging with Deep Learning) 2021 conference. This project has been carried out in the the scope of a Masters Thesis at the [Computer Vision Lab](https://vision.ee.ethz.ch/) at [ETHZ](https://ethz.ch/en.html).


<!-- TABLE OF CONTENTS -->
## Table of Contents
* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project
Pleaser refer to the MIDL 2021 publication [The OOD Detection Blindspot of Unsupervised Lesion Detection](https://openreview.net/forum?id=ZDD2TbZn7X1).

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


### Prerequisites
- [pyenv](https://github.com/pyenv/pyenv) - to manage your local python versions
- [pipenv](https://github.com/pypa/pipenv) - manage python dependencies within a virtual environment
- [cuda](https://developer.nvidia.com/cuda-toolkit) - well, the NVIDIA stuff
- [cuDNN](https://developer.nvidia.com/cudnn) - more NVIDIA stuff, deep learning on steroids


### Installation

1. Clone the repo
```sh
git clone https://github.com/matthaeusheer/uncertify.git
```
2. Install python dependencies via `pipenv`
```sh
cd uncertify
pipenv install
```
3. Active virtual environment
```sh
pipenv shell
```


<!-- USAGE EXAMPLES -->
## Usage
Please check out the python scripts in the `scripts` folder and jupyter notebooks in the `notebooks` folder.


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contributors
Matthäus Heer - maheer@ethz.ch / matthaeus.heer@uzh.ch
Janis Postels
Xiaoran Chen
Shadi Albarqouni

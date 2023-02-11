<h1><center>MASKTUNE: Mitigating Spurious Correlations by forcing to explore</center></h1>
Project for Neural Networks exam.

Contributors: Giordano Cicchetti, Jacopo Brunetti.

In this project we have reimplemented the technique described in the paper [MaskTune: Mitigating Spurious Correlations by Forcing to Explore.](http://arxiv.org/abs/2210.00055) 

We have selected two main tasks: 
1. (a) Classification in presence of spurious correlations;  

2. (b) Selective classification (classification with option to abstain).

We cover three different datasets under (a): 


1.   MNIST with synthetic spurious features;
2.   CelebA;
3.   BackGround Challenge (IN9);

 
We cover two different datasets under (b):

1.   CIFAR-10;
2.   SVHN;

Code and details are in the Python Notebooks.

# How to run

For each investigated dataset, in the corrispondent folder, we have produced a dedicated Python Notebook.
We suggest you to start from MNIST, because it is the principal Notebook where we put also different annotations on our code.


To execute the code as first select runtime=GPU and then you have to run cells in the Python Notebooks.

Each Notebook is divided in sections:
1. Dependecies Installation, Datasets and Models loading;
2. Network Training;
3. Network Testing before MaskTune application;
4. MaskTune application;
5. Final results Testing.

Each cell of the notebooks can be executed, some of them requires a large amount of time, e.g. the cells for training the networks.
For this pourpose we have stored the trained model in this GitHub or in a Google Drive folder and can be loaded in the Notebooks using dedicated cells. So if you want to skip the training phase you can load pretrained models and you can test them using the test datasets.


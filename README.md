# Blood-Cells-Recognition
This project is made by an in-class Kaggle-like competition. (Machine Learning II Exam 2 Competition - By Prof. Amir Jafari)
<br>

## Introduction
You can read more about this disease in the following link: https://www.cdc.gov/malaria/about/biology/. If a model can successfully identify these types of cells from images taken with a microscope, this would allow to automate a very time-consuming testing process, leaving human doctors with more time to treat the actual disease. Furthermore, an early malaria detection can save lives!
<br>
Towards the leader-board, we will use the Binary Cross-Entropy loss. The targets will be one-hot-encoded following the order: ["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]. For example, if an image contains red blood cells and trophozoite cells, the target will be [1, 0, 0, 1, 0, 0, 0].
<br>

## Environment
I run these codes in the cloud computing engines, like GCP or AWS. The deep learning package is Pytorch, and I will run my codes as tensors in GPU. Please be aware you need to have enough CUDA memory to run these codes. For your information, my computing engine has 8GB CUDA memory.

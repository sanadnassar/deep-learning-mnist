Handwritten Digit Classification with CNN:-

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the
Modified National Institute of Standards and Technology (MNIST) dataset. It is built with PyTorch framework and includes
features like training, saving/loading model checkpoints, and visualizing training metrics.


 

Here are some key features of the  project:

1. CNN Architecture: A simple yet effective CNN for image classification.

2. Training & Checkpoints: Train the model on the MNIST dataset and save checkpoints for easy restoration.

3. Visualizations: Plot training loss and accuracy over epochs. During the training process, the model's loss and
   accuracy are plotted to help visualize the training progress, these graphs show how the model's performance improves over time.

4. Model Evaluation: Evaluate model performance on the test dataset.


Training process:-

The model is trained on the MNIST dataset, which contains 70,000 handwritten digits (0-9). In the first session, the model
is trained for a set number of epochs, and the model and optimizer states are saved in a checkpoint file (checkpoint1.pt).
In the second session, the checkpoint is reloaded, and training continues for additional epochs. After each training session, the model
is evaluated on the test dataset to track its performance.


Made by Sanad Nassar

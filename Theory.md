Introduction to Deep Learning


NAME:	Arshiya M. Saiyyad
ROLL NO.:	UGMR20230014


A. What is Deep Learning?
Deep learning is a specialized branch of machine learning, which itself is a subset of artificial intelligence. It involves the use of artificial neural networks, which are inspired by the structure and function of the human brain, to process and learn from large amounts of data.
Key Characteristics:
Multiple Layers: Unlike traditional neural networks, deep learning models have multiple layers (hence the term "deep") that allow them to learn and represent data in increasing levels of abstraction.

Data-Driven: These models require vast amounts of data to learn effectively, making them suitable for applications where data is abundant.

Complex Feature Learning: Deep learning models automatically discover intricate patterns in data, eliminating the need for manual feature extraction.
Example:
Imagine you want to teach a computer to recognize different animals in photos. You'd provide it with a vast collection of labeled images (e.g., cats, dogs, horses). The deep learning model would learn to identify the unique features of each animal through multiple layers of processing.


B. Basic Concepts of Deep Learning
1. Neurons and Layers:
Neurons: The basic units of a neural network that receive inputs, process them, and pass on the output to the next layer.
Layers: Groups of neurons. There are three main types:
oInput Layer: Receives the initial data.
oHidden Layers: Intermediate layers where the data is processed. There can be multiple hidden layers.
oOutput Layer: Produces the final prediction or classification.
2. Activation Functions:
These functions determine if a neuron should be activated (fired) or not. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.
3. Training and Learning:
Forward Propagation: Data is passed through the network from the input to the output layer, making predictions.

Backward Propagation: The network adjusts its weights based on the errors in predictions, using algorithms like gradient descent to minimize the error.
4. Loss Function:
A function that measures how well the neural network's predictions match the actual data. The goal of training is to minimize this loss.


C. Convolutional Neural Networks (CNNs)
What are CNNs?
CNNs are a type of deep learning model specifically designed for processing structured grid data like images. They are highly effective for image recognition and classification tasks.
Key Components:
Convolutional Layers: These layers apply filters to the input image to create feature maps. Each filter detects specific features, such as edges or textures.

Pooling Layers: These layers reduce the dimensionality of feature maps, retaining essential information while making computations more efficient.

Fully Connected Layers: After several convolutional and pooling layers, the output is flattened and passed through fully connected layers to make final predictions.
Example:
In image classification, CNNs can automatically learn to identify various features of animals, like fur patterns, shapes, and colors, and classify images accordingly.
D. Recurrent Neural Networks (RNNs)
What are RNNs?
RNNs are a type of neural network designed to handle sequential data. They are particularly effective for tasks where context and order are important, such as time series analysis, natural language processing, and speech recognition.
Key Components:
Recurrent Connections: Unlike traditional neural networks, RNNs have connections that loop back, allowing them to maintain information across sequences. This helps in remembering the previous inputs while processing the current one.

Hidden States: These states store information about previous inputs, enabling the network to understand context and sequence.
Example:
RNNs can be used for language translation, where understanding the context of previous words is crucial for accurate translation.

E. Transformers
What are Transformers?
Transformers are a recent and highly advanced type of neural network architecture, particularly effective for natural language processing tasks. They have revolutionized the field by enabling models to handle long-range dependencies and parallelize training.
Key Components:
Attention Mechanism: This allows the model to weigh the importance of different parts of the input sequence, focusing more on relevant parts. It helps in capturing relationships and dependencies in data.

Encoder-Decoder Structure: In tasks like language translation, the encoder processes the input sequence, and the decoder generates the output sequence.
Example:
Transformers are used in models like GPT-3, which can perform a wide range of language tasks, from writing essays to answering questions, by understanding and generating human-like text.

Github Collaboratory Link:

https://github.com/Arshiya109/Fundamentals-of-Deep-Learning

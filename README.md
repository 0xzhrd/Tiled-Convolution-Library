# Tiled-Convolution-Library

Header only CUDA library for 2D convolution with arbitrary user supplies filter banks; auditable, dependency free, and portable across Jetson Orin(sm_87), Ada(sm_89)
Benchmarked against cuDNN, NPP, and OpenCV-CUDA.

//fill after the sweep, should contain the orin heatmap and then a reading interpretation


## Purpose of this
Some image filters belong outside the neural network, some classical hand-crafted filters are still valuable; especially in real time, safety critical, or resource constrained systems(i.e 
    -Gradient structure tensor: estimating camera motion from image sequences
    -Multi-scale LoG saliency: highlights small targets so the system cna pay attention to them
    -Blur metric: measure how sharp/blurry a frame is and accept or rejct frames before further processing.)
It is required to be explicit(you can see exactly what they are doing), auditable(easy to inspect, verify and debug) and fast on the hardware that flies

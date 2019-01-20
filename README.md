# About
The following repository is a simple implementation for Neural Network using the C programming language.

# How to use
The Neural Network functions are stated in the header file `ann.h`, each network is associated with a structure defined:

```
typedef struct networks{
    int n_layers;
    int dim[LAYER_SIZE];
    double weights[LAYER_SIZE][MAX_SIZE][MAX_SIZE];
    double biases[LAYER_SIZE][MAX_SIZE];
}network;

```
### Macros

`MAX_SIZE` - The Maximum possible size for the number of neurons in a single layer ( this is used to determine the maximum possible dimensions of weights and biases)

`LAYER_SIZE` - Maximum possible number of layers.

Note - Setting high values for `MAX_SIZE` and `LAYER_SIZE` might take up too much memory and your program might be unable to run.



### Initializing
It is advisable to initialize the `network structure` and allocate memory for it in the heap using `malloc` function
```
#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

network* ann = (network *)malloc(sizeof(network));
```

Intializing 2D and 3D arrays in the heap can be done using the `alloc.h` header provided in this repository

```
#include <stdio.h>
#include <stdlib.h>
#include "ann.h"
#include "alloc.h"

double **train_data = init_2Darray(3,2); //Equivalent to double train_data[3][2]; but allocated in the heap 
double **test_data = init_3Darray(3,4); //Equivalent to double test_data[3][4]; but allocated in the heap

```


### Functions

`init_ann(network *,int dimensions[],no_of_layers) ` - Automatically Initializes the neural network and assigns initial (random) values for weights and biases based on the network structure defined in the `dimensions` array.

`init_ann_with_weights(network *,int dimensions[],double weights[][][], double biases[][],no_of_layers)` - Initializes neural network with weights and bias input as an argument.

`train(network *,double **data,length,learning_rate)` - Trains the network using gradient descent back-propogation algorithm and data provided.

`test(network *,double **data,length)` - Tests the network based on the test data input and shows accuracy.

`predict(network* , double *data)` - Gives output prediction for a particular test case stated in the data input.


### Examples

Initializing a neural network with 3 layers ( 2 neurons in input, 2 neurons in the hidden layers, 1 neuron in the output layer ):

```

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ann.h"

int main(){
  
    network *ann = (network*)malloc(sizeof(network));
    int dim[3] = {2,2,1}; //Dimension array
    init_ann(ann,dim,3); // Automatically assigns random values for weights and biases and initializes network 
    
    return 0;
}

```

1. Neural network for predicting XOR - [XOR network](https://github.com/gkrishnan724/ANN-IN-C/blob/master/XOR/XorNetwork.c)

2. Neural network for classifying handwritten digits based on kaggle dataset - [digit classifier](https://github.com/gkrishnan724/ANN-IN-C/tree/master/Digit%20Classification)

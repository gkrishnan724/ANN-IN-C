#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define MAX_SIZE 100000

//Declarations
typedef struct network{
    int n_layers;
    int dim[MAX_SIZE];
    double weigths[MAX_SIZE][MAX_SIZE][MAX_SIZE];
    int biases[MAX_SIZE];
}network;


//Function prototypes

double sigmoid(double x);
network init_ann(int[] ,int);
network init_ann_with_weights(int[], int, int[MAX_SIZE][MAX_SIZE][MAX_SIZE], int[]);
double* feed_forward(network*,int[MAX_SIZE][MAX_SIZE]);
void train(network*, int[MAX_SIZE][MAX_SIZE]);
void test(network*,int[MAX_SIZE][MAX_SIZE]);


//Definitions

network init_ann(int dim[],int n_layers){
    network ann;
    ann.n_layers = n_layers;
    arrayCopy(ann.dim,dim,n_layers);
    for(int i=1;i<n_layers;i++){
        for(int j=0;j<dim[i];j++){
            for(int k=0;k<dim[i-1];k++){
                ann.weigths[i-1][j][k] = rand()%10;
            }
        }
    }
    for(int i=1;i<n_layers;i++){
        ann.biases[i-1] = rand()%10;
    }
    return ann;

}

network init_ann_with_weights(int dim[],int weights[MAX_SIZE][MAX_SIZE][MAX_SIZE],int biases[MAX_SIZE],int n_layers){
    network ann;
    arrayCopy(ann.dim,dim,n_layers);
    for(int i=1;i<n_layers;i++){
        for(int j=0;j<dim[i];j++){
            for(int k=0;k<dim[i-1];k++){
                ann.weigths[i-1][j][k] = weights[i-1][j][k];
            }
        }
    }
    for(int i=1;i<n_layers;i++){
        ann.biases[i-1] = biases[i-1];
    }
    return ann;
}

void train(network* ann,double data[MAX_SIZE][MAX_SIZE],int length){
    double* output;
    for(int t=0;t<length;t++){
        // feed forward pass
        output = feed_forward(ann,data[t]);

        //computing cost and gradient descent
    }
}

void test(network* ann,int data[]){
    
}



double* feed_forward(network* ann,double data[]){
    double* input;
    double output[MAX_SIZE];
    for(int i=0;i<ann->n_layers-1;i++){
        if(i==0){
            input = data;
        }
        else{
            input = output;
        }
        for(int j=0;j<ann->dim[i+1];j++){
            double f_sum = 0;
            for(int k=0;k<ann->dim[i];k++){
                f_sum += ann->weigths[i][j][k] * input[k];
            }
            f_sum += 
            output[j] = sigmoid(f_sum);
        }
    }
    return output;
}

double sigmoid(double x){
    return 1/ (1 + exp(x)); 
}





//Misc Functions

void arrayCopy(int dest[],int source[],int length);

void arrayCopy(int dest[],int source[],int length){
    for(int i=0;i<length;i++){
        dest[i] = source[i];
    }
}




#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define MAX_SIZE 1000

//Declarations ANN structure
typedef struct networks{
    int n_layers;
    int dim[MAX_SIZE];
    double weights[MAX_SIZE][MAX_SIZE][MAX_SIZE];
    double biases[MAX_SIZE][MAX_SIZE];
}network;


//Function prototypes

double sigmoid(double x);
void init_ann(network*,int[] ,int);
void init_ann_with_weights(network*,int[],double [MAX_SIZE][MAX_SIZE][MAX_SIZE], double[MAX_SIZE][MAX_SIZE],int);
void feed_forward(network*,double output[MAX_SIZE][MAX_SIZE]);
void train(network*, double [MAX_SIZE][MAX_SIZE],int,double);
void test(network*,double[MAX_SIZE][MAX_SIZE],int);

//Misc Functions
void arrayCopy(double dest[],double source[],int length);

//Definitions

void init_ann(network* ann,int dim[],int n_layers){
    
    time_t t;
    /* Intializes random number generator */
    srand((unsigned)time(&t));
    ann->n_layers = n_layers;
    
    for(int i=0;i<n_layers;i++){
        ann->dim[i] = dim[i];
    }
    for(int i=1;i<n_layers;i++){
        for(int j=0;j<dim[i];j++){
            for(int k=0;k<dim[i-1];k++){
                ann->weights[i-1][j][k] = ((double)rand()/(double)RAND_MAX)*(1/sqrt(dim[i-1]+dim[i]));
            }
        }
    }
    for(int i=0;i<n_layers;i++){
        int val = 1;
        for(int j=0;j<ann->dim[i];j++){
            if(i == 0){
                ann->biases[i][j] = 0;
            }
            else{
                ann->biases[i][j] = val;
            }
        }
        
    }
    
}

void init_ann_with_weights(network* ann,int dim[],double weights[MAX_SIZE][MAX_SIZE][MAX_SIZE],double biases[MAX_SIZE][MAX_SIZE],int n_layers){
    
    ann->n_layers = n_layers;
    for(int i=0;i<n_layers;i++){
        ann->dim[i] = dim[i];
    }
    for(int i=1;i<n_layers;i++){
        for(int j=0;j<dim[i];j++){
            for(int k=0;k<dim[i-1];k++){
                ann->weights[i-1][j][k] = weights[i-1][j][k];
            }
        }
    }
    for(int i=0;i<n_layers;i++){
        for(int j=0;j<ann->dim[i];j++){
            if(i == 0){
                ann->biases[i][j] = 0;
            }
            else{
                ann->biases[i][j] = biases[i][j];
            }
        }
    }
}


void train(network* ann,double data[MAX_SIZE][MAX_SIZE],int length,double learning_rate){
    
    for(int t=0;t<length;t++){
        double output[ann->n_layers][MAX_SIZE];
        //Set first layer output as input data
        arrayCopy(output[0],data[t],ann->dim[0]);
        // feed forward pass to determine values in all nodes.
        feed_forward(ann,output);

        /*
            Updating weights for each and every input data.
        */

        //delta values
        double d[ann->n_layers][MAX_SIZE];
        
        //last layer delta computation
        for(int i=0;i<ann->dim[ann->n_layers-1];i++){
            double expected_value = (i == data[t][ann->dim[0]])?1:0;
            double observed_value = output[ann->n_layers - 1][i];
            d[ann->n_layers-1][i] = observed_value*(1-observed_value)*(observed_value - expected_value);
        }
        //Hidden layers delta computation
        for(int i=ann->n_layers-2;i>=0;i--){
            for(int j=0;j<ann->dim[i];j++){
                double fsum = 0;
                for(int k=0;k<ann->dim[i+1];k++){
                    fsum += d[i+1][k] * ann->weights[i][k][j];
                }
                d[i][j] = output[i][j]*(1-output[i][j])*fsum;
            }
        }

        //Updating weigths and biases

        for(int i=ann->n_layers-2;i>=0;i--){
            for(int j=0;j<ann->dim[i+1];j++){
                for(int k=0;k<ann->dim[i];k++){
                    ann->weights[i][j][k] -= learning_rate*output[i][k]*d[i+1][j];
                    
                }
                ann->biases[i+1][j] -= learning_rate*d[i+1][j];
            }
        }

        

    }
}

void test(network* ann,double data[MAX_SIZE][MAX_SIZE],int length){
    int correct = 0;
    int incorrect = 0;
    for(int t=0;t<length;t++){
        double output[ann->n_layers][MAX_SIZE];
        arrayCopy(output[0],data[t],ann->dim[0]);
        feed_forward(ann,output);
        int maxval = 0;
        for(int i=0;i<ann->dim[ann->n_layers-1];i++){
            if(output[ann->n_layers-1][i] > output[ann->n_layers - 1][maxval]){
                maxval = i;
            }
        }

        printf("Object %d is classified as %d\n",data[t][ann->dim[0]],maxval);
        if(maxval == data[t][ann->dim[0]]){
            correct += 1;
        }
        else{
            incorrect += 1;
        }
    }
    double accuracy = correct/length;
    printf("correct: %d, incorrect: %d, accuracy: %d",correct,incorrect,accuracy);
}



void feed_forward(network* ann,double output[MAX_SIZE][MAX_SIZE]){
    double* input; 
    for(int i=1;i<ann->n_layers-1;i++){
        input = output[i-1];
        for(int j=0;j<ann->dim[i+1];j++){
            double f_sum = 0;
            for(int k=0;k<ann->dim[i];k++){
                f_sum += ann->weights[i-1][j][k] * input[k];
            }
            f_sum += ann->biases[i][j];
            output[i][j] = sigmoid(f_sum);
        }
    }
}

double sigmoid(double x){
    return 1/ (1 + exp(x)); 
}





//Misc Functions


void arrayCopy(double dest[],double source[],int length){
    for(int i=0;i<length;i++){
        dest[i] = source[i];
    }
}




#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../ann.h"

int main(){
  
    network *ann = (network*)malloc(sizeof(network));
    int dim[3] = {2,4,1};
    init_ann(ann,dim,3);

    //Check Weights
    // for(int i=1;i<3;i++){
    //     for(int j=0;j<dim[i-1];j++){
    //         for(int k=0;k<dim[i];k++){
    //             printf("%lf ",ann->weights[i-1][k][j]);
    //         }
    //         printf("\n");
    //     }
    // }
    double train_data[MAX_SIZE][MAX_SIZE];
    train_data[0][0] = 0;
    train_data[0][1] = 0;
    train_data[0][2] = 0;
    train_data[1][0] = 0;
    train_data[1][1] = 1;
    train_data[1][2] = 1;
    train_data[2][0] = 1;
    train_data[2][1] = 0;
    train_data[2][2] = 1;
    train_data[3][0] = 1;
    train_data[3][1] = 1;
    train_data[3][2] = 0;
    for(int i=0;i<100000;i++){
        train(ann,train_data,4,0.25);
    }
    test(ann,train_data,4);

    //Check Weights
    // for(int i=1;i<3;i++){
    //     for(int j=0;j<dim[i-1];j++){
    //         for(int k=0;k<dim[i];k++){
    //             printf("%lf ",ann->weights[i-1][k][j]);
    //         }
    //         printf("\n");
    //     }
    // }
    
    return 0;
}
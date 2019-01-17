#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../alloc.h"
#include "../ann.h"

void train_from_csv(char* filename,char* buffer,double** train_data,int length);
void predict_from_csv(char *sourceFile,char* destFile,char* buffer, int length);
network* ann;

int main(){
    ann = (network*)malloc(sizeof(network));
    double **train_data = init_2Darray(MAX_SIZE,MAX_SIZE);
    int *dim = (int*)malloc(3*sizeof(int));
    // 3 Layer neural network 784*32*10
    dim[0] = 784;
    dim[1] = 32;
    dim[2] = 10;
    init_ann(ann,dim,3);
    char* buffer = (char*)malloc(10000*sizeof(char));
    train_from_csv("train.csv",buffer,train_data,2000000000);
    predict_from_csv("test.csv","submission.csv",buffer,2000000000);
}

void train_from_csv(char* filename,char* buffer,double** train_data,int length){
    FILE *fptr;
    if((fptr = fopen(filename,"r")) == NULL){
        printf("Unable to open file %s\n",filename);
        exit(1);
    }
    int count = 0;
    char *line;
    fgets(buffer,length,fptr);
    while(fgets(buffer,length,fptr)){
        line = strtok(buffer,",");        
        train_data[count][784] = atof(line);
        // printf("%f\n",train_data[count][784]);
        for(int i=0;i<784;i++){
            line = strtok(NULL,",");
            train_data[count][i] = atof(line)/255;
            // printf("%f ",train_data[count][i]);
        }
        // printf("\n");
        count = (count+1)%MAX_SIZE;
        if(count == 0){
            train(ann,train_data,MAX_SIZE,0.25);
            printf("Trained 1000 samples..\n");
        }

        
    }
    train(ann,train_data,count,0.25);
    printf("Done Reading..\n");

}

void predict_from_csv(char *sourceFile,char* destFile,char* buffer, int length){
    FILE *fptr;
    FILE *dest;
    if((fptr = fopen(sourceFile,"r")) == NULL){
        printf("Unable to open file %s\n",sourceFile);
        exit(1);
    }
    if((dest = fopen(destFile,"w")) == NULL){
        printf("Unable to open file %s\n",destFile);
        exit(1);
    }
    char *line;
    int count = 0;
    double data[MAX_SIZE];
    fprintf(dest,"ImageId,Label\n");
    fgets(buffer,length,fptr);
    int actual;
    while(fgets(buffer,length,fptr)){
        for(int i=0;i<784;i++){
            if(i == 0){
                line = strtok(buffer,",");
                data[i] = atof(line)/255;
            }
            else{
                line = strtok(NULL,",");
                data[i] = atof(line)/255;
            }
            // printf("%f ",data[i-1]);
        }
        int result = predict(ann,data);
        fprintf(dest,"%d,%d\n",count+1,result);
        count+=1;
    }
}
#include "benchmark.h"
#include <stdio.h>
__global__ 
void multGPU(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < pmatrix->rows)
    {
        pvector_out->data[i] = 0;
        for(int j = 0; j < pmatrix->collums; j++)
        {
            pvector_out->data[i] += pmatrix->data[(pmatrix->collums * i) + j] * pvector_in->data[j];            
        }
    }   
}

void initalisation(vector_t* &input, matrix_t* &matrix, vector_t* &output);
void clearMemory(vector_t* &input, matrix_t* &matrix, vector_t* &output);

void singleCpuThreadExecution(vector_t* &input, matrix_t* &matrix, vector_t* &output);
void multiCpuThreadExecution(vector_t* &input, matrix_t* &matrix, vector_t* &output);
void gpuExceution(vector_t* &input, matrix_t* &matrix, vector_t* &output);

int main()
{    
    matrix_t* matrix;
    vector_t* input;
    vector_t* output;

    initalisation(input, matrix, output);

    gpuExceution(input, matrix, output);
    multiCpuThreadExecution(input, matrix, output);  
    singleCpuThreadExecution(input, matrix, output);

    clearMemory(input, matrix, output);
     
    return 0;
}

void initalisation(vector_t* &input, matrix_t* &matrix, vector_t* &output)
{
    matrix = initMatrixOnCPU(OUTPUT_LENGTH, INPUT_LENGTH);
    input = initVectorOnCPU(INPUT_LENGTH);
    output = initVectorOnCPU(OUTPUT_LENGTH);

    setMatrixValues(matrix);
    setInputValues(input);
}

void clearMemory(vector_t* &input, matrix_t* &matrix, vector_t* &output)
{
    deleteMatrixOnCPU(matrix);
    deleteVectorOnCPU(input);
    deleteVectorOnCPU(output);   
}

void singleCpuThreadExecution(vector_t* &input, matrix_t* &matrix, vector_t* &output)
{
    clock_t start = clock();
    multSingleThreadCPU(input, matrix, output);
    clock_t fin = clock();

    printf("%.3Fms singel Thread CPU time\n",1000. * (double)(fin - start) / (double)CLOCKS_PER_SEC);
}

void multiCpuThreadExecution(vector_t* &input, matrix_t* &matrix, vector_t* &output)
{
    clock_t start = clock();
    multCPU(input, matrix, output);
    clock_t fin = clock();

    printf("%.3Fms CPU time with %d threads\n", 1000. * (double)(fin - start) / (double)CLOCKS_PER_SEC, THREADS);
}

void gpuExceution(vector_t* &input, matrix_t* &matrix, vector_t* &output)
{
    matrix = moveMatrixToGPU(matrix);
    input = moveVectorToGPU(input);
    output = moveVectorToGPU(output);

    clock_t start = clock();
    multGPU<<<1, OUTPUT_LENGTH>>>(input, matrix, output);
    clock_t fin = clock();
    
    matrix = moveMatrixToCPU(matrix);
    input = moveVectorToCPU(input);
    output = moveVectorToCPU(output);

    printf("%.3Fms GPU time\n",1000. * (double)(fin - start) / (double)CLOCKS_PER_SEC);
}
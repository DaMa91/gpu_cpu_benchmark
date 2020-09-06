#include"benchmark.h"

#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<pthread.h>

void deleteVectorOnCPU(vector_t* vector)
{
    free(vector->data);
    free(vector);
}

void deleteMatrixOnCPU(matrix_t* matrix)
{
    free(matrix->data);
    free(matrix);
}

vector_t* initVectorOnCPU(int collums)
{
    vector_t* vector = (vector_t*)malloc(sizeof(vector_t));
    vector->collums = collums;
    vector->data = (float*) malloc(collums * sizeof(float));
    return vector;
}

matrix_t* initMatrixOnCPU(int rows, int collums)
{
    matrix_t* matrix = (matrix_t*) malloc(sizeof(matrix_t));
    matrix->rows = rows;
    matrix->collums = collums;
    matrix->dataLength = rows * collums;
    matrix->data = (float*) malloc(matrix->dataLength * sizeof(float));
    return matrix;
}

void setMatrixValues(matrix_t* matrix)
{
    for(int i = 0; i < matrix->dataLength; i++)
    {
        matrix->data[i] = /*(float)matrix->dataLength / (float)(i + 1)*/ 1.f;
    }
}

void setInputValues(vector_t* input)
{
    for(int i = 0; i < input->collums; i++)
    {
        input->data[i] = /*(float) i*/ 1.f;
    }
}

vector_t* moveVectorToGPU(vector_t* cpuVector)
{
    vector_t* gpuVector;
    float* gpuData;

    cudaMalloc(&gpuVector, sizeof(vector_t)); 
    cudaMalloc(&gpuData, cpuVector->collums * sizeof(float));

    cudaMemcpy(gpuData, cpuVector->data, cpuVector->collums * sizeof(float), cudaMemcpyHostToDevice);    
    free(cpuVector->data);

    cpuVector->data = gpuData;

    cudaMemcpy(gpuVector, cpuVector, sizeof(vector_t), cudaMemcpyHostToDevice);
    free(cpuVector);
    
    return gpuVector;
}

vector_t* moveVectorToCPU(vector_t* gpuVector)
{
    vector_t* cpuVector = (vector_t*) malloc(sizeof(vector_t));
    cudaMemcpy(cpuVector, gpuVector, sizeof(vector_t), cudaMemcpyDeviceToHost);
    
    float* cpuData = (float*) malloc(cpuVector->collums * sizeof(float));
    cudaMemcpy(cpuData, cpuVector->data, cpuVector->collums * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(cpuVector->data);
    cudaFree(gpuVector);

    cpuVector->data = cpuData;

    return cpuVector;
}

matrix_t* moveMatrixToGPU(matrix_t* cpuMatrix)
{
    matrix_t* gpuMatrix;
    float* gpuData;

    cudaMalloc(&gpuMatrix, sizeof(matrix_t)); 
    cudaMalloc(&gpuData, cpuMatrix->dataLength * sizeof(float));

    cudaMemcpy(gpuData, cpuMatrix->data, cpuMatrix->dataLength * sizeof(float), cudaMemcpyHostToDevice);    
    free(cpuMatrix->data);

    cpuMatrix->data = gpuData;

    cudaMemcpy(gpuMatrix, cpuMatrix, sizeof(matrix_t), cudaMemcpyHostToDevice);
    free(cpuMatrix);
    
    return gpuMatrix;
}

matrix_t* moveMatrixToCPU(matrix_t* gpuMatrix)
{
    matrix_t* cpuMatrix = (matrix_t*) malloc(sizeof(matrix_t));
    cudaMemcpy(cpuMatrix, gpuMatrix, sizeof(matrix_t), cudaMemcpyDeviceToHost);
    
    float* cpuData = (float*) malloc(cpuMatrix->dataLength * sizeof(float));
    cudaMemcpy(cpuData, cpuMatrix->data, cpuMatrix->dataLength * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(cpuMatrix->data);
    cudaFree(gpuMatrix);

    cpuMatrix->data = cpuData;

    return cpuMatrix;
}

void printCPUVector(vector_t* vector)
{
    printf("[");
    for(int i = 0; i < vector->collums; i++)
    {
        printf(" %f", vector->data[i]);
    }
    printf(" ]");
}    

void printCPUMatrix(matrix_t* matrix)
{
    printf("[\n");
    int collums = matrix->collums;
    for(int i = 0; i < matrix->rows; i++)
    {
        printf("\t[");
        for(int j = 0; j < collums; j++)
        {
            printf(" %f", matrix->data[(i * collums) + j]);
        }
        printf(" ]\n");
    }
    printf("]");
}

static void* multCPUThread(void* _params)
{
    thread_params_t params = *(thread_params_t*)_params;
    vector_t* pvector_in = params.pvector_in;
    matrix_t* pmatrix = params.pmatrix;
    vector_t* pvector_out = params.pvector_out;
    multThread(pvector_in, pmatrix, pvector_out, params.id * THREAD_RANGE);
    return 0;
}

void multThread(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out, int start)
{
    int collums = pmatrix->collums;
    for(int i = start; i < pmatrix->rows && i < (start + THREAD_RANGE); i++)
    {
        pvector_out->data[i] = 0;
        for(int j = 0; j < collums; j++)
        {
            pvector_out->data[i] += pmatrix->data[(i * collums) + j] * pvector_in->data[j];
        }
    }
}

void multCPU(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out)
{
    volatile thread_params_t params[16];
    pthread_t threads[15];
    for(int i = 0; i < 16; i++)
    {
        params[i].id = (uint8_t) i;
        params[i].pvector_in = pvector_in;
        params[i].pmatrix = pmatrix;
        params[i].pvector_out = pvector_out;
        
        if(i < 15)
        {
            pthread_create(&threads[i], NULL, multCPUThread,(void*) &params[i]);
        }
    }
    
    multCPUThread((void*) &params[15]);
    for(int i = 0; i < 15; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

void multSingleThreadCPU(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out)
{
    int collums = pmatrix->collums;
    for(int i = 0; i < pmatrix->rows; i++)
    {
        pvector_out->data[i] = 0;
        for(int j = 0; j < collums; j++)
        {
            pvector_out->data[i] += pmatrix->data[(i * collums) + j] * pvector_in->data[j];
        }
    }
}

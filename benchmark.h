#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>

#define INPUT_LENGTH 10000
#define OUTPUT_LENGTH 10000
#define THREADS 16
#define THREAD_RANGE (INPUT_LENGTH / THREADS)

typedef struct 
{
    int collums;
    int rows;
    float* data;
    int dataLength;
} matrix_t; 

typedef struct 
{
    int collums;
    float *data;
} vector_t;

typedef struct 
{
    matrix_t* pmatrix;
    vector_t* pvector_in;
    vector_t* pvector_out;
    uint8_t id;
}thread_params_t;

vector_t* initVectorOnCPU(int collums);
matrix_t* initMatrixOnCPU(int rows, int collums);

void setMatrixValues(matrix_t* matrix);
void setInputValues(vector_t* input);

void deleteVectorOnCPU(vector_t* vector);
void deleteMatrixOnCPU(matrix_t* matrix);

vector_t* moveVectorToGPU(vector_t* cpuVector);
vector_t* moveVectorToCPU(vector_t* gpuVector);
matrix_t* moveMatrixToGPU(matrix_t* cpuMatrix);
matrix_t* moveMatrixToCPU(matrix_t* gpuMatrix);

void printCPUVector(vector_t* vector);
void printCPUMatrix(matrix_t* matrix);

void multThread(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out, int start);
void multCPU(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out);

void multSingleThreadCPU(vector_t* pvector_in, matrix_t* pmatrix, vector_t* pvector_out);

#endif
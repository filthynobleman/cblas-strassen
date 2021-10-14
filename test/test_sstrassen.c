/**
 * @file        test_sstrassen.c
 * 
 * @brief       Testing the function cblas_sstrassen() with variable options' configurations.
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-10-12
 */
#include <cblas_strassen.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>




int main(int argc, char const *argv[])
{
    CS_INT M = 1024;
    CS_INT N = 1024;
    CS_INT K = 1024;
    CS_INT Seed = 0;

    if (argc >= 4)
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        Seed = atoi(argv[4]);
    }


    clock_t Start, End;
    float Error;
    CS_INT i;

    printf("Allocation of memory... ");
    Start = clock();
    float* A = (float*)malloc(M * K * sizeof(float));
    if (A == NULL)
    {
        fprintf(stderr, "\nCannot allocate A.\n");
        exit(-1);
    }
    float* B = (float*)malloc(K * N * sizeof(float));
    if (B == NULL)
    {
        fprintf(stderr, "\nCannot allocate B.\n");
        exit(-1);
    }
    float* C = (float*)malloc(M * N * sizeof(float));
    if (C == NULL)
    {
        fprintf(stderr, "\nCannot allocate C.\n");
        exit(-1);
    }
    size_t SuppElmnts = ((M + 1) * (N + 1) + (M + 1) * (K + 1) + (K + 1) * (N + 1)) / 2;
    float* Support = (float*)malloc(SuppElmnts * sizeof(float));
    if (Support == NULL)
    {
        fprintf(stderr, "\nCannot allocate support memory.\n");
        exit(-1);
    }
    float* wM = Support;
    float* wLH = wM + (M + 1) * (N + 1) / 2;
    float* wRH = wLH + (M + 1) * (K + 1) / 2;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);



    printf("Filling matrices... ");
    srand(Seed);
    Start = clock();
    for (i = 0; i < M * K; ++i)
        A[i] = rand() / (float)RAND_MAX;
    for (i = 0; i < K * N; ++i)
        B[i] = rand() / (float)RAND_MAX;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);



    printf("Product A * B.\n");
    Start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    End = clock();
    printf("CBLAS_SGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_sstrassen(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0f, A, K, B, N, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_SSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_snrm2(M * N, C, 1));



    printf("Product A' * B.\n");
    Start = clock();
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, N, 0.0f, C, N);
    End = clock();
    printf("CBLAS_SGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_sstrassen(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, -1.0f, A, M, B, N, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_SSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_snrm2(M * N, C, 1));



    printf("Product A * B'.\n");
    Start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
    End = clock();
    printf("CBLAS_SGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_sstrassen(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, -1.0f, A, K, B, K, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_SSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_snrm2(M * N, C, 1));



    printf("Product A' * B'.\n");
    Start = clock();
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, N);
    End = clock();
    printf("CBLAS_SGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_sstrassen(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, -1.0f, A, M, B, K, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_SSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_snrm2(M * N, C, 1));



    printf("Product A * B (Column major).\n");
    Start = clock();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
    End = clock();
    printf("CBLAS_SGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_sstrassen(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0f, A, M, B, K, 1.0f, C, M, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_SSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_snrm2(M * N, C, 1));


    
    return 0;
}
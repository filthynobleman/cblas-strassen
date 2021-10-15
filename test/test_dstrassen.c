/**
 * @file        test_dstrassen.c
 * 
 * @brief       Testing the function cblas_dstrassen() with variable options' configurations.
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
    MKL_INT M = 1024;
    MKL_INT N = 1024;
    MKL_INT K = 1024;
    MKL_INT Seed = 0;

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
    double Error;
    MKL_INT i;

    printf("Allocation of memory... ");
    Start = clock();
    double* A = (double*)malloc(M * K * sizeof(double));
    if (A == NULL)
    {
        fprintf(stderr, "\nCannot allocate A.\n");
        exit(-1);
    }
    double* B = (double*)malloc(K * N * sizeof(double));
    if (B == NULL)
    {
        fprintf(stderr, "\nCannot allocate B.\n");
        exit(-1);
    }
    double* C = (double*)malloc(M * N * sizeof(double));
    if (C == NULL)
    {
        fprintf(stderr, "\nCannot allocate C.\n");
        exit(-1);
    }
    size_t SuppElmnts = ((M + 1) * (N + 1) + (M + 1) * (K + 1) + (K + 1) * (N + 1)) / 2;
    double* Support = (double*)malloc(SuppElmnts * sizeof(double));
    if (Support == NULL)
    {
        fprintf(stderr, "\nCannot allocate support memory.\n");
        exit(-1);
    }
    double* wM = Support;
    double* wLH = wM + (M + 1) * (N + 1) / 2;
    double* wRH = wLH + (M + 1) * (K + 1) / 2;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);



    printf("Filling matrices... ");
    srand(Seed);
    Start = clock();
    for (i = 0; i < M * K; ++i)
        A[i] = rand() / (double)RAND_MAX;
    for (i = 0; i < K * N; ++i)
        B[i] = rand() / (double)RAND_MAX;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);


    printf("Product A * B.\n");
    Start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    End = clock();
    printf("CBLAS_DGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_dstrassen(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0f, A, K, B, N, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_DSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dnrm2(M * N, C, 1));



    printf("Product A' * B.\n");
    Start = clock();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, N, 0.0f, C, N);
    End = clock();
    printf("CBLAS_DGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_dstrassen(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, -1.0f, A, M, B, N, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_DSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dnrm2(M * N, C, 1));



    printf("Product A * B'.\n");
    Start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
    End = clock();
    printf("CBLAS_DGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_dstrassen(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, -1.0f, A, K, B, K, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_DSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dnrm2(M * N, C, 1));



    printf("Product A' * B'.\n");
    Start = clock();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, N);
    End = clock();
    printf("CBLAS_DGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_dstrassen(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, -1.0f, A, M, B, K, 1.0f, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_DSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dnrm2(M * N, C, 1));



    printf("Product A * B (Column major).\n");
    Start = clock();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
    End = clock();
    printf("CBLAS_DGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_dstrassen(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0f, A, M, B, K, 1.0f, C, M, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_DSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dnrm2(M * N, C, 1));


    
    return 0;
}
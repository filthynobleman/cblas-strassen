/**
 * @file        test_dstrassen.c
 * 
 * @brief       Testing the function cblas_zstrassen() with variable options' configurations.
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
#include <complex.h>




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


    _Dcomplex Zero      = _Cbuild(0.0f, 0.0f);
    _Dcomplex One       = _Cbuild(1.0f, 0.0f);
    _Dcomplex NegOne    = _Cbuild(-1.0f, 0.0);
    clock_t Start, End;
    CS_INT i;

    printf("Allocation of memory... ");
    Start = clock();
    _Dcomplex* A = (_Dcomplex*)malloc(M * K * sizeof(_Dcomplex));
    if (A == NULL)
    {
        fprintf(stderr, "\nCannot allocate A.\n");
        exit(-1);
    }
    _Dcomplex* B = (_Dcomplex*)malloc(K * N * sizeof(_Dcomplex));
    if (B == NULL)
    {
        fprintf(stderr, "\nCannot allocate B.\n");
        exit(-1);
    }
    _Dcomplex* C = (_Dcomplex*)malloc(M * N * sizeof(_Dcomplex));
    if (C == NULL)
    {
        fprintf(stderr, "\nCannot allocate C.\n");
        exit(-1);
    }
    size_t SuppElmnts = ((M + 1) * (N + 1) + (M + 1) * (K + 1) + (K + 1) * (N + 1)) / 2;
    _Dcomplex* Support = (_Dcomplex*)malloc(SuppElmnts * sizeof(_Dcomplex));
    if (Support == NULL)
    {
        fprintf(stderr, "\nCannot allocate support memory.\n");
        exit(-1);
    }
    _Dcomplex* wM = Support;
    _Dcomplex* wLH = wM + (M + 1) * (N + 1) / 2;
    _Dcomplex* wRH = wLH + (M + 1) * (K + 1) / 2;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);



    printf("Filling matrices... ");
    srand(Seed);
    Start = clock();
    for (i = 0; i < M * K; ++i)
        A[i] = _Cbuild(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
    for (i = 0; i < K * N; ++i)
        B[i] = _Cbuild(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);


    printf("Product A * B.\n");
    Start = clock();
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &One, A, K, B, N, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_zstrassen(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &NegOne, A, K, B, N, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dznrm2(M * N, C, 1));



    printf("Product A' * B.\n");
    Start = clock();
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, M, N, K, &One, A, M, B, N, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_zstrassen(CblasRowMajor, CblasConjTrans, CblasNoTrans, M, N, K, &NegOne, A, M, B, N, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dznrm2(M * N, C, 1));



    printf("Product A * B'.\n");
    Start = clock();
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K, &One, A, K, B, K, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_zstrassen(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K, &NegOne, A, K, B, K, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dznrm2(M * N, C, 1));



    printf("Product A' * B'.\n");
    Start = clock();
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, N, K, &One, A, M, B, K, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_zstrassen(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, N, K, &NegOne, A, M, B, K, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dznrm2(M * N, C, 1));



    printf("Product A * B (Column major).\n");
    Start = clock();
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &One, A, M, B, K, &Zero, C, M);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_zstrassen(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &NegOne, A, M, B, K, &One, C, M, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_dznrm2(M * N, C, 1));


    
    return 0;
}
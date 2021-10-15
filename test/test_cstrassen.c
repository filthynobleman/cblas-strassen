/**
 * @file        test_dstrassen.c
 * 
 * @brief       Testing the function cblas_cstrassen() with variable options' configurations.
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


    _Fcomplex Zero      = _FCbuild(0.0f, 0.0f);
    _Fcomplex One       = _FCbuild(1.0f, 0.0f);
    _Fcomplex NegOne    = _FCbuild(-1.0f, 0.0);
    clock_t Start, End;
    MKL_INT i;

    printf("Allocation of memory... ");
    Start = clock();
    _Fcomplex* A = (_Fcomplex*)malloc(M * K * sizeof(_Fcomplex));
    if (A == NULL)
    {
        fprintf(stderr, "\nCannot allocate A.\n");
        exit(-1);
    }
    _Fcomplex* B = (_Fcomplex*)malloc(K * N * sizeof(_Fcomplex));
    if (B == NULL)
    {
        fprintf(stderr, "\nCannot allocate B.\n");
        exit(-1);
    }
    _Fcomplex* C = (_Fcomplex*)malloc(M * N * sizeof(_Fcomplex));
    if (C == NULL)
    {
        fprintf(stderr, "\nCannot allocate C.\n");
        exit(-1);
    }
    size_t SuppElmnts = ((M + 1) * (N + 1) + (M + 1) * (K + 1) + (K + 1) * (N + 1)) / 2;
    _Fcomplex* Support = (_Fcomplex*)malloc(SuppElmnts * sizeof(_Fcomplex));
    if (Support == NULL)
    {
        fprintf(stderr, "\nCannot allocate support memory.\n");
        exit(-1);
    }
    _Fcomplex* wM = Support;
    _Fcomplex* wLH = wM + (M + 1) * (N + 1) / 2;
    _Fcomplex* wRH = wLH + (M + 1) * (K + 1) / 2;
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);



    printf("Filling matrices... ");
    srand(Seed);
    Start = clock();
    for (i = 0; i < M * K; ++i)
        A[i] = _FCbuild(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
    for (i = 0; i < K * N; ++i)
        B[i] = _FCbuild(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
    End = clock();
    printf("Elapsed time is %.3f.\n\n\n", (End - Start) / (double)CLOCKS_PER_SEC);


    printf("Product A * B.\n");
    Start = clock();
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &One, A, K, B, N, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_cstrassen(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &NegOne, A, K, B, N, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_scnrm2(M * N, C, 1));



    printf("Product A' * B.\n");
    Start = clock();
    cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, M, N, K, &One, A, M, B, N, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_cstrassen(CblasRowMajor, CblasConjTrans, CblasNoTrans, M, N, K, &NegOne, A, M, B, N, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_scnrm2(M * N, C, 1));



    printf("Product A * B'.\n");
    Start = clock();
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K, &One, A, K, B, K, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_cstrassen(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K, &NegOne, A, K, B, K, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_scnrm2(M * N, C, 1));



    printf("Product A' * B'.\n");
    Start = clock();
    cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, N, K, &One, A, M, B, K, &Zero, C, N);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_cstrassen(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, N, K, &NegOne, A, M, B, K, &One, C, N, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_scnrm2(M * N, C, 1));



    printf("Product A * B (Column major).\n");
    Start = clock();
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &One, A, M, B, K, &Zero, C, M);
    End = clock();
    printf("CBLAS_CGEMM took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);
    Start = clock();
    cblas_cstrassen(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &NegOne, A, M, B, K, &One, C, M, wLH, wRH, wM);
    End = clock();
    printf("CBLAS_CSTRASSEN took %.3f s.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    printf("Error is %.3e.\n\n", cblas_scnrm2(M * N, C, 1));


    
    return 0;
}
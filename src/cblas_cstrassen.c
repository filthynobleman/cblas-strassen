/**
 * @file        cblas_cstrassen.c
 * 
 * @brief       Implementation of single floating point precision complex Strassen's algorithm.
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-10-12
 */
#include <cblas_strassen.h>
#include <complex.h>

#define BASE_CASE 2048


void cblas_cstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                    CS_INT M, CS_INT N, CS_INT K,
                    const void* alpha, const void* A, CS_INT ldA, const void* B, CS_INT ldB,
                    const void* beta, void* C, CS_INT ldC,
                    void* wlh, void* wrh, void* wm)
{
    // If layout is column major, convert to row major
    if (lo == CblasColMajor)
    {
        cblas_cstrassen(CblasRowMajor, transB, transA, N, M, K, alpha, B, ldB, A, ldA, beta, C, ldC, wrh, wlh, wm);
        return;
    }

    // Check base case
    if (M <= BASE_CASE || N <= BASE_CASE || K <= BASE_CASE)
    {
        cblas_cgemm(CblasRowMajor, transA, transB, 
                    M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }


    // Get sizes of submatrices
    CS_INT M2 = (M + 1) / 2;
    CS_INT N2 = (N + 1) / 2;
    CS_INT K2 = (K + 1) / 2;


    // Compute the working memory to use in recursive calls
    _Fcomplex* wslh = ((_Fcomplex*)wlh) + M2 * K2;
    _Fcomplex* wsrh = ((_Fcomplex*)wrh) + K2 * N2;
    _Fcomplex* wsm = ((_Fcomplex*)wm) + M2 * N2;
    CS_INT i;
    CS_INT ldLH;
    CS_INT ldRH;
    CS_INT ldM;


    // Compute the submatrices of A, B and C
    const _Fcomplex* A11 = ((_Fcomplex*)A);                 // M2 X K2
    const _Fcomplex* A12 = ((_Fcomplex*)A) + K2;            // M2 X (K - K2)
    const _Fcomplex* A21 = ((_Fcomplex*)A) + M2 * ldA;      // (M - M2) X K2
    const _Fcomplex* A22 = A21 + K2;                        // (M - M2) X (K - K2)
    if (transA == CblasTrans || transA == CblasConjTrans)
    {
        A21 = ((_Fcomplex*)A) + M2;
        A12 = ((_Fcomplex*)A) + K2 * ldA;
        A22 = A12 + M2;
    }
    const _Fcomplex* B11 = ((_Fcomplex*)B);                 // K2 X N2
    const _Fcomplex* B12 = ((_Fcomplex*)B) + N2;            // K2 X (N - N2)
    const _Fcomplex* B21 = ((_Fcomplex*)B) + K2 * ldB;      // (K - K2) X N2
    const _Fcomplex* B22 = B21 + N2;                            // (K - K2) X (N - N2)
    if (transB == CblasTrans || transB == CblasConjTrans)
    {
        B21 = ((_Fcomplex*)B) + K2;
        B12 = ((_Fcomplex*)B) + N2 * ldB;
        B22 = B12 + K2;
    }
    _Fcomplex* C11 = ((_Fcomplex*)C);                       // M2 X N2
    _Fcomplex* C12 = ((_Fcomplex*)C) + N2;                  // M2 X (N - N2)
    _Fcomplex* C21 = ((_Fcomplex*)C) + M2 * ldC;            // (M - M2) X N2
    _Fcomplex* C22 = C21 + N2;                              // (M - M2) X (N - N2)


    _Fcomplex Zero      = _FCbuild(0.0f, 0.0f);
    _Fcomplex One       = _FCbuild(1.0f, 0.0f);
    _Fcomplex NegOne    = _FCbuild(-1.0f, 0.0);
    _Fcomplex NegAlpha  = _FCbuild(-crealf(*((_Fcomplex*)alpha)), 
                                   -cimagf(*((_Fcomplex*)alpha)));


    // M1 = (A11 + A22) * (B11 + B22)
    // L = A11 + A22
    if (transA == CblasNoTrans)
    {
        // Sum rows
        for (i = 0; i < M - M2; ++i)
        {
            cblas_ccopy(K2, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
            cblas_caxpy(K - K2, &One, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
        }
        if (M2 > M - M2)
            cblas_ccopy(K2, A11 + (M2 - 1) * ldA, 1, ((_Fcomplex*)wlh) + (M2 - 1) * K2, 1);
        ldLH = K2;
    }
    else
    {
        // Sum columns of A'
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(M2, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
            cblas_caxpy(M - M2, &One, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
        }
        if (K2 > K - K2)
            cblas_ccopy(M2, A11 + (K2 - 1) * ldA, 1, ((_Fcomplex*)wlh) + (K2 - 1) * M2, 1);
        ldLH = M2;
    }
    // R = B11 + B22
    if (transB == CblasNoTrans)
    {
        // Sum rows ob B
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(N2, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
            cblas_caxpy(N - N2, &One, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
        }
        if (K2  > K - K2)
            cblas_ccopy(N2, B11 + (K2 - 1) * ldB, 1, ((_Fcomplex*)wrh) + (K2 - 1) * N2, 1);
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_ccopy(K2, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
            cblas_caxpy(K - K2, &One, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
        }
        if (N2  > N - N2)
            cblas_ccopy(K2, B11 + (N2 - 1) * ldB, 1, ((_Fcomplex*)wrh) + (N2 - 1) * K2, 1);
        ldRH = K2;
    }
    // M1 = L * R
    cblas_cstrassen(CblasRowMajor, transA, transB, 
                    M2, N2, K2, &One, ((_Fcomplex*)wlh), ldLH, ((_Fcomplex*)wrh), ldRH, &Zero, ((_Fcomplex*)wm), N2,
                    wslh, wsrh, wsm);
    // C11 = beta * C11 + alpha * M1, C22 = beta * C22 + alpha * M1
    for (i = 0; i < M - M2; ++i)
    {
        cblas_caxpby(N2, alpha, ((_Fcomplex*)wm) + i * N2, 1, beta, C11 + i * ldC, 1);
        cblas_caxpby(N - N2, alpha, ((_Fcomplex*)wm) + i * N2, 1, beta, C22 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_caxpby(N2, alpha, ((_Fcomplex*)wm) + (M2 - 1) * N2, 1, beta, C11 + (M2 - 1) * ldC, 1);

    
    // M2 = (A21 + A22) * B11
    // L = A21 + A22
    if (transA == CblasNoTrans)
    {
        // Sum rows of A
        for (i = 0; i < M - M2; ++i)
        {
            cblas_ccopy(K2, A21 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
            cblas_caxpy(K - K2, &One, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
        }
        ldLH = K2;
    }
    else
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(M - M2, A21 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
            cblas_caxpy(M - M2, &One, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
        }
        if (K2 > K - K2)
            cblas_ccopy(M - M2, A21 + (K2 - 1) * ldA, 1, ((_Fcomplex*)wlh) + (K2 - 1) * M2, 1);
        ldLH = M - M2;
    }
    // M2 = L * B11
    cblas_cstrassen(CblasRowMajor, transA, transB,
                    M - M2, N2, K2, &One, ((_Fcomplex*)wlh), ldLH, B11, ldB, &Zero, ((_Fcomplex*)wm), N2,
                    wslh, wsrh, wsm);
    // C21 = beta * C21 + alpha * M2, C22 -= alpha * M2
    for (i = 0; i < M - M2; ++i)
    {
        cblas_caxpby(N2, alpha, ((_Fcomplex*)wm) + i * N2, 1, beta, C21 + i * ldC, 1);
        cblas_caxpy(N - N2, &NegAlpha, ((_Fcomplex*)wm) + i * N2, 1, C22 + i * ldC, 1);
    }


    // M3 = A11 * (B12 - B22)
    // R = B12 - B22
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(N - N2, B12 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (N - N2), 1);
            cblas_caxpy(N - N2, &NegOne, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (N - N2), 1);
        }
        if (K2 > K - K2)
            cblas_ccopy(N - N2, B12 + (K2 - 1) * ldB, 1, ((_Fcomplex*)wrh) + (K2 - 1) * (N - N2), 1);
        ldRH = N - N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_ccopy(K2, B12 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
            cblas_caxpy(K - K2, &NegOne, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
        }
        ldRH = K2;
    }
    // M3 = A11 * R
    cblas_cstrassen(CblasRowMajor, transA, transB, 
                    M2, N - N2, K2, &One, A11, ldA, ((_Fcomplex*)wrh), ldRH, &Zero, ((_Fcomplex*)wm), N - N2,
                    wslh, wsrh, wsm);
    //C12 = beta * C12 + alpha * M3, C22 += alpha * M3
    for (i = 0; i < M - M2; ++i)
    {
        cblas_caxpby(N - N2, alpha, ((_Fcomplex*)wm) + i * (N - N2), 1, beta, C12 + i * ldC, 1);
        cblas_caxpy(N - N2, alpha, ((_Fcomplex*)wm) + i * (N - N2), 1, C22 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_caxpby(N - N2, alpha, ((_Fcomplex*)wm) + (M2 - 1) * (N - N2), 1, beta, C12 + i * ldC, 1);


    // M4 = A22 * (B21 - B11)
    // R = B21 - B11
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(N2, B21 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
            cblas_caxpy(N2, &NegOne, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
        }
        // A22 has K - K2 rows, so an eventual last odd row is ignored
        // if (K2 > K - K2)
        //     cblas_caxpby(N2, &One, B11 + (K2 - 1) * ldB, 1, &Zero, ((_Fcomplex*)wrh) + (K2 - 1) * N2, 1);
        ldRH = N2;
    }
    else
    {
        // A22 has K - K2 rows, so an eventual last odd row is ignored
        for (i = 0; i < N2; ++i)
        {
            cblas_ccopy(K - K2, B21 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (K - K2), 1);
            cblas_caxpy(K - K2, &NegOne, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (K - K2), 1);
        }
        ldRH = K - K2;
    }
    // M4 = A22 * R
    cblas_cstrassen(CblasRowMajor, transA, transB, 
                    M - M2, N2, K - K2, &One, A22, ldA, ((_Fcomplex*)wrh), ldRH, &Zero, ((_Fcomplex*)wm), N2,
                    wslh, wsrh, wsm);
    // C11 += alpha * M4, C21 += alpha * M4
    for (i = 0; i < M - M2; ++i)
    {
        cblas_caxpy(N2, alpha, ((_Fcomplex*)wm) + i * N2, 1, C11 + i * ldC, 1);
        cblas_caxpy(N2, alpha, ((_Fcomplex*)wm) + i * N2, 1, C21 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_caxpy(N2, alpha, ((_Fcomplex*)wm) + (M2 - 1) * N2, 1, C11 + (M2 - 1) * N2, 1);
    

    // M5 = (A11 + A12) * B22
    // L = (A11 + A12)
    if (transA == CblasNoTrans)
    {
        // B22 has K - K2 rows, last column of A can be safely ignored
        for (i = 0; i < M2; ++i)
        {
            cblas_ccopy(K - K2, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * (K - K2), 1);
            cblas_caxpy(K - K2, &One, A12 + i * ldA, 1, ((_Fcomplex*)wlh) + i * (K - K2), 1);
        }
        ldLH = K - K2;
    }
    else
    {
        // B22 has K - K2 rows, last column of A can be safely ignored
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(M2, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
            cblas_caxpy(M2, &One, A12 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
        }
        ldLH = M2;
    }
    // M5 = L * B22
    cblas_cstrassen(CblasRowMajor, transA, transB, 
                    M2, N - N2, K - K2, &One, ((_Fcomplex*)wlh), ldLH, B22, ldB, &Zero, ((_Fcomplex*)wm), N - N2,
                    wslh, wsrh, wsm);
    // C11 -= alpha * M5, C12 += alpha * M5
    for (i = 0; i < M2; ++i)
    {
        cblas_caxpy(N - N2, &NegAlpha, ((_Fcomplex*)wm) + i * (N - N2), 1, C11 + i * ldC, 1);
        cblas_caxpy(N - N2, alpha, ((_Fcomplex*)wm) + i * (N - N2), 1, C12 + i * ldC, 1);
    }


    // M6 = (A21 - A11) * (B11 + B12)
    // L = A21 - A11
    if (transA == CblasNoTrans)
    {
        for (i = 0; i < M - M2; ++i)
        {
            cblas_ccopy(K2, A21 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
            cblas_caxpy(K2, &NegOne, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * K2, 1);
        }
        if (M2 > M - M2)
            cblas_caxpby(K2, &NegOne, A11 + (M2 - 1) * ldA, 1, &Zero, ((_Fcomplex*)wlh) + (M2 - 1) * K2, 1);
        ldLH = K2;
    }
    else
    {
        for (i = 0; i < K2; ++i)
        {
            cblas_ccopy(M2, A11 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
            cblas_caxpby(M - M2, &One, A21 + i * ldA, 1, &NegOne, ((_Fcomplex*)wlh) + i * M2, 1);
        }
        ldLH = M2;
    }
    // R = B11 + B12
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K2; ++i)
        {
            cblas_ccopy(N2, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
            cblas_caxpy(N - N2, &One, B12 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
        }
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_ccopy(K2, B11 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
            cblas_caxpy(K2, &One, B12 + i * ldB, 1, ((_Fcomplex*)wrh) + i * K2, 1);
        }
        if (N2 > N - N2)
            cblas_ccopy(K2, B11 + (N2 - 1) * ldB, 1, ((_Fcomplex*)wrh) + (N2 - 1) * K2, 1);
        ldRH = K2;
    }
    // M6 = L * R
    // C22 += alpha * M6
    cblas_cstrassen(CblasRowMajor, transA, transB,
                    M - M2, N - N2, K2, alpha, ((_Fcomplex*)wlh), ldLH, ((_Fcomplex*)wrh), ldRH, &One, C22, ldC,
                    wslh, wsrh, wsm);


    // M7 = (A12 - A22) * (B21 + B22)
    // L = A12 - A22
    if (transA == CblasNoTrans)
    {
        for (i = 0; i < M - M2; ++i)
        {
            cblas_ccopy(K - K2, A12 + i * ldA, 1, ((_Fcomplex*)wlh) + i * (K - K2), 1);
            cblas_caxpy(K - K2, &NegOne, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * (K - K2), 1);
        }
        if (M2 > M - M2)
            cblas_ccopy(K - K2, A12 + (M2 - 1) * ldA, 1, ((_Fcomplex*)wlh) + (M2 - 1) * (K - K2), 1);
        ldLH = K - K2;
    }
    else
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(M2, A12 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
            cblas_caxpy(M - M2, &NegOne, A22 + i * ldA, 1, ((_Fcomplex*)wlh) + i * M2, 1);
        }
        ldLH = M2;
    }
    // R = B21 + B22
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_ccopy(N2, B21 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
            cblas_caxpy(N - N2, &One, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * N2, 1);
        }
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_ccopy(K - K2, B21 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (K - K2), 1);
            cblas_caxpy(K - K2, &One, B22 + i * ldB, 1, ((_Fcomplex*)wrh) + i * (K - K2), 1);
        }
        if (N2 > N - N2)
            cblas_ccopy(K - K2, B21 + (N2 - 1) * ldB, 1, ((_Fcomplex*)wrh) + (N2 - 1) * (K - K2), 1);
        ldRH = K - K2;
    }
    // M7 = L * R
    // C11 += alpha * M7
    cblas_cstrassen(CblasRowMajor, transA, transB,
                    M2, N2, K - K2, alpha, ((_Fcomplex*)wlh), ldLH, ((_Fcomplex*)((_Fcomplex*)wrh)), ldRH, &One, C11, ldC,
                    wslh, wsrh, wsm);
}
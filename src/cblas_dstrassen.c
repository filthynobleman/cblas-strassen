/**
 * @file        cblas_dstrassen.c
 * 
 * @brief       Implementation of double doubleing point precision Strassen's algorithm.
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-10-12
 */
#include <cblas_strassen.h>


#define BASE_CASE 1024


void cblas_dstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     double alpha, const double* A, MKL_INT ldA, const double* B, MKL_INT ldB,
                     double beta, double* C, MKL_INT ldC,
                     double* wlh, double* wrh, double* wm)
{
    // If layout is column major, convert to row major
    if (lo == CblasColMajor)
    {
        cblas_dstrassen(CblasRowMajor, transB, transA, N, M, K, alpha, B, ldB, A, ldA, beta, C, ldC, wrh, wlh, wm);
        return;
    }

    // Check base case
    if (M <= BASE_CASE || N <= BASE_CASE || K <= BASE_CASE)
    {
        cblas_dgemm(CblasRowMajor, transA, transB, 
                    M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }


    // Get sizes of submatrices
    MKL_INT M2 = (M + 1) / 2;
    MKL_INT N2 = (N + 1) / 2;
    MKL_INT K2 = (K + 1) / 2;


    // Compute the working memory to use in recursive calls
    double* wslh = wlh + M2 * K2;
    double* wsrh = wrh + K2 * N2;
    double* wsm = wm + M2 * N2;
    MKL_INT i;
    MKL_INT ldLH;
    MKL_INT ldRH;
    MKL_INT ldM;


    // Compute the submatrices of A, B and C
    const double* A11 = A;                   // M2 X K2
    const double* A12 = A + K2;              // M2 X (K - K2)
    const double* A21 = A + M2 * ldA;        // (M - M2) X K2
    const double* A22 = A21 + K2;            // (M - M2) X (K - K2)
    if (transA == CblasTrans || transA == CblasConjTrans)
    {
        A21 = A + M2;
        A12 = A + K2 * ldA;
        A22 = A12 + M2;
    }
    const double* B11 = B;                   // K2 X N2
    const double* B12 = B + N2;              // K2 X (N - N2)
    const double* B21 = B + K2 * ldB;        // (K - K2) X N2
    const double* B22 = B21 + N2;            // (K - K2) X (N - N2)
    if (transB == CblasTrans || transB == CblasConjTrans)
    {
        B21 = B + K2;
        B12 = B + N2 * ldB;
        B22 = B12 + K2;
    }
    double* C11 = C;                         // M2 X N2
    double* C12 = C + N2;                    // M2 X (N - N2)
    double* C21 = C + M2 * ldC;              // (M - M2) X N2
    double* C22 = C21 + N2;                  // (M - M2) X (N - N2)


    // M1 = (A11 + A22) * (B11 + B22)
    // L = A11 + A22
    if (transA == CblasNoTrans)
    {
        // Sum rows
        for (i = 0; i < M - M2; ++i)
        {
            cblas_dcopy(K2, A11 + i * ldA, 1, wlh + i * K2, 1);
            cblas_daxpy(K - K2, 1.0f, A22 + i * ldA, 1, wlh + i * K2, 1);
        }
        if (M2 > M - M2)
            cblas_dcopy(K2, A11 + (M2 - 1) * ldA, 1, wlh + (M2 - 1) * K2, 1);
        ldLH = K2;
    }
    else
    {
        // Sum columns of A'
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(M2, A11 + i * ldA, 1, wlh + i * M2, 1);
            cblas_daxpy(M - M2, 1.0f, A22 + i * ldA, 1, wlh + i * M2, 1);
        }
        if (K2 > K - K2)
            cblas_dcopy(M2, A11 + (K2 - 1) * ldA, 1, wlh + (K2 - 1) * M2, 1);
        ldLH = M2;
    }
    // R = B11 + B22
    if (transB == CblasNoTrans)
    {
        // Sum rows ob B
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(N2, B11 + i * ldB, 1, wrh + i * N2, 1);
            cblas_daxpy(N - N2, 1.0f, B22 + i * ldB, 1, wrh + i * N2, 1);
        }
        if (K2  > K - K2)
            cblas_dcopy(N2, B11 + (K2 - 1) * ldB, 1, wrh + (K2 - 1) * N2, 1);
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_dcopy(K2, B11 + i * ldB, 1, wrh + i * K2, 1);
            cblas_daxpy(K - K2, 1.0f, B22 + i * ldB, 1, wrh + i * K2, 1);
        }
        if (N2  > N - N2)
            cblas_dcopy(K2, B11 + (N2 - 1) * ldB, 1, wrh + (N2 - 1) * K2, 1);
        ldRH = K2;
    }
    // M1 = L * R
    cblas_dstrassen(CblasRowMajor, transA, transB, 
                    M2, N2, K2, 1.0f, wlh, ldLH, wrh, ldRH, 0.0f, wm, N2,
                    wslh, wsrh, wsm);
    // C11 = beta * C11 + alpha * M1, C22 = beta * C22 + alpha * M1
    for (i = 0; i < M - M2; ++i)
    {
        cblas_daxpby(N2, alpha, wm + i * N2, 1, beta, C11 + i * ldC, 1);
        cblas_daxpby(N - N2, alpha, wm + i * N2, 1, beta, C22 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_daxpby(N2, alpha, wm + (M2 - 1) * N2, 1, beta, C11 + (M2 - 1) * ldC, 1);

    
    // M2 = (A21 + A22) * B11
    // L = A21 + A22
    if (transA == CblasNoTrans)
    {
        // Sum rows of A
        for (i = 0; i < M - M2; ++i)
        {
            cblas_dcopy(K2, A21 + i * ldA, 1, wlh + i * K2, 1);
            cblas_daxpy(K - K2, 1.0f, A22 + i * ldA, 1, wlh + i * K2, 1);
        }
        ldLH = K2;
    }
    else
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(M - M2, A21 + i * ldA, 1, wlh + i * M2, 1);
            cblas_daxpy(M - M2, 1.0f, A22 + i * ldA, 1, wlh + i * M2, 1);
        }
        if (K2 > K - K2)
            cblas_dcopy(M - M2, A21 + (K2 - 1) * ldA, 1, wlh + (K2 - 1) * M2, 1);
        ldLH = M - M2;
    }
    // M2 = L * B11
    cblas_dstrassen(CblasRowMajor, transA, transB,
                    M - M2, N2, K2, 1.0f, wlh, ldLH, B11, ldB, 0.0f, wm, N2,
                    wslh, wsrh, wsm);
    // C21 = beta * C21 + alpha * M2, C22 -= alpha * M2
    for (i = 0; i < M - M2; ++i)
    {
        cblas_daxpby(N2, alpha, wm + i * N2, 1, beta, C21 + i * ldC, 1);
        cblas_daxpy(N - N2, -alpha, wm + i * N2, 1, C22 + i * ldC, 1);
    }


    // M3 = A11 * (B12 - B22)
    // R = B12 - B22
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(N - N2, B12 + i * ldB, 1, wrh + i * (N - N2), 1);
            cblas_daxpy(N - N2, -1.0f, B22 + i * ldB, 1, wrh + i * (N - N2), 1);
        }
        if (K2 > K - K2)
            cblas_dcopy(N - N2, B12 + (K2 - 1) * ldB, 1, wrh + (K2 - 1) * (N - N2), 1);
        ldRH = N - N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_dcopy(K2, B12 + i * ldB, 1, wrh + i * K2, 1);
            cblas_daxpy(K - K2, -1.0f, B22 + i * ldB, 1, wrh + i * K2, 1);
        }
        ldRH = K2;
    }
    // M3 = A11 * R
    cblas_dstrassen(CblasRowMajor, transA, transB, 
                    M2, N - N2, K2, 1.0f, A11, ldA, wrh, ldRH, 0.0f, wm, N - N2,
                    wslh, wsrh, wsm);
    //C12 = beta * C12 + alpha * M3, C22 += alpha * M3
    for (i = 0; i < M - M2; ++i)
    {
        cblas_daxpby(N - N2, alpha, wm + i * (N - N2), 1, beta, C12 + i * ldC, 1);
        cblas_daxpy(N - N2, alpha, wm + i * (N - N2), 1, C22 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_daxpby(N - N2, alpha, wm + (M2 - 1) * (N - N2), 1, beta, C12 + i * ldC, 1);


    // M4 = A22 * (B21 - B11)
    // R = B21 - B11
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(N2, B21 + i * ldB, 1, wrh + i * N2, 1);
            cblas_daxpy(N2, -1.0f, B11 + i * ldB, 1, wrh + i * N2, 1);
        }
        // A22 has K - K2 rows, so an eventual last odd row is ignored
        // if (K2 > K - K2)
        //     cblas_daxpby(N2, 1.0f, B11 + (K2 - 1) * ldB, 1, 0.0f, wrh + (K2 - 1) * N2, 1);
        ldRH = N2;
    }
    else
    {
        // A22 has K - K2 rows, so an eventual last odd row is ignored
        for (i = 0; i < N2; ++i)
        {
            cblas_dcopy(K - K2, B21 + i * ldB, 1, wrh + i * (K - K2), 1);
            cblas_daxpy(K - K2, -1.0f, B11 + i * ldB, 1, wrh + i * (K - K2), 1);
        }
        ldRH = K - K2;
    }
    // M4 = A22 * R
    cblas_dstrassen(CblasRowMajor, transA, transB, 
                    M - M2, N2, K - K2, 1.0f, A22, ldA, wrh, ldRH, 0.0f, wm, N2,
                    wslh, wsrh, wsm);
    // C11 += alpha * M4, C21 += alpha * M4
    for (i = 0; i < M - M2; ++i)
    {
        cblas_daxpy(N2, alpha, wm + i * N2, 1, C11 + i * ldC, 1);
        cblas_daxpy(N2, alpha, wm + i * N2, 1, C21 + i * ldC, 1);
    }
    if (M2 > M - M2)
        cblas_daxpy(N2, alpha, wm + (M2 - 1) * N2, 1, C11 + (M2 - 1) * N2, 1);
    

    // M5 = (A11 + A12) * B22
    // L = (A11 + A12)
    if (transA == CblasNoTrans)
    {
        // B22 has K - K2 rows, last column of A can be safely ignored
        for (i = 0; i < M2; ++i)
        {
            cblas_dcopy(K - K2, A11 + i * ldA, 1, wlh + i * (K - K2), 1);
            cblas_daxpy(K - K2, 1.0f, A12 + i * ldA, 1, wlh + i * (K - K2), 1);
        }
        ldLH = K - K2;
    }
    else
    {
        // B22 has K - K2 rows, last column of A can be safely ignored
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(M2, A11 + i * ldA, 1, wlh + i * M2, 1);
            cblas_daxpy(M2, 1.0f, A12 + i * ldA, 1, wlh + i * M2, 1);
        }
        ldLH = M2;
    }
    // M5 = L * B22
    cblas_dstrassen(CblasRowMajor, transA, transB, 
                    M2, N - N2, K - K2, 1.0f, wlh, ldLH, B22, ldB, 0.0f, wm, N - N2,
                    wslh, wsrh, wsm);
    // C11 -= alpha * M5, C12 += alpha * M5
    for (i = 0; i < M2; ++i)
    {
        cblas_daxpy(N - N2, -alpha, wm + i * (N - N2), 1, C11 + i * ldC, 1);
        cblas_daxpy(N - N2, alpha, wm + i * (N - N2), 1, C12 + i * ldC, 1);
    }


    // M6 = (A21 - A11) * (B11 + B12)
    // L = A21 - A11
    if (transA == CblasNoTrans)
    {
        for (i = 0; i < M - M2; ++i)
        {
            cblas_dcopy(K2, A21 + i * ldA, 1, wlh + i * K2, 1);
            cblas_daxpy(K2, -1.0f, A11 + i * ldA, 1, wlh + i * K2, 1);
        }
        if (M2 > M - M2)
            cblas_daxpby(K2, -1.0f, A11 + (M2 - 1) * ldA, 1, 0.0f, wlh + (M2 - 1) * K2, 1);
        ldLH = K2;
    }
    else
    {
        for (i = 0; i < K2; ++i)
        {
            cblas_dcopy(M2, A11 + i * ldA, 1, wlh + i * M2, 1);
            cblas_daxpby(M - M2, 1.0f, A21 + i * ldA, 1, -1.0f, wlh + i * M2, 1);
        }
        ldLH = M2;
    }
    // R = B11 + B12
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K2; ++i)
        {
            cblas_dcopy(N2, B11 + i * ldB, 1, wrh + i * N2, 1);
            cblas_daxpy(N - N2, 1.0f, B12 + i * ldB, 1, wrh + i * N2, 1);
        }
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_dcopy(K2, B11 + i * ldB, 1, wrh + i * K2, 1);
            cblas_daxpy(K2, 1.0f, B12 + i * ldB, 1, wrh + i * K2, 1);
        }
        if (N2 > N - N2)
            cblas_dcopy(K2, B11 + (N2 - 1) * ldB, 1, wrh + (N2 - 1) * K2, 1);
        ldRH = K2;
    }
    // M6 = L * R
    // C22 += alpha * M6
    cblas_dstrassen(CblasRowMajor, transA, transB,
                    M - M2, N - N2, K2, alpha, wlh, ldLH, wrh, ldRH, 1.0f, C22, ldC,
                    wslh, wsrh, wsm);


    // M7 = (A12 - A22) * (B21 + B22)
    // L = A12 - A22
    if (transA == CblasNoTrans)
    {
        for (i = 0; i < M - M2; ++i)
        {
            cblas_dcopy(K - K2, A12 + i * ldA, 1, wlh + i * (K - K2), 1);
            cblas_daxpy(K - K2, -1.0f, A22 + i * ldA, 1, wlh + i * (K - K2), 1);
        }
        if (M2 > M - M2)
            cblas_dcopy(K - K2, A12 + (M2 - 1) * ldA, 1, wlh + (M2 - 1) * (K - K2), 1);
        ldLH = K - K2;
    }
    else
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(M2, A12 + i * ldA, 1, wlh + i * M2, 1);
            cblas_daxpy(M - M2, -1.0f, A22 + i * ldA, 1, wlh + i * M2, 1);
        }
        ldLH = M2;
    }
    // R = B21 + B22
    if (transB == CblasNoTrans)
    {
        for (i = 0; i < K - K2; ++i)
        {
            cblas_dcopy(N2, B21 + i * ldB, 1, wrh + i * N2, 1);
            cblas_daxpy(N - N2, 1.0f, B22 + i * ldB, 1, wrh + i * N2, 1);
        }
        ldRH = N2;
    }
    else
    {
        for (i = 0; i < N - N2; ++i)
        {
            cblas_dcopy(K - K2, B21 + i * ldB, 1, wrh + i * (K - K2), 1);
            cblas_daxpy(K - K2, 1.0f, B22 + i * ldB, 1, wrh + i * (K - K2), 1);
        }
        if (N2 > N - N2)
            cblas_dcopy(K - K2, B21 + (N2 - 1) * ldB, 1, wrh + (N2 - 1) * (K - K2), 1);
        ldRH = K - K2;
    }
    // M7 = L * R
    // C11 += alpha * M7
    cblas_dstrassen(CblasRowMajor, transA, transB,
                    M2, N2, K - K2, alpha, wlh, ldLH, wrh, ldRH, 1.0f, C11, ldC,
                    wslh, wsrh, wsm);
}
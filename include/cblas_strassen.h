/**
 * @file        cblas_strassen.h
 * 
 * @brief       An implementation of Strassen's algorithm with a CBLAS style interface.
 * 
 * @details     This file contains the signature of a function implementing the Strassen's
 *              algorithm for general matrix multiplication.\n 
 *              The signature follows the typical CBLAS style.
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-10-12
 */
#ifndef CBLAS_STRASSEN_H_
#define CBLAS_STRASSEN_H_

#include <mkl_cblas.h>

/**
 * @brief       Single precision floating point matrix multiplication with Strassen's algorithm.
 * 
 * @details     The function provides an implementation of the single precision floating point matrix
 *              multiplication using the Strassen's algorithm. The function computes
 *              C = beta * C + alpha * op(A) * op(B)
 * 
 * @param lo        Matrix layout. Either CblasRowMajor or CblasColMajor.
 * @param transA    Whether or not the matrix A must be transposed.
 * @param transB    Whether or not the matrix B must be transposed.
 * @param M         Number of rows of matrix op(A).
 * @param N         Number of columns of matrix op(B).
 * @param K         Number of columns of matrix op(A) and rows of op(B).
 * @param alpha     Scaling factor for the multiplication's result.
 * @param A         The first matrix to multiply.
 * @param ldA       The leading dimension of A.
 * @param B         The second matrix to multiply.
 * @param ldB       The leading dimension of B.
 * @param beta      Premultiply factor for result's destination.
 * @param C         The destination of the multiplication's result.
 * @param ldC       The leading dimension of C.
 * @param wlh       Working space. Must contain at least (M * K) / 2 elements.
 * @param wrh       Working space. Must contain at least (K * N) / 2 elements.
 * @param wm        Working space. Must contain at least (M * N) / 2 elements.
 */
void cblas_sstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     float alpha, const float* A, MKL_INT ldA, const float* B, MKL_INT ldB,
                     float beta, float* C, MKL_INT ldC,
                     float* wlh, float* wrh, float* wm);


/**
 * @brief       Double precision floating point matrix multiplication with Strassen's algorithm.
 * 
 * @details     The function provides an implementation of the double precision floating point matrix
 *              multiplication using the Strassen's algorithm. The function computes
 *              C = beta * C + alpha * op(A) * op(B)
 * 
 * @param lo        Matrix layout. Either CblasRowMajor or CblasColMajor.
 * @param transA    Whether or not the matrix A must be transposed.
 * @param transB    Whether or not the matrix B must be transposed.
 * @param M         Number of rows of matrix op(A).
 * @param N         Number of columns of matrix op(B).
 * @param K         Number of columns of matrix op(A) and rows of op(B).
 * @param alpha     Scaling factor for the multiplication's result.
 * @param A         The first matrix to multiply.
 * @param ldA       The leading dimension of A.
 * @param B         The second matrix to multiply.
 * @param ldB       The leading dimension of B.
 * @param beta      Premultiply factor for result's destination.
 * @param C         The destination of the multiplication's result.
 * @param ldC       The leading dimension of C.
 * @param wlh       Working space. Must contain at least (M * K) / 2 elements.
 * @param wrh       Working space. Must contain at least (K * N) / 2 elements.
 * @param wm        Working space. Must contain at least (M * N) / 2 elements.
 */
void cblas_dstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     double alpha, const double* A, MKL_INT ldA, const double* B, MKL_INT ldB,
                     double beta, double* C, MKL_INT ldC,
                     double* wlh, double* wrh, double* wm);


/**
 * @brief       Single precision floating point complex matrix multiplication with Strassen's algorithm.
 * 
 * @details     The function provides an implementation of the single precision floating point complex matrix
 *              multiplication using the Strassen's algorithm. The function computes
 *              C = beta * C + alpha * op(A) * op(B)
 * 
 * @param lo        Matrix layout. Either CblasRowMajor or CblasColMajor.
 * @param transA    Whether or not the matrix A must be transposed or conjugate transposed.
 * @param transB    Whether or not the matrix B must be transposed or conjugate transposed.
 * @param M         Number of rows of matrix op(A).
 * @param N         Number of columns of matrix op(B).
 * @param K         Number of columns of matrix op(A) and rows of op(B).
 * @param alpha     Scaling factor for the multiplication's result.
 * @param A         The first matrix to multiply.
 * @param ldA       The leading dimension of A.
 * @param B         The second matrix to multiply.
 * @param ldB       The leading dimension of B.
 * @param beta      Premultiply factor for result's destination.
 * @param C         The destination of the multiplication's result.
 * @param ldC       The leading dimension of C.
 * @param wlh       Working space. Must contain at least (M * K) / 2 elements.
 * @param wrh       Working space. Must contain at least (K * N) / 2 elements.
 * @param wm        Working space. Must contain at least (M * N) / 2 elements.
 */
void cblas_cstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     const void* alpha, const void* A, MKL_INT ldA, const void* B, MKL_INT ldB,
                     const void* beta, void* C, MKL_INT ldC,
                     void* wlh, void* wrh, void* wm);

/**
 * @brief       Double precision floating point complex matrix multiplication with Strassen's algorithm.
 * 
 * @details     The function provides an implementation of the double precision floating point complex matrix
 *              multiplication using the Strassen's algorithm. The function computes
 *              C = beta * C + alpha * op(A) * op(B)
 * 
 * @param lo        Matrix layout. Either CblasRowMajor or CblasColMajor.
 * @param transA    Whether or not the matrix A must be transposed or conjugate transposed.
 * @param transB    Whether or not the matrix B must be transposed or conjugate transposed.
 * @param M         Number of rows of matrix op(A).
 * @param N         Number of columns of matrix op(B).
 * @param K         Number of columns of matrix op(A) and rows of op(B).
 * @param alpha     Scaling factor for the multiplication's result.
 * @param A         The first matrix to multiply.
 * @param ldA       The leading dimension of A.
 * @param B         The second matrix to multiply.
 * @param ldB       The leading dimension of B.
 * @param beta      Premultiply factor for result's destination.
 * @param C         The destination of the multiplication's result.
 * @param ldC       The leading dimension of C.
 * @param wlh       Working space. Must contain at least (M * K) / 2 elements.
 * @param wrh       Working space. Must contain at least (K * N) / 2 elements.
 * @param wm        Working space. Must contain at least (M * N) / 2 elements.
 */
void cblas_zstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     const void* alpha, const void* A, MKL_INT ldA, const void* B, MKL_INT ldB,
                     const void* beta, void* C, MKL_INT ldC,
                     void* wlh, void* wrh, void* wm);





#endif // CBLAS_STRASSEN_H_
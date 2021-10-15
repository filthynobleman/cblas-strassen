# CBLAS_STRASSEN
This repository contains an efficient implementation of the Strassen's algorithm.  
The algorithm exposes a CBLAS compatible interface and supports all the four standard datatypes: single and double floating point precision real and complex numbers.  

<img src="times.png" width="800px">

The algorithm improves the execution time for matrix multiplication at the cost of `3/2 N^2` additional memory.

## Installation
This section provides instruction for the installation of the library.

### Prerequisites
In order to compile and install the library a working installation of [CMake](https://cmake.org/) is needed, together with a compatible C compiler.  
In addition, the library relies on the BLAS implementation from Intel, provided in the [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.dqqfw4) library

### Compile and Install
In the repository folder, create a build directory and move inside
```bash
    mkdir build
    cd build
```
Configure the CMake project, build and install
```bash
    cmake ..
    cmake --build .
    cmake --install .
```

In the configuration you can add the definition of the variable `BUILD_SAMPLES` by using
```bash
    cmake .. -DBUILD_SAMPLES=ON
```
to enable the building of the samples application. A detailed description of these applications is provided in [the proper section](#sample-applications).

In the build phase, the default build type is `Debug`. You can build a `Release` version by typing
```bash
    cmake --build . --config Release
```
The `Release` version builds the libraries with optimization flags.

After the installation, in the installation folder defined by the variable `CMAKE_INSTALL_PREFIX` the following tree will be created (**notice that the name of the libraries is different under other operating systems.**):
```
INSTALL_DIR
|
|--- CBLAS_STRASSEN
    |
    |--- include
    |   |
    |   |---- cblas_strassen.h
    |
    |--- lib
        |
        |---- cblas_strassen.so
        |---- cblas_strassen_static.a
```



## Usage
To use the function `cblas_?strassen` you need first to include the header file.
```c
#include <cblas_strassen.h>
```
The signature of the function is the following, depending on the datatype:
```c
void cblas_sstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     float alpha, const float* A, MKL_INT ldA, const float* B, MKL_INT ldB,
                     float beta, float* C, MKL_INT ldC,
                     float* wlh, float* wrh, float* wm);
void cblas_dstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     double alpha, const double* A, MKL_INT ldA, const double* B, MKL_INT ldB,
                     double beta, double* C, MKL_INT ldC,
                     double* wlh, double* wrh, double* wm);
void cblas_cstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     const void* alpha, const void* A, MKL_INT ldA, const void* B, MKL_INT ldB,
                     const void* beta, void* C, MKL_INT ldC,
                     void* wlh, void* wrh, void* wm);
void cblas_zstrassen(CBLAS_LAYOUT lo, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT M, MKL_INT N, MKL_INT K,
                     const void* alpha, const void* A, MKL_INT ldA, const void* B, MKL_INT ldB,
                     const void* beta, void* C, MKL_INT ldC,
                     void* wlh, void* wrh, void* wm);
```
The function computes the matrix product `C = beta * C + alpha * op(A) * op(B)`, where `op(X)` can be `X`, the transpose or the conjugate transpose of `X`.  
Strassen's algorithm relies on intermediate sums to compute a matrix product by only using 7 recursive multiplications.  
  
The function's parameter are the following:
 - `lo` is the matrix layout. Can be either `CblasRowMajor` or `CblasColMajor`, respectively for row major and column major ordering.
 - `transA` and `transB` defines the transposition operations to apply to `A` and `B`. Can be `CblasNoTrans`, `CblasTrans` or `CblasConjTrans`, respectively for no transposition, transposition or conjugate transposition.
 - `M` is the number of rows of `op(A)` and of the resulting matrix `C`.
 - `N` is the number of columns of `op(B)` and of the resulting matrix `C`.
 - `K` is the number of columns of `op(A)` and the number of rows of `op(B)`.
 - `alpha` is the scaling factor of the product.
 - `A` and `B` are the input matrices.
 - `ldA` and `ldB` are the leading dimensions of, respectively, `A` and `B`.
 - `beta` is the scaling factor of `C` before summing the product.
 - `C` is the resulting matrix.
 - `ldC` is the leading dimension of `C`.
 - `wlh` is a support array that can store `(M + 1) * (K + 1) / 2` elements. Used to store the left factors of the intermediate products.
 - `wrh` is a support array that can store `(K + 1) * (N + 1) / 2` elements. Used to store the right factors of the intermediate products.
 - `wm` is a support array that can store `(M + 1) * (N + 1) / 2` elements. Used to store the intermediate products.

For more detailed informations on the parameters of a matrix multiplication sub-routine, consult the [Intel MKL documentation for `cblas_?gemm`](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html).


Here is a sample code for using the function to compute the double precision real product `C = A^T * B`:
```c
// Allocate the support memory
size_t num_entries = ((M + 1) * (N + 1) / 2) +
                     ((M + 1) * (K + 1) / 2) +
                     ((K + 1) * (N + 1) / 2);
double* wm = malloc(num_entries * sizeof(double));
double* wlh = wm + ((M + 1) * (N + 1) / 2);
double* wrh = wlh + ((M + 1) * (K + 1) / 2);
// Compute the product C = A^T * B
cblas_dstrassen(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                1.0, A, ldA, B, ldB,
                0.0, C, ldC,
                wlh, wrh, wm);
```
 

## Sample Applications
If you have build the sample applications, they should be found in the build directory.  
The applications named `test_?strassen` are simple applications that multiply two random matrices and all their combinations of transpositions, comparing the error and the time with the standard `cblas_?gemm`. The syntax to run them is
```
./test_?strassen M N K
```
where `M`, `N` and `K` are the dimensions of the matrices.

The application `test_times` runs a set of random multiplications of different sizes and produces a CSV file comparing the times of `cblas_?strassen` against `cblas_?gemm`. The syntax to run the application is the following:
```
./test_times [OPTIONS] MIN_SIZE MAX_SIZE STEP_SIZE
```
where
 - `MIN_SIZE` is the smallest matrix size to test
 - `MAX_SIZE` is the largest matrix size to test
 - `STEP_SIZE` is the size increment between two consecutive tests

The application accepts the following options, which must be provided before the mandatory arguments
 - `-o FILENAME` to specify the path to the CSV file to write. Default is `test_times.csv` in working directory.
 - `-a TA` and `-b TB` to specify if the matrices to multiply must be transposed or not. Admitted values are `N` for no transposition, `T` for transposition and `H` for conjugate transposition. Default is `N` for both.
 - `-d DATATYPE` to specify the multiplication datatype. Legal values are `S`, `D` (for single and double real precision), `C` or `Z` (for single and double complex precision). Default is `S`.
 - `-n NUM_REPS` to specify how many times each multiplication must be carried out for a reliable time average. Default is `10`.
 - `-h` to display a brief help message.
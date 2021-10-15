/**
 * @file        test_times.c
 * 
 * @brief       Test the execution times of cblas_strassen() against cblas_gemm().
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-10-14
 */
#include <cblas_strassen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <complex.h>



// Windows does not have getopt. To guarantee cross-platform support I use the AT&T open implementation
// https://opensource.apple.com/source/patch_cmds/patch_cmds-17/diffstat/porting/getopt.c.auto.html
#define ERR() if(opterr){return -2;}

int opterr = 1;
int optind = 1;
int optopt;
char *optarg;

int getopt(int argc, char * const *argv, const char *opts)
{
    static int sp = 1;
    register int c;
    register char *cp;

    if (sp == 1) 
    {
        if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0')
            return (EOF);
        else if (strcmp(argv[optind], "--") == 0)
        {
            optind++;
            return (EOF);
        }
    }
    optopt = c = argv[optind][sp];
    if (c == ':' || (cp = strchr(opts, c)) == NULL) 
    {
        if (argv[optind][++sp] == '\0') 
        {
            optind++;
            sp = 1;
        }
        return ('?');
    }
    if (*++cp == ':') 
    {
        if (argv[optind][sp + 1] != '\0')
            optarg = &argv[optind++][sp + 1];
        else if (++optind >= argc) 
        {
            sp = 1;
            return ('?');
        } 
        else
            optarg = argv[optind++];
    	sp = 1;
    } 
    else 
    {
        if (argv[optind][++sp] == '\0') 
        {
            sp = 1;
            optind++;
        }
        optarg = NULL;
    }
    return (c);
}




void usage(const char* argv0, int is_help)
{
    fprintf(stderr, "Usage: %s [ OPTIONS ] MIN_SIZE MAX_SIZE STEP\n", argv0);
    fprintf(stderr, "Computes the execution times of cblas_sstrassen against cblas_sgemm for different matrix sizes.\n");
    fprintf(stderr, "Each iteration is averaged on 10 repetitions. Results are stored in the CSV file test_times.csv.\n\n");
    fprintf(stderr, "       MIN_SIZE and MAX_SIZE are the minimum and maximum sizes for matrices.\n");
    fprintf(stderr, "       STEP is the difference in sizes between two iterations.\n");
    fprintf(stderr, "       [-o FILENAME] to save the result in FILENAME rather than test_times.csv.\n");
    fprintf(stderr, "       [-a TA] and [-b TB] to (conjugate-)transpose either the left or right factor. Can be N, T or H.\n");
    fprintf(stderr, "       [-d DATATYPE] to select the datatype. Can be S, D, C or Z.\n");
    fprintf(stderr, "       [-n NUM_REPS] to change the number of repetitions of each iteration.\n");
    fprintf(stderr, "       [-h] to print this message.\n\n");
    
    if (!is_help)
        exit(-1);
    else
        exit(0);
}



int main(int argc, char * const *argv)
{
    MKL_INT NMin;
    MKL_INT NMax;
    MKL_INT NStep;
    MKL_INT Steps;
    MKL_INT NReps = 10;
    const char* OutFile = "test_times.csv";
    CBLAS_TRANSPOSE TA = CblasNoTrans;
    CBLAS_TRANSPOSE TB = CblasNoTrans;
    char DataType = 'S';

    // Parse optional arguments
    int opt;
    while ((opt = getopt(argc, argv, "o:a:b:d:n:h")) != EOF)
    {
        switch (opt)
        {
        case 'h': usage(argv[0], 0 == 0);
        case 'o': OutFile = optarg; break;
        case 'a': 
        {
            switch (*optarg)
            {
            case 'n':
            case 'N':
                TA = CblasNoTrans;
                break;
            
            case 't':
            case 'T':
                TA = CblasTrans;
                break;
            
            case 'h':
            case 'H':
                TA = CblasConjTrans;
                break;
            
            default:
                fprintf(stderr, "%c is not a valid transposition option.\n", *optarg);
                exit(-1);
            }
            break;
        }
        case 'b': 
        {
            switch (*optarg)
            {
            case 'n':
            case 'N':
                TB = CblasNoTrans;
                break;
            
            case 't':
            case 'T':
                TB = CblasTrans;
                break;
            
            case 'h':
            case 'H':
                TB = CblasConjTrans;
                break;
            
            default:
                fprintf(stderr, "%c is not a valid transposition option.\n", *optarg);
                exit(-1);
            }
            break;
        }
        case 'd':
            DataType = *optarg;
            if (DataType >= 'a' && DataType <= 'z')
                DataType -= 'a' - 'A';
            if (DataType != 'S' && DataType != 'D' && DataType != 'C' && DataType != 'Z')
            {
                fprintf(stderr, "%c is not a valid datatype.\n", DataType);
                exit(-1);
            }
            break;
        case 'n':
            NReps = atoi(optarg);
            break;
        
        default:
            fprintf(stderr, "There is no option -%c.\n", opt);
            usage(argv[0], 0 == 1);
        }
    }

    // Get mandatory arguments
    if (argc < optind + 2)
        usage(argv[0], 0 == 1);

    NMin    = atoi(argv[optind]);
    NMax    = atoi(argv[optind + 1]);
    NStep   = atoi(argv[optind + 2]);
    Steps   = (NMax - NMin + 1) / NStep;


    // Variable declarations
    void* mA;
    void* mB;
    void* mC;
    void* mM;
    void* mLH;
    void* mRH;
    MKL_INT N;
    MKL_INT i;
    MKL_INT j;
    clock_t Start;
    clock_t End;
    double* Times = calloc(2 * NReps * Steps, sizeof(double));
    if (Times == NULL)
    {
        fprintf(stderr, "Cannot allocate times array.\n");
        exit(-1);
    }


    // Allocate matrices
    size_t ElemSize;
    switch (DataType)
    {
    case 'S': ElemSize = 4; break;
    case 'D': ElemSize = 8; break;
    case 'C': ElemSize = 8; break;
    case 'Z': ElemSize = 16; break;
    default: break;
    }
    printf("Matrices allocation... ");
    Start = clock();
    mA = malloc(NMax * NMax * ElemSize);
    if (mA == NULL)
    {
        fprintf(stderr, "Cannot allocate A.\n");
        exit(-1);
    }
    mB = malloc(NMax * NMax * ElemSize);
    if (mB == NULL)
    {
        fprintf(stderr, "Cannot allocate B.\n");
        exit(-1);
    }
    mC = malloc(NMax * NMax * ElemSize);
    if (mC == NULL)
    {
        fprintf(stderr, "Cannot allocate C.\n");
        exit(-1);
    }
    mM = malloc(3 * (NMax + 1) * (NMax + 1) * ElemSize / 2);
    if (mM == NULL)
    {
        fprintf(stderr, "Cannot allocate support memory.\n");
        exit(-1);
    }
    mLH = (char*)mM + (NMax + 1) * (NMax + 1) * ElemSize / 2;
    mRH = (char*)mLH + (NMax + 1) * (NMax + 1) * ElemSize / 2;
    End = clock();
    printf("Elapsed time is %.3fs.\n", (End - Start) / (double)CLOCKS_PER_SEC);

    // Fill matrices
    srand(0); // Reproducibility
    printf("Filling matrices... ");
    Start = clock();
    for (i = 0; i < NMax * NMax; ++i)
    {
        if (DataType == 'S')
        {
            ((float*)mA)[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            ((float*)mB)[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        }
        else if (DataType == 'D')
        {
            ((double*)mA)[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
            ((double*)mB)[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        }
        else if (DataType == 'C')
        {
            ((_Fcomplex*)mA)[i] = _FCbuild(2.0f * (rand() / (double)RAND_MAX) - 1.0f,
                                           2.0f * (rand() / (double)RAND_MAX) - 1.0f);
            ((_Fcomplex*)mB)[i] = _FCbuild(2.0f * (rand() / (double)RAND_MAX) - 1.0f,
                                           2.0f * (rand() / (double)RAND_MAX) - 1.0f);
        }
        else if (DataType == 'Z')
        {
            ((_Dcomplex*)mA)[i] = _Cbuild(2.0 * (rand() / (double)RAND_MAX) - 1.0,
                                          2.0 * (rand() / (double)RAND_MAX) - 1.0);
            ((_Dcomplex*)mB)[i] = _Cbuild(2.0 * (rand() / (double)RAND_MAX) - 1.0,
                                          2.0 * (rand() / (double)RAND_MAX) - 1.0);
        }
    }
    End = clock();
    printf("Elapsed time is %.3fs.\n\n", (End - Start) / (double)CLOCKS_PER_SEC);


    for (i = 0; i < Steps; ++i)
    {
        N = NMin + i * NStep;
        for (j = 0; j < NReps; ++j)
        {
            if (DataType == 'S')
            {
                Start = clock();
                cblas_sgemm(CblasRowMajor, TA, TB, N, N, N, 1.0f, mA, N, mB, N, 0.0f, mC, N);
                End = clock();
                Times[Steps * NReps + i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
                
                Start = clock();
                cblas_sstrassen(CblasRowMajor, TA, TB, N, N, N, 1.0f, mA, N, mB, N, 0.0f, mC, N, mM, mLH, mRH);
                End = clock();
                Times[i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
            }
            else if (DataType == 'D')
            {
                Start = clock();
                cblas_dgemm(CblasRowMajor, TA, TB, N, N, N, 1.0f, mA, N, mB, N, 0.0f, mC, N);
                End = clock();
                Times[Steps * NReps + i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
                
                Start = clock();
                cblas_dstrassen(CblasRowMajor, TA, TB, N, N, N, 1.0f, mA, N, mB, N, 0.0f, mC, N, mM, mLH, mRH);
                End = clock();
                Times[i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
            }
            else if (DataType == 'C')
            {
                _Fcomplex Zero = _FCbuild(0.0f, 0.0f);
                _Fcomplex One = _FCbuild(1.0f, 0.0f);
                Start = clock();
                cblas_cgemm(CblasRowMajor, TA, TB, N, N, N, &One, mA, N, mB, N, &Zero, mC, N);
                End = clock();
                Times[Steps * NReps + i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
                
                Start = clock();
                cblas_cstrassen(CblasRowMajor, TA, TB, N, N, N, &One, mA, N, mB, N, &Zero, mC, N, mM, mLH, mRH);
                End = clock();
                Times[i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
            }
            else if (DataType == 'Z')
            {
                _Dcomplex Zero = _Cbuild(0.0f, 0.0f);
                _Dcomplex One = _Cbuild(1.0f, 0.0f);
                Start = clock();
                cblas_zgemm(CblasRowMajor, TA, TB, N, N, N, &One, mA, N, mB, N, &Zero, mC, N);
                End = clock();
                Times[Steps * NReps + i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
                
                Start = clock();
                cblas_zstrassen(CblasRowMajor, TA, TB, N, N, N, &One, mA, N, mB, N, &Zero, mC, N, mM, mLH, mRH);
                End = clock();
                Times[i * NReps + j] = (End - Start) / (double)CLOCKS_PER_SEC;
            }
        }
    }



    // Save results
    FILE* Stream = fopen(OutFile, "w");
    if (Stream == NULL)
    {
        fprintf(stderr, "Cannot open file %s. Printing on standard output.\n\n", OutFile);
        Stream = stdout;
    }
    fprintf(Stream, "Size,CBLAS_%cSTRASSEN,CBLAS_%cGEMM\n", DataType, DataType);
    for (i = 0; i < Steps; ++i)
    {
        double GEMMTime = 0.0;
        double STRASSTime = 0.0;
        for (j = 0; j < NReps; ++j)
        {
            GEMMTime += Times[Steps * NReps + i * NReps + j];
            STRASSTime += Times[i * NReps + j];
        }
        fprintf(Stream, "%llu,%.4f,%.4f\n", NMin + i * NStep, STRASSTime / NReps, GEMMTime / NReps);
    }

    if (Stream != stdout)
        fclose(Stream);


    
    return 0;
}

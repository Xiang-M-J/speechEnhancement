/*
 * MATLAB Compiler: 8.5 (R2022b)
 * Date: Sun May  5 18:06:19 2024
 * Arguments:
 * "-B""macro_default""-W""lib:pesq,version=1.0""-T""link:lib""-d""D:\work\speec
 * hEnhancement\traditional-speech-enhancement\objective_measures\quality\pesq\f
 * or_testing""-v""D:\work\speechEnhancement\traditional-speech-enhancement\obje
 * ctive_measures\quality\pesq.m"
 */

#ifndef pesq_h
#define pesq_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_pesq_C_API 
#define LIB_pesq_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_pesq_C_API 
bool MW_CALL_CONV pesqInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_pesq_C_API 
bool MW_CALL_CONV pesqInitialize(void);

extern LIB_pesq_C_API 
void MW_CALL_CONV pesqTerminate(void);

extern LIB_pesq_C_API 
void MW_CALL_CONV pesqPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_pesq_C_API 
bool MW_CALL_CONV mlxPesq(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_pesq_C_API bool MW_CALL_CONV mlfPesq(int nargout, mxArray** scores, mxArray* ref_wav, mxArray* deg_wav);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif

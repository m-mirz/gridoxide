// Syntax-check stand-in for CUDA's library_types.h: only what cuDSS's headers use.
#pragma once
typedef enum cudaDataType_t { CUDA_R_32F = 0, CUDA_R_64F = 1, CUDA_C_32F = 4, CUDA_C_64F = 5, CUDA_R_32I = 10, CUDA_R_64I = 24 } cudaDataType_t;
typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType;

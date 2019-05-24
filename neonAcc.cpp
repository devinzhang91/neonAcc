#include "neonAcc.h"

#include <android/log.h>
#include <math.h>
#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void NeonAcc::neon_vecAdd(float *a, float *b, float *c, unsigned int len){
	int n = len/4;
	int m = len%4;


	for(int i=0; i<n; i++){
        float32x4_t va = vld1q_f32(&a[i*4]);
        float32x4_t vb = vld1q_f32(&b[i*4]);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(&c[i*4], vc);
	}
    if(m==0) return;

    float32x2_t va, vb, vc;
	switch (m){
        case 1:
            c[len-1] = a[len-1] + b[len-1];
            break;
        case 2:
            va = vld1_f32(&a[len-2]);
            vb = vld1_f32(&b[len-2]);
            vc = vadd_f32(va, vb);
            vst1_f32(&c[len-2], vc);
            break;
        case 3:
            va = vld1_f32(&a[len-3]);
            vb = vld1_f32(&b[len-3]);
            vc = vadd_f32(va, vb);
            vst1_f32(&c[len-3], vc);
            c[len-1] = a[len-1] + b[len-1];
            break;
	}

}

void NeonAcc::neon_vecMulc(float *a, float *b, float *c, unsigned int len) {
    int n = len / 4;
    int m = len % 4;

    for(int i=0; i<n; i++){
        float32x4_t va = vld1q_f32(&a[i*4]);
        float32x4_t vb = vld1q_f32(&b[i*4]);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(&c[i*4], vc);
    }
    if(m==0) return;

    float32x2_t va, vb, vc;
    switch (m){
        case 1:
            c[len-1] = a[len-1] * b[len-1];
            break;
        case 2:
            va = vld1_f32(&a[len-2]);
            vb = vld1_f32(&b[len-2]);
            vc = vmul_f32(va, vb);
            vst1_f32(&c[len-2], vc);
            break;
        case 3:
            va = vld1_f32(&a[len-3]);
            vb = vld1_f32(&b[len-3]);
            vc = vmul_f32(va, vb);
            vst1_f32(&c[len-3], vc);
            c[len-1] = a[len-1] * b[len-1];
            break;
    }
}

void NeonAcc::neon_matMulc(Matrix matA, Matrix matB, Matrix matC) {
    int a = matA.col / 4; //==matB.row
    int b = matB.col / 4; //==matC.col
    for(int i=0; i<matC.row*matC.col; i++){
        matC.data[i]=0;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads())
#endif
    for(int i=0; i<matA.row; i++){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads())
#endif
        for(int j=0; j<matA.col; j++){
            float32x4_t va = vdupq_n_f32(matA.data[i*matA.col + j]);
            for(int k=0; k<b; k++){
                float32x4_t vb = vld1q_f32(&matB.data[j*matB.col + k*4]);
                float32x4_t vc = vld1q_f32(&matC.data[i*matC.col + k*4]);
                vc = vfmaq_f32(vc, va, vb);     //vfmaq_laneq_f32
                vst1q_f32(&matC.data[i*matC.col + k*4], vc);
            }
        }
    }
}

void NeonAcc::neon_matMulc1(Matrix matA, Matrix matB, Matrix matC) {
    int a = matA.col / 4; //==matB.row
    int b = matB.col / 4; //==matC.col
    for(int i=0; i<matC.row*matC.col; i++){
        matC.data[i]=0;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads())
#endif
    for(int i=0; i<matA.row; i++){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads())
#endif
        for(int j=0; j<a; j++){
            float32x4_t va = vld1q_f32(&matA.data[i*matA.col + j*4]);
            for(int k=0; k<b; k++){
                float32x4_t vb = vld1q_f32(&matB.data[j*matB.col + k*4]);
                float32x4_t vc = vld1q_f32(&matC.data[i*matC.col + k*4]);
                vc = vfmaq_laneq_f32(vc, vb, va, 0);
                vc = vfmaq_laneq_f32(vc, vb, va, 1);
                vc = vfmaq_laneq_f32(vc, vb, va, 2);
                vc = vfmaq_laneq_f32(vc, vb, va, 3);
                vst1q_f32(&matC.data[i*matC.col + k*4], vc);
            }
        }
    }
}

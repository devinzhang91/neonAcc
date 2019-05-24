//
// Created by devin on 5/5/19.
//
#include <jni.h>
#include <string>
#include <android/log.h>
#include <math.h>
#include <arm_neon.h>
#include <omp.h>
#include "math_neon/math_neon.h"
#include "neonAcc.h"

const char * JNINEON_TAG = "JNINEON_TAG";


extern "C" JNIEXPORT void JNICALL Java_com_example_neontest_MainActivity_vectorTest
        (JNIEnv* env, jobject obj) {


    __android_log_print(ANDROID_LOG_DEBUG, JNINEON_TAG,  "Vector test.\n");
}


//extern "C" JNIEXPORT void JNICALL Java_com_example_neontest_MainActivity_vectorAdd
//        (JNIEnv* env, jobject obj) {
//
//    float ret[4] = {0.0, 0.0, 0.0, 0.0};
//
//    float32x4_t v1 = (float32x4_t) { 0.0f,  1.0f,  2.0f,  3.0f};
//    float32x4_t v2 = (float32x4_t) {-0.0f, -1.0f, -2.0f, -3.0f};
//    float32x4_t v3 = vaddq_f32(v1, v2);    // v3 = { 0.0f,  0.0f,  0.0f,  0.0f}
//    float32x4_t v4 = vmulq_f32(v1, v2);    // v4 = { 0.0f, -1.0f, -4.0f, -9.0f}
//    vst1q_f32(ret, v4);
//    __android_log_print(ANDROID_LOG_DEBUG, JNINEON_TAG,  "v4: %f %f %f %f\n", ret[0], ret[1], ret[2], ret[3]);
//    __android_log_print(ANDROID_LOG_DEBUG, JNINEON_TAG,  "Vector add.\n");
//}

extern "C" JNIEXPORT void JNICALL Java_com_example_neontest_MainActivity_vectorAdd
        (JNIEnv* env, jobject obj,
         jfloatArray vecA, jfloatArray vecB, jfloatArray vecC) {
    jfloat* arrayVecA = env->GetFloatArrayElements(vecA, NULL);
    jfloat* arrayVecB = env->GetFloatArrayElements(vecB, NULL);
    jfloat* arrayVecC = env->GetFloatArrayElements(vecC, NULL);
    jint vecLen = env->GetArrayLength(vecC);

    NeonAcc neonAcc ;
    neonAcc.neon_vecAdd(arrayVecA, arrayVecB, arrayVecC, vecLen);

    env->ReleaseFloatArrayElements(vecA, arrayVecA, JNI_COMMIT);
    env->ReleaseFloatArrayElements(vecB, arrayVecB, JNI_COMMIT);
    env->ReleaseFloatArrayElements(vecC, arrayVecC, JNI_COMMIT);

}

extern "C" JNIEXPORT void JNICALL Java_com_example_neontest_MainActivity_vectorMulc
        (JNIEnv* env, jobject obj,
         jfloatArray vecA, jfloatArray vecB, jfloatArray vecC) {
    jfloat* arrayVecA = env->GetFloatArrayElements(vecA, NULL);
    jfloat* arrayVecB = env->GetFloatArrayElements(vecB, NULL);
    jfloat* arrayVecC = env->GetFloatArrayElements(vecC, NULL);
    jint vecLen = env->GetArrayLength(vecC);

    NeonAcc neonAcc ;
    neonAcc.neon_vecMulc(arrayVecA, arrayVecB, arrayVecC, vecLen);

    env->ReleaseFloatArrayElements(vecA, arrayVecA, JNI_COMMIT);
    env->ReleaseFloatArrayElements(vecB, arrayVecB, JNI_COMMIT);
    env->ReleaseFloatArrayElements(vecC, arrayVecC, JNI_COMMIT);

}

extern "C" JNIEXPORT jobject JNICALL Java_com_example_neontest_MainActivity_matrixMulc
        (JNIEnv* env, jobject obj,
         jobject matAj, jobject matBj, jobject matCj) {

#ifdef _OPENMP
#pragma omp parallel num_threads(omp_get_max_threads())
    __android_log_print(ANDROID_LOG_DEBUG, JNINEON_TAG,  "Message from %d thread\n", omp_get_thread_num());
#endif

    Matrix matAc, matBc, matCc;
    float* data; jfloatArray matdata;
    jfieldID matrowFid, matcolFid, matdataFid;

    //get java class
    jclass jcMatrix = env->FindClass("com/example/neontest/Matrix");
    //class member id
    matrowFid = env->GetFieldID(jcMatrix, "row", "I");
    matcolFid = env->GetFieldID(jcMatrix, "col", "I");
    matdataFid = env->GetFieldID(jcMatrix, "data", "[F");
    //matA
    matAc.row = env->GetIntField(matAj, matrowFid);
    matAc.col = env->GetIntField(matAj, matcolFid);
    matAc.data = new float[matAc.row*matAc.col];
    matdata = (jfloatArray)env->GetObjectField(matAj, matdataFid);
    data = env->GetFloatArrayElements(matdata, 0);
    memcpy(matAc.data, data, matAc.row*matAc.col*sizeof(float));

    //matB
    matBc.row = env->GetIntField(matBj, matrowFid);
    matBc.col = env->GetIntField(matBj, matcolFid);
    matBc.data = new float[matBc.row*matBc.col];
    matdata = (jfloatArray)env->GetObjectField(matBj, matdataFid);
    data = env->GetFloatArrayElements(matdata, 0);
    memcpy(matBc.data, data, matBc.row*matBc.col*sizeof(float));

    //matC
    matCc.row = env->GetIntField(matCj, matrowFid);
    matCc.col = env->GetIntField(matCj, matcolFid);
    matCc.data = new float[matCc.row*matCc.col];
    matdata = (jfloatArray)env->GetObjectField(matCj, matdataFid);
    data = env->GetFloatArrayElements(matdata, 0);
    memcpy(matCc.data, data, matCc.row*matCc.col*sizeof(float));

    //
    NeonAcc neonAcc ;
//    matmul4_c(matAc.data, matBc.data, matCc.data);
//    matmul4_neon(matAc.data, matBc.data, matCc.data);
    neonAcc.neon_matMulc(matAc, matBc, matCc);

    //new object
    jobject joMatrix = env->AllocObject(jcMatrix);
    env->SetIntField(joMatrix, matrowFid, matCc.row);
    env->SetIntField(joMatrix, matcolFid, matCc.col);
    //copy matC in object
    jfloatArray jarr = env->NewFloatArray(matBc.row*matBc.col);
    jfloat *jft = env->GetFloatArrayElements(jarr, 0);
    memcpy(jft, matCc.data, matCc.row*matCc.col);
    env->SetFloatArrayRegion(jarr, 0, matCc.row*matCc.col, jft);
    env->SetObjectField(joMatrix, matdataFid, jarr);

    //test code
    for(int i=0; i<matCc.col; i++){
        for(int j=0; j<matAc.row; j++)
            __android_log_print(ANDROID_LOG_DEBUG, JNINEON_TAG,  "%d:%2.2f ", j+i*matCc.row, matCc.data[j+i*matCc.row]);
    }

    delete matAc.data;
    delete matBc.data;
    delete matCc.data;

    return joMatrix;
}



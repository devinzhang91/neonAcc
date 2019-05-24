#ifndef ___NEON_ACC_H__
#define ___NEON_ACC_H__

typedef struct{
	int row;
	int col;
	float* data;
}Matrix;

class NeonAcc{
public:
	void neon_vecAdd(float *a, float *b, float *c, unsigned int len);
	void neon_vecMulc(float *a, float *b, float *c, unsigned int len);

	void neon_matMulc(Matrix matA,  Matrix matB, Matrix matC);
	void neon_matMulc1(Matrix matA,  Matrix matB, Matrix matC);
};

#endif

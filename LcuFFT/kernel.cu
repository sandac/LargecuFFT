#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "cufft.h"
#include <cusparse.h>
#include "cuComplex.h"
#include<cublas_v2.h>
#include<cusparse_v2.h>
#define  NX 500

__device__  bool flag = true;
//cudaError_t TestFFT(cuComplex* idata, cuComplex* odata);
cudaError_t TestFFT(cuComplex* idata,cuComplex* odata)
{
	/*int* cdata_dev = 0;
	if (cudaMalloc((void**)&cdata_dev, sizeof(int) * 1024 * 1024 * 1024) != cudaSuccess)
		printf("CUDA MALLOC CDATA FAILED!\n");
	cudaFree(cdata_dev);*/
	cufftHandle plan;
	cufftComplex *data;


	cudaMalloc((void**)&data, sizeof(cufftComplex)*NX);
	cudaMemcpy(data, idata, sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	
	
	if (cufftPlan1d(&plan,NX,CUFFT_C2C,1)!=CUFFT_SUCCESS)
	{
		std::cout << "Plan creation failed" << std::endl;
	}
	
	

	if (cufftExecC2C(plan,data,data,CUFFT_FORWARD)!=CUFFT_SUCCESS)
	{
		std::cout << "ExecC2C Forward failed" << std::endl;
	}
	cudaDeviceSynchronize();
	
	/*if (cudaDeviceSynchronize()!=cudaSuccess)
	{
		std::cout << "Failed to synchronized" << std::endl;
	}*/
	cudaMemcpy(odata, data, sizeof(cufftComplex)*NX, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);

	return cudaSuccess;
}




//int TestCuSparse()
//{
//	cuComplex Znear[8];
//	int Z_near_row[5];
//	int Z_near_col[8];
//	int Z_near_length = 8;
//	Z_near_row[0] = 0;
//	Z_near_row[1] = 2;
//	Z_near_row[2] = 4;
//	Z_near_row[3] = 6;
//	Z_near_row[4] = 8;
//
//	Z_near_col[0] = 0;
//	Z_near_col[1] = 2;
//	Z_near_col[2] = 0;
//	Z_near_col[3] = 1;
//	Z_near_col[4] = 1;
//	Z_near_col[5] = 3;
//	Z_near_col[6] = 1;
//	Z_near_col[7] = 3;
//	Znear[0].x = 1.0f; Znear[0].y = 1;
//	Znear[1].x = 1.0f; Znear[1].y = 3.0;
//	Znear[2].x = 1.0f; Znear[2].y = -1.0f;
//	Znear[3].x = 2.0f; Znear[3].y = 1.0f;
//	Znear[4].x = 3.0f; Znear[4].y = 4.0f;
//	Znear[5].x = 2.0f; Znear[5].y = 1.0f;
//	Znear[6].x = 1.0f; Znear[6].y = 5.0f;
//	Znear[7].x = 4.0f; Znear[7].y = 5.0f;
//
//	cuComplex V[4];
//	V[0].x = 1; V[0].y = 1;
//	V[1].x = 2; V[1].y = 3;
//	V[2].x = 3; V[2].y = 4;
//	V[3].x = 4; V[3].y = 5;
//
//	cuComplex V1[4];
//	V1[0].x = 2; V1[0].y = 3;
//	V1[1].x = 1; V1[1].y = 7;
//	V1[2].x = 3; V1[2].y = 4;
//	V1[3].x = 5; V1[3].y = 9;
//
//
//
//
//	cuComplex alpha = { 1.0f, 0.0f };
//	cuComplex beta = { 0.0f, 0.0f };
//	cusparseHandle_t handle;
//	cusparseMatDescr_t descr;
//	cusparseCreate(&handle);
//	cusparseCreateMatDescr(&descr);
//
//	cuComplex* Znear_dev = 0;
//	int* Znear_row_dev = 0;
//	int* Znear_col_dev = 0;
//	cuComplex* V_dev = 0;
//	cuComplex* res_dev = 0;
//	cudaMalloc((void**)&Znear_dev, sizeof(cuComplex) * 8);
//	cudaMalloc((void**)&Znear_row_dev, sizeof(int) * 5);
//	cudaMalloc((void**)&Znear_col_dev, sizeof(int) * 8);
//	cudaMalloc((void**)&V_dev, sizeof(cuComplex) * 4);
//	cudaMalloc((void**)&res_dev, sizeof(cuComplex) * 4);
//	cudaMemcpy(Znear_dev, Znear, sizeof(cuComplex) * 8,cudaMemcpyHostToDevice);
//	cudaMemcpy(Znear_row_dev, Z_near_row, sizeof(int) * 5, cudaMemcpyHostToDevice);
//	cudaMemcpy(Znear_col_dev, Z_near_col, sizeof(int) * 8, cudaMemcpyHostToDevice);
//	cudaMemcpy(V_dev, V, sizeof(cuComplex) * 4, cudaMemcpyHostToDevice);
//	cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 8, &alpha, descr, Znear_dev, Znear_row_dev, Znear_col_dev, V_dev, &beta, res_dev);
//	cudaMemcpy(V, res_dev, sizeof(cuComplex) * 4, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < 4;i++)
//	{
//		printf("%lf+%lfi\n", V[i].x, V[i].y);
//	}
//
//	cudaMemcpy(V_dev, V1, sizeof(cuComplex) * 4, cudaMemcpyHostToDevice);
//	cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 8, &alpha, descr, Znear_dev, Znear_row_dev, Znear_col_dev, V_dev, &beta, res_dev);
//	cudaMemcpy(V1, res_dev, sizeof(cuComplex) * 4, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < 4; i++)
//	{
//		printf("%lf+%lfi\n", V1[i].x, V1[i].y);
//	} 
//
//
//
//
//
//
//
//
//
//
//	cusparseDestroy(handle);
//	cusparseDestroyMatDescr(descr);
//
//
//	return 0;
//}


void TestcublasCdotc()
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	cuComplex x[3] = { { 1, 2 }, { 2, 3 }, { 3, 4 } };
	cuComplex y[3] = { { 3, 1 }, { 1, 4 }, { 1, 2 } };
	cuComplex tmp[3];
	//for (int i = 0; i < 4;i++)
	//{
	//	printf("%f,%f\n", x[i].x, x[i].y);
	//}
	cuComplex *x_dev = 0;
	cuComplex *y_dev = 0;
	if (cudaMalloc((void**)&x_dev, sizeof(cuComplex) * 3) != cudaSuccess ||
		cudaMalloc((void**)&y_dev, sizeof(cuComplex) * 3) != cudaSuccess)
		printf("cuda malloc failed!\n");
	if (cudaMemcpy(x_dev, x, sizeof(cuComplex) * 3, cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(y_dev, y, sizeof(cuComplex) * 3, cudaMemcpyHostToDevice) != cudaSuccess)
		printf("cuda memcpy failed!\n");
	/*cublasSetVector(3, sizeof(cuComplex), x, 1, x_dev, 1);
	cublasSetVector(3, sizeof(cuComplex), y, 1, y_dev, 1);*/
	cudaDeviceSynchronize();
	
	/*for (int i = 0; i < 4; i++){
		printf("%f,%f\n", tmp[i].x, tmp[i].y);
	}*/
	//cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	cuComplex result;

	cublasCdotc(handle, 3, x_dev, 1, y_dev, 1, &result);
	
	cudaDeviceSynchronize();
	//cudaMemcpy(result, result_dev, sizeof(cuComplex), cudaMemcpyDeviceToHost);
	printf("%20.15f,%20.15f\n", result.x,result.y);
	cublasDestroy(handle);
}




__global__ void f1_kernel(int* arr, int size){
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	//if (id == 0)
		//printf("call f1_kernel\n");
	if (id < size){
		//while (flag){}
		int k = 1;
		int j = 100000000;
		int res = 9;
		while (k){
			while (j>0){
				j--;
				res = res*k;
			}
			k--;
			//printf("k:%d\n", k);
		}
		k = INT_MAX;
		j = INT_MAX;
		/*while (k + j){
			k--;
			j--;
			res = res*k + 1 * res*res + res - res * 8*j;
		}*/
		for (int i = 0; i < 1000000000; i++){
			k=k-1+k^7;
		}
		arr[id] += 1;
	}
}
cudaError_t f1(int* arr,int size){
	printf("call f1\n");
	dim3 grid(size / 1024 + 1, 1, 1);
	dim3 block(1024, 1, 1);
	f1_kernel << <grid, block >> >(arr, size);
	//if (cudaDeviceSynchronize() != cudaSuccess){
	//	printf("cudasync failed\n");
	//	}
	return cudaSuccess;
}

__global__ void f2_kernel(float* arr, int size){
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if (id == 1)
		printf("call f2_kernel\n");
	if (id < size){
		arr[id] /=0.111;
	}
	flag = false;
}

cudaError_t f2(float* arr, int size){
	//int res[1000];
	//flag = false;
	//cudaMemcpy(res, arr, sizeof(int) * 1000, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 1000; i++)
	//{
	//	printf("f%d,", res[i]);
	//}
	printf("call f2\n");
	dim3 grid(size / 1024 + 1, 1, 1);
	dim3 block(1024, 1, 1);
	f2_kernel << <grid, block >> >(arr, size);
	printf("here\n");
	
	/*if (cudaDeviceSynchronize() != cudaSuccess){
		printf("cudasync failed\n");
	}*/
	return cudaSuccess;
}

cudaError_t f3(int* arr, int size){
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseMatDescr_t descr = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
		printf("cusparse create handle failed\n");
	if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS)
		printf("cusparse create matrix descr failed\n");

	float* V = new float[size];
	for (int i = 0; i < size; i++){
		V[i] = 2.0f;
	}
	int* csr_col = new int[size];
	for (int i = 0; i < size;i++)
	{
		csr_col[i] = 0;
	}
	int* csr_row = new int[size + 1];
	
	for (int i = 0; i < size + 1;i++)
	{
		csr_row[i] = i;
	}
	float* csr_val = new float[size];
	for (int i = 0; i < size; i++){
		csr_val[i] = 1.0f;
	}
	float* V_dev = 0;
	int* csr_col_dev = 0;
	int* csr_row_dev = 0;
	float* csr_val_dev = 0;
	cudaMalloc((void**)&V_dev, sizeof(float)*size);
	cudaMalloc((void**)&csr_row_dev, sizeof(int)*(size + 1));
	cudaMalloc((void**)&csr_col_dev, sizeof(int)*size);
	cudaMalloc((void**)&csr_val_dev, sizeof(float)*size);

	cudaMemcpy(V_dev, V, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(csr_row_dev, csr_row, sizeof(int)*(size + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_col_dev, csr_col, sizeof(int)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(csr_val_dev, csr_val, sizeof(float)*size, cudaMemcpyHostToDevice);


	f2(V_dev, size);

	if (cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, size, size, size, &alpha, descr, csr_val_dev,
		csr_row_dev, csr_col_dev, V_dev, &beta, V_dev) != CUSPARSE_STATUS_SUCCESS){
		printf("cusparse csr mv failed\n");
	}

	float* res = new float[size];
	cudaMemcpy(res, V_dev, sizeof(float)*size, cudaMemcpyDeviceToHost);


	for (int i = 0; i < 100; i++){
		std::cout << res[i] << " ,";
	}
	std::cout << std::endl;


















	return cudaSuccess;
}


int test1(){

	f3(NULL, 100000);


	//int len = 10000;
	//int* arr = new int[len];
	//memset(arr, 0, sizeof(int)*len);
	////int arr[1000] = { 0 };
	//int *arr_dev = 0;
	//int *arr1_dev = 0;
	//cudaMalloc((void**)&arr_dev, sizeof(int) * len);
	//cudaMalloc((void**)&arr1_dev, sizeof(int) * len);
	//cudaMemcpy(arr_dev, arr, sizeof(int) * len, cudaMemcpyHostToDevice);
	//cudaMemcpy(arr1_dev, arr, sizeof(int) * len, cudaMemcpyHostToDevice);



	//f1(arr_dev, len);
	//f2(arr1_dev, len);



	//int * res = new int[len];
	////int res[len];
	//cudaMemcpy(res, arr1_dev, sizeof(int) * len, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 1000; i++)
	//{
	//	printf("%d,", res[i]);
	//}
	//printf("\n");
	return 0;
}

int main()
{
	

	test1();
	getchar();



	//TestcublasCdotc();

	//TestCuSparse();


	//clock_t start, end;
	//start = clock();
	//cufftHandle plan;
	//for (int i = 0; i < 10000000000;i++)
	//{
	//	int m=9;
	//	int n;
	//	//n = m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m*m;
	//}
	//getchar();
	//end = clock();
	//printf("%f\n", end - start);




















	cuComplex* input_data = new cuComplex[NX];
	cuComplex* output_data = new cuComplex[NX];
	std::cout << "malloc finished" << std::endl;
	//cuComplex* output_data = new cuComplex[NX];
	for (int i = 0; i < NX; i++)
	{
		input_data[i].x = i;
		input_data[i].y = 0;
		//onput_data[i].y = rand() / 12;
	}

	std::cout << "Initial input_data finished" << std::endl;

	//cuComplex* data_dev = 0;
	//cudaMalloc((void**)&data_dev, sizeof(cuComplex)*NX);
	//cudaMemcpy(data_dev, input_data, sizeof(cuComplex)*NX, cudaMemcpyHostToDevice);
	//memset(input_data, 0, sizeof(cuComplex)*NX);

	//cudaMemcpy(input_data, data_dev + 100, sizeof(cuComplex) * 100, cudaMemcpyDeviceToHost);

	for (int j = 0; j < 100; j++)
	{
		std::cout << input_data[j].x << std::endl;
	}
	TestFFT(input_data, input_data);
	for (int i = 0; i < 10; i++)
	{
		std::cout << input_data[i].x << "+" << input_data[i].y << std::endl;
	}









	//clock_t start, end;
	//start = clock();
	//TestFFT(input_data, input_data);

	//end = clock();
	//std::cout << "time cost" << end - start << std::endl;

	////std::cout << "Finished/*,Time cost:"/* <<end-start*/<< std::endl;


	//for (int i = 0; i < 10; i++)
	//{
	//	std::cout << input_data[i].x << "+" << input_data[i].y << std::endl;
	//}

	//delete[] input_data;
	//delete[] output_data;
	getchar();
	return 0;
}
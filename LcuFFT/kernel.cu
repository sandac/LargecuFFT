#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "cufft.h"
#define  NX 250000000


//cudaError_t TestFFT(cuComplex* idata, cuComplex* odata);
cudaError_t TestFFT(cuComplex* idata,cuComplex* odata)
{
	cufftHandle plan;
	cufftComplex *data;
	cudaMalloc((void**)&data, sizeof(cufftComplex)*NX);
	cudaMemcpy(data, idata, sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	clock_t start, end;
	start = clock();
	
	if (cufftPlan1d(&plan,NX,CUFFT_C2C,1)!=CUFFT_SUCCESS)
	{
		std::cout << "Plan creation failed" << std::endl;
	}
	
	if (cufftExecC2C(plan,data,data,CUFFT_FORWARD)!=CUFFT_SUCCESS)
	{
		std::cout << "ExecC2C Forward failed" << std::endl;
	}
	cudaDeviceSynchronize();
	end = clock();
	std::cout << "time cost" << end - start << std::endl;
	/*if (cudaDeviceSynchronize()!=cudaSuccess)
	{
		std::cout << "Failed to synchronized" << std::endl;
	}*/
	cudaMemcpy(odata, data, sizeof(cufftComplex)*NX, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	return cudaSuccess;
}


int main()
{

	cuComplex* input_data = new cuComplex[NX];
	std::cout << "malloc finished" << std::endl;
	//cuComplex* output_data = new cuComplex[NX];
	for (int i = 0; i < NX; i++)
	{
		input_data[i].x = rand() / 10;
		//onput_data[i].y = rand() / 12;
	}

	std::cout << "Initial input_data finished" << std::endl;


	
	TestFFT(input_data, input_data);



	std::cout << "Finished/*,Time cost:"/* <<end-start*/<< std::endl;


	for (int i = 0; i < 10; i++)
	{
		std::cout << input_data[i].x << "+" << input_data[i].y << std::endl;
	}

	delete[] input_data;
	//delete[] output_data;
	getchar();
	return 0;
}
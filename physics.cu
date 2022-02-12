// 
// Wrapper for CUDA functions
// 
// 2022, Jonathan Tainer
// 

#include <math.h>
#include "kernel.h"
#include "physics.h"

PointMass* createElements(int numOfElements) {
	PointMass* ptr;
	cudaMalloc((void**)&ptr, sizeof(PointMass) * numOfElements);
	return ptr;
}

void deleteElements(PointMass* element) {
	cudaFree(element);
}

void copyToDev(PointMass* devElement, PointMass* sysElement, int numOfElements) {
	cudaMemcpy(devElement, sysElement, sizeof(PointMass) * numOfElements, cudaMemcpyHostToDevice);
}

void copyToSys(PointMass* sysElement, PointMass* devElement, int numOfElements) {
	cudaMemcpy(sysElement, devElement, sizeof(PointMass) * numOfElements, cudaMemcpyDeviceToHost);
}

void step(PointMass* element, int numOfElements, float dt) {
	

	updateAcc<<<1, numOfElements>>>(element, numOfElements);
	updatePos<<<1, numOfElements>>>(element, numOfElements, dt);
}



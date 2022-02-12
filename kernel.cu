// 
// CUDA kernels to compute motion of a system of particles
// 
// 2022, Jonathan Tainer
// 

#include "kernel.h"

#define G 1.f

__global__
void updateAcc(PointMass* element, int numOfElements) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numOfElements) {
		
		// Reset acceleration vector to zero
		element[tid].acc = (vec2) { 0.f, 0.f };


		// Compute the gravitational force of every other element on this element
		for (int i = 0; i < numOfElements; i++) {
			if (i != tid) {
				float dx = element[i].pos.x - element[tid].pos.x;
				float dy = element[i].pos.y - element[tid].pos.y;
				float radsqr = (dx * dx) + (dy * dy);
				float rad = sqrt(radsqr);
				
				float acc = element[i].mass * element[tid].mass * G / radsqr;
				
				float cosine = dx / rad;
				float sine = dy / rad;

				float accx = acc * cosine;
				float accy = acc * sine;

				// Add the computed acceleration component to the element's net acceleration
				element[tid].acc.x += accx;
				element[tid].acc.y += accy;
			}
		}
	}
}

__global__
void updatePos(PointMass* element, int numOfElements, float dt) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tid < numOfElements) {

		// Integrate acceleration to find velocity
		element[tid].vel.x += element[tid].acc.x * dt;
		element[tid].vel.y += element[tid].acc.y * dt;

		// Integrate velocity to find position
		element[tid].pos.x += element[tid].vel.x * dt;
		element[tid].pos.y += element[tid].vel.y * dt;

	}
}

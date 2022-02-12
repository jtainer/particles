// 
// CUDA kernels to compute the motion of a system of particles
//
// 2022, Jonathan Tainer
//

#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include "pointmass.h"

__global__
void updateAcc(PointMass* element, int numOfElements);

__global__
void updatePos(PointMass* element, int numOfElements, float dt);

#endif

// 
// Point mass for gravity simulation
//
// 2022, Jonathan Tainer
//

#ifndef POINTMASS_H
#define POINTMASS_H

typedef struct vec2 {
	float x;
	float y;
} vec2;

typedef struct PointMass {
	vec2 pos;
	vec2 vel;
	vec2 acc;
	float mass;
} PointMass;

#endif

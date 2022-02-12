// 
// Gravitational particle system
//
// 2022, Jonathan Tainer
//

#include "physics.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <raylib.h>

int main() {

	srand(time(NULL));

	const int numOfElements = 512;

	PointMass* sysptr = (PointMass*) malloc(sizeof(PointMass) * numOfElements);;

	for (int i = 0; i < numOfElements; i++) {
		sysptr[i].pos = (vec2) { (float) (rand() % 640), (float) (rand() % 480) };
		sysptr[i].vel = (vec2) { (float) (rand() % 5) - 2, (float) (rand() % 5) - 2 };
		sysptr[i].mass = 10.f;
	}

	printf("%f, %f\n", sysptr[0].pos.x, sysptr[0].pos.y);

	PointMass* devptr = createElements(numOfElements);
	copyToDev(devptr, sysptr, numOfElements);

	// Raylib setup
	InitWindow(640, 480, "Gravity");
//	SetTargetFPS(60);

	while (!WindowShouldClose()) {
		
		BeginDrawing();
		ClearBackground(RAYWHITE);
		
		for (int i = 0; i < numOfElements; i++) {
			DrawCircleV((Vector2) { sysptr[i].pos.x, sysptr[i].pos.y }, 5.f, RED);
		}
		
		DrawFPS(10, 10);
		EndDrawing();

		// Compute next state of particle system
		step(devptr, numOfElements, 0.01f);
		copyToSys(sysptr, devptr, numOfElements);
		printf("%f, %f\n", sysptr[0].pos.x, sysptr[0].pos.y);
	}

	CloseWindow();


	deleteElements(devptr);
	free(sysptr);
	return 0;
}

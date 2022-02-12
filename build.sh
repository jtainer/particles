nvcc -c kernel.cu physics.cu
g++ -c example.cpp
g++ -L/usr/local/cuda/lib64 kernel.o physics.o example.o -lcudart -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

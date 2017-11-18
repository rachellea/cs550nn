#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct mnist_data {
	unsigned char data[28][28];
	unsigned int label;
} mnist_data;

typedef struct mnist_data_ann {
	double data[28*28];
	double label;
} mnist_data_ann;

int loadMNISTData(const char *image_filename, const char *label_filename,
	mnist_data **data, unsigned int *count);
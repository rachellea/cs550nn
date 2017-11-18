#include <stdio.h>
#include "genann.h"
#include "mnist.cuh"

#define SIZE 28

void copy_to_double(mnist_data_ann *data_ann, mnist_data *data_raw, int cnt) {
	for (int i = 0; i < cnt; i++) {
		data_ann[i].label = (double)data_raw[i].label;
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < SIZE; k++) {
				data_ann[i].data[j*28+k] = (double)data_raw[i].data[j][k];
			}
		}
	}
}

int main(int argc, char *argv[])
{
	printf("GENANN example 1.\n");
	printf("Train a mnist.\n");

	char* labelFileName = "t10k-labels.idx1-ubyte";
	char* imageFileName = "t10k-images.idx3-ubyte";
	// char* labelFileName = "train-labels.idx1-ubyte";
	// char* imageFileName = "train-images.idx3-ubyte";

	mnist_data *data_raw;
	mnist_data_ann *data_ann;
	unsigned int cnt;
	int ret;

	if (ret = loadMNISTData(imageFileName, labelFileName, &data_raw, &cnt)) {
		printf("An error occured: %d\n", ret);
	}

	data_ann = (mnist_data_ann*)malloc(sizeof(double) * (28 * 28 + 1) * cnt);
	copy_to_double(data_ann, data_raw, cnt);

	int i;

	/* New network with 2 inputs,
	* 1 hidden layer of 2 neurons,
	* and 1 output. */
	genann *ann = genann_init(28*28, 2, 28*28, 1);

	/* Train on the four labeled data points many times. */
	for (i = 0; i < 300; ++i) {
		for (int j = 0; j < cnt; j++) {
			genann_train(ann, data_ann[j].data, &data_ann[j].label, 0.9);
		}
	}

	/* Run the network and see what it predicts. */
	printf("%lf, %lf\n", *genann_run(ann, data_ann[0].data), data_ann[0].label);
	printf("%lf, %lf\n", *genann_run(ann, data_ann[1].data), data_ann[1].label);
	printf("%lf, %lf\n", *genann_run(ann, data_ann[2].data), data_ann[2].label);
	printf("%lf, %lf\n", *genann_run(ann, data_ann[3].data), data_ann[3].label);

	char c;
	scanf_s("%c", &c);
	genann_free(ann);
	free(data_raw);
	return 0;
}
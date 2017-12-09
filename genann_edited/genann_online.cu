/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015, 2016 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#define LOOKUP_SIZE 4096

__device__ double genann_act_sigmoid_device(double a) {
	if (a < -45.0) return 0;
	if (a > 45.0) return 1;
	return 1.0 / (1 + exp(-a));
}


double genann_act_sigmoid(double a) {
	if (a < -45.0) return 0;
	if (a > 45.0) return 1;
	return 1.0 / (1 + exp(-a));
}


double genann_act_sigmoid_cached(double a) {
	/* If you're optimizing for memory usage, just
	 * delete this entire function and replace references
	 * of genann_act_sigmoid_cached to genann_act_sigmoid
	 */
	const double min = -15.0;
	const double max = 15.0;
	static double interval;
	static int initialized = 0;
	static double lookup[LOOKUP_SIZE];

	/* Calculate entire lookup table on first run. */
	if (!initialized) {
		interval = (max - min) / LOOKUP_SIZE;
		int i;
		for (i = 0; i < LOOKUP_SIZE; ++i) {
			lookup[i] = genann_act_sigmoid(min + interval * i);
		}
		/* This is down here to make this thread safe. */
		initialized = 1;
	}

	int i;
	i = (int)((a - min) / interval + 0.5);
	if (i <= 0) return lookup[0];
	if (i >= LOOKUP_SIZE) return lookup[LOOKUP_SIZE - 1];
	return lookup[i];
}


double genann_act_threshold(double a) {
	return a > 0;
}


double genann_act_linear(double a) {
	return a;
}


genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
	if (hidden_layers < 0) return 0;
	if (inputs < 1) return 0;
	if (outputs < 1) return 0;
	if (hidden_layers > 0 && hidden < 1) return 0;


	const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
	const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
	const int total_weights = (hidden_weights + output_weights);

	const int total_neurons = (inputs + hidden * hidden_layers + outputs);

	/* Allocate extra size for weights, outputs, and deltas. */
	const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
	genann *ret = (genann*)malloc(size);
	if (!ret) return 0;

	ret->inputs = inputs;
	ret->hidden_layers = hidden_layers;
	ret->hidden = hidden;
	ret->outputs = outputs;

	ret->total_weights = total_weights;
	ret->total_neurons = total_neurons;

	/* Set pointers. */
	ret->weight = (double*)((char*)ret + sizeof(genann));
	ret->output = ret->weight + ret->total_weights;
	ret->delta = ret->output + ret->total_neurons;

	genann_randomize(ret);

	ret->activation_hidden = genann_act_sigmoid_cached;
	ret->activation_output = genann_act_sigmoid_cached;

	return ret;
}


genann *genann_read(FILE *in) {
	int inputs, hidden_layers, hidden, outputs;
	fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);

	genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		fscanf(in, " %le", ann->weight + i);
	}

	return ann;
}

/* calculate genann pointers */
void set_genann_pointers(genann *ret) {
	/* Set pointers. */
	ret->weight = (double*)((char*)ret + sizeof(genann));
	ret->output = ret->weight + ret->total_weights;
	ret->delta = ret->output + ret->total_neurons;
}

/* calculate genann pointers for the network on device memory */
__global__ void set_genann_pointers_device(genann *d_genann) {
	/* Set pointers. */
	d_genann->weight = (double*)((char*)d_genann + sizeof(genann));
	d_genann->output = d_genann->weight + d_genann->total_weights;
	d_genann->delta = d_genann->output + d_genann->total_neurons;
}

/* copy a genann struct to GPU using CUDA APIs 
 * also recalculate the pointer locations so that they point to device memory 
 */
genann *genann_device_copy(genann const *ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	genann *ret;
	cudaMalloc((void **)&ret, size);
	if (!ret) return 0;

	cudaMemcpy(ret, ann, size, cudaMemcpyHostToDevice);
	set_genann_pointers_device<<<1, 1>>>(ret);
	return ret;
}

void copy_back_genann_and_print(genann const* d_genann, genann * ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	cudaMemcpy(ann, d_genann, size, cudaMemcpyDeviceToHost);
	set_genann_pointers(ann);
	double *w = ann->weight + (ann->hidden_layers
		? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
		: (0));
	int n = (ann->hidden_layers ? ann->hidden : ann->inputs) + 1;
	double *d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
	/*
	for (int i = 0; i < ann->outputs; i++) {
		printf("weight: %lf\n", w[i]);
	}
	double *o = ann->output + ann->inputs;
	for (int i = 0; i < ann->hidden; i++) {
		printf("hidden output: %lf\n", o[i]);
	}
	printf("output : %lf\n", o[ann->hidden]);*/
}

void genann_randomize(genann *ann) {
	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		double r = GENANN_RANDOM();
		/* Sets weights from -0.5 to 0.5. */
		ann->weight[i] = r - 0.5;
	}
}


void genann_free(genann *ann) {
	/* The weight, output, and delta pointers go to the same buffer. */
	free(ann);
}

// first output
__device__ double *d_o;
// first delta
__device__ double *d_d;
// first delta of the next layer
__device__ double *d_dd;
// first output in the previous layer
__device__ double *d_i;
// first weight to first output delta
__device__ double *d_w;
// fisrt weight in the following layer
__device__ double *d_ww;
__device__ double *d_t;

/* calculate the addresses for hidden layers in forward run */
__global__ void calculate_address_hidden_forward(genann const *d_genann, int h) {
	d_w = d_genann->weight;
	d_o = d_genann->output + d_genann->inputs;
	d_i = d_genann->output;

	for (int i = 1; i <= h; i++) {
		d_w += d_genann->hidden * (i == 1 ? d_genann->inputs+1 : d_genann->hidden+1);
	}
	d_o += h * d_genann->hidden;
	for (int i = 1; i <= h; i++) {
		d_i += i == 1 ? d_genann->inputs : d_genann->hidden;
	}
}

/* do forward run for hidden layers */
__global__ void genann_run_hidden(genann const *d_genann, int h) {
	int j = threadIdx.x;
	int n = (h == 0 ? d_genann->inputs : d_genann->hidden);
	double sum = d_w[(n+1)*j] * -1.0;

	for (int k = 0; k < n; ++k) {
		sum += d_w[(n+1)*j+k+1] * d_i[k];
	}
	d_o[j] = genann_act_sigmoid_device(sum);
}   

/* run for the output layer */
__global__ void genann_run_output(genann const *d_genann) {
	int j = threadIdx.x;
	int n = d_genann->hidden_layers ? d_genann->hidden : d_genann->inputs;

	/* Figure output layer. */
	double sum = d_w[j * (n+1)] * -1.0;
	for (int k = 0; k < n; ++k) {
		sum += d_w[j * (n+1) + k + 1] * d_i[k];
	}
	d_o[j] = genann_act_sigmoid_device(sum);
}

/* internal forward run 
 * returns the genann on GPU memory
 */
genann* genann_run_internal(genann *ann, double const *inputs) {
	/* Copy the inputs to the scratch area, where we also store each neuron's
	* output, for consistency. This way the first layer isn't a special case. */
	memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

	/* copy to device to run on GPU */
	genann *d_genann = genann_device_copy(ann);

	int h;

	/* Figure hidden layers, if any. */
	for (h = 0; h < ann->hidden_layers; ++h) {
		calculate_address_hidden_forward << <1, 1 >> > (d_genann, h);
		genann_run_hidden << <1, ann->hidden >> > (d_genann, h);
	}

	calculate_address_hidden_forward << <1, 1 >> > (d_genann, ann->hidden_layers);
	genann_run_output << <1, ann->outputs >> > (d_genann);
	copy_back_genann_and_print(d_genann, ann);

	return d_genann;
}

/* external API for running genann forward */
double const *genann_run(genann *ann, double const *inputs) {
	genann *d_genann = genann_run_internal(ann, inputs);
	cudaFree(d_genann);
	return ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
}

__global__ void calculate_device_addresses_output_delta(genann const *d_genann, double *d_desired_outputs) {
	d_o = d_genann->output + d_genann->inputs + d_genann->hidden * d_genann->hidden_layers; /* First output. */
	d_d = d_genann->delta + d_genann->hidden * d_genann->hidden_layers; /* First delta. */
	d_t = d_desired_outputs; /* First desired output. */
}

__global__ void calculate_device_addresses_hidden_delta(genann const *d_genann, int h) {
	d_o = d_genann->output + d_genann->inputs + (h * d_genann->hidden);
	d_d = d_genann->delta + (h * d_genann->hidden);
	d_dd = d_genann->delta + ((h + 1) * d_genann->hidden);
	d_ww = d_genann->weight + ((d_genann->inputs + 1) * d_genann->hidden) + ((d_genann->hidden + 1) * d_genann->hidden * (h));
}

__global__ void calculate_device_addresses_train_outputs(genann const *d_genann) {
	/* Find first output delta. */
	d_d = d_genann->delta + d_genann->hidden * d_genann->hidden_layers;

	/* Find first weight to first output delta. */
	d_w = d_genann->weight + (d_genann->hidden_layers
		? ((d_genann->inputs + 1) * d_genann->hidden + (d_genann->hidden + 1) * d_genann->hidden * (d_genann->hidden_layers - 1))
		: (0));

	/* Find first output in previous layer. */
	d_i = d_genann->output + (d_genann->hidden_layers
		? (d_genann->inputs + (d_genann->hidden) * (d_genann->hidden_layers - 1))
		: 0);
}

/* calculate the device addresses for training hidden layer
 * h is the index of the hidden layer
 */
__global__ void calculate_device_addresses_train_hidden(genann const *d_genann, int h) {
	/* Find first delta in this layer. */
	d_d = d_genann->delta + (h * d_genann->hidden);

	/* Find first input to this layer. */
	d_i = d_genann->output + (h
		? (d_genann->inputs + d_genann->hidden * (h - 1))
		: 0);

	/* Find first weight to this layer. */
	d_w = d_genann->weight + (h
		? ((d_genann->inputs + 1) * d_genann->hidden + (d_genann->hidden + 1) * (d_genann->hidden) * (h - 1))
		: 0);
}

/* Kernel for calculating output layer deltas*/
__global__ void calculate_output_layer_deltas(genann *d_genann) {
	int i = threadIdx.x;
	d_d[i] = (d_t[i] - d_o[i]) * d_o[i] * (1.0 - d_o[i]);
}

/* calcualte the deltas of the hidden layer */
__global__ void calc_hidden_delta(genann const *d_genann, int h) {
	double delta = 0;
	int j = threadIdx.x;

	for (int k = 0; k < (h == d_genann->hidden_layers - 1 ? d_genann->outputs : d_genann->hidden); ++k) {
		const double forward_delta = d_dd[k];
		const int windex = k * (d_genann->hidden + 1) + (j + 1);
		const double forward_weight = d_ww[windex];
		delta += forward_delta * forward_weight;
	}

	d_d[j] = d_o[j] * (1.0 - d_o[j]) * delta;
	__syncthreads();
}

/* train for the weights of the output layer */
__global__ void train_output_weights(genann const *d_genann, double learning_rate) {
	int j = threadIdx.x;
	int n = (d_genann->hidden_layers ? d_genann->hidden : d_genann->inputs) + 1;
	for (int k = 0; k < n; ++k) {
		if (k == 0) {
			d_w[n*j+k] += d_d[j] * learning_rate * -1.0;
		}
		else {
			d_w[n*j + k] += d_d[j] * learning_rate * d_i[k - 1];
		}
	}
}

__global__ void train_hidden_weights(genann const *d_genann, int h, double learning_rate) {
	int j = threadIdx.x;
	int n = (h == 0 ? d_genann->inputs : d_genann->hidden) + 1;
	for (int k = 0; k < n; ++k) {
		if (k == 0) {
			d_w[n*j+k] += d_d[j] * learning_rate * -1.0;
		}
		else {
			d_w[n*j + k] += d_d[j] * learning_rate * d_i[k - 1];
		}
	}
}


/* impl of the external API genann_train */
void genann_train(genann *ann, double *inputs, double *desired_outputs, double learning_rate) {
	genann *d_genann = genann_run_internal(ann, inputs);

	double *d_desired_outputs;
	cudaMalloc((void **)&d_desired_outputs, sizeof(double) * ann->outputs);
	cudaMemcpy(d_desired_outputs, desired_outputs, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);

	int h;

	/* First set the output layer deltas. */
	calculate_device_addresses_output_delta << <1, 1 >> > (d_genann, d_desired_outputs);

	calculate_output_layer_deltas<<<1, ann->outputs>>>(d_genann);

	/* Set hidden layer deltas, start on last layer and work backwards. */
	/* Note that loop is skipped in the case of hidden_layers == 0. */
	for (h = ann->hidden_layers - 1; h >= 0; --h) {
		/* Find first output and delta in this layer. */
		calculate_device_addresses_hidden_delta << <1, 1 >> > (d_genann, h);

		/*  CALL TO KERNEL FOR SETTING HIDDEN LAYER DELTAS*/
		calc_hidden_delta << <1, ann->hidden >> > (d_genann, h);
	}

	/* Train the outputs. */
	{
		/* calculate the device addresses */
		calculate_device_addresses_train_outputs << <1, 1 >> > (d_genann);
		
		/* Set output layer weights. */
		train_output_weights << <1, ann->outputs >> > (d_genann, learning_rate);
	}


	/* Train the hidden layers. */
	for (h = ann->hidden_layers - 1; h >= 0; --h) {
		calculate_device_addresses_train_hidden << <1, 1 >> > (d_genann, h);
		
		train_hidden_weights << <1, ann->hidden >> > (d_genann, h, learning_rate);
	}

	copy_back_genann_and_print(d_genann, ann);
	cudaFree(d_genann);
	cudaFree(d_desired_outputs);
}


void genann_write(genann const *ann, FILE *out) {
	fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		fprintf(out, " %.20e", ann->weight[i]);
	}
}
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

__global__ void set_genann_pointers_device(genann *d_ret) {
	/* Set pointers. */
	d_ret->weight = (double*)((char*)d_ret + sizeof(genann));
	d_ret->output = d_ret->weight + d_ret->total_weights;
	d_ret->delta = d_ret->output + d_ret->total_neurons;
}

genann *genann_copy(genann const *ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	genann *ret = (genann*)malloc(size);
	if (!ret) return 0;

	memcpy(ret, ann, size);
	set_genann_pointers(ret);
	return ret;
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


double const *genann_run(genann const *ann, double const *inputs) {
	double const *w = ann->weight;
	double *o = ann->output + ann->inputs;
	double const *i = ann->output;

	/* Copy the inputs to the scratch area, where we also store each neuron's
	 * output, for consistency. This way the first layer isn't a special case. */
	memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

	int h, j, k;

	const genann_actfun act = ann->activation_hidden;
	const genann_actfun acto = ann->activation_output;

	/* Figure hidden layers, if any. */
	for (h = 0; h < ann->hidden_layers; ++h) {
		for (j = 0; j < ann->hidden; ++j) {
			double sum = *w++ * -1.0;
			for (k = 0; k < (h == 0 ? ann->inputs : ann->hidden); ++k) {
				sum += *w++ * i[k];
			}
			*o++ = act(sum);
		}


		i += (h == 0 ? ann->inputs : ann->hidden);
	}

	double const *ret = o;

	/* Figure output layer. */
	for (j = 0; j < ann->outputs; ++j) {
		double sum = *w++ * -1.0;
		for (k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs); ++k) {
			sum += *w++ * i[k];
		}
		*o++ = acto(sum);
	}

	/* Sanity check that we used all weights and wrote all outputs. */
	assert(w - ann->weight == ann->total_weights);
	assert(o - ann->output == ann->total_neurons);

	return ret;
}


/* Kernel for calculating output layer deltas*/
__global__ void calculate_output_layer_deltas(genann *d_genann, double const *d_desired_outputs) {
	double const *o = d_genann->output + d_genann->inputs + d_genann->hidden * d_genann->hidden_layers; /* First output. */
	double *d = d_genann->delta + d_genann->hidden * d_genann->hidden_layers; /* First delta. */
	double const *t = d_desired_outputs; /* First desired output. */

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < d_genann->outputs) {
		d[i] = (t[i] - o[i]) * o[i] * (1.0 - o[i]);
	}

}

__device__ double *d_o;
__device__ double *d_d;
__device__ double *d_dd;
__device__ double *d_ww;

__global__ void calculate_device_addresses_hidden_delta(genann const *d_genann, int h) {
	d_o = d_genann->output + d_genann->inputs + (h * d_genann->hidden);
	d_d = d_genann->delta + (h * d_genann->hidden);
	d_dd = d_genann->delta + ((h + 1) * d_genann->hidden);
	d_ww = d_genann->weight + ((d_genann->inputs + 1) * d_genann->hidden) + ((d_genann->hidden + 1) * d_genann->hidden * (h));
}

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


/* Rachel and Bill are editing this function*/
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
	/* To begin with, we must run the network forward. */
	genann_run(ann, inputs);

	/* copy to device to run on GPU */
	genann *d_genann = genann_device_copy(ann);
	
	double *d_desired_outputs;
	cudaMalloc((void **)&d_desired_outputs, sizeof(double) * ann->outputs);
	cudaMemcpy(d_desired_outputs, desired_outputs, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);

	int h, j, k;

	/* First set the output layer deltas. */
	calculate_output_layer_deltas<<<1, ann->outputs>>>(d_genann, d_desired_outputs);


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
		/* Find first output delta. */
		double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

		/* Find first weight to first output delta. */
		double *w = ann->weight + (ann->hidden_layers
			? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
			: (0));

		/* Find first output in previous layer. */
		double const * const i = ann->output + (ann->hidden_layers
			? (ann->inputs + (ann->hidden) * (ann->hidden_layers - 1))
			: 0);

		/* Set output layer weights. */
		for (j = 0; j < ann->outputs; ++j) {
			for (k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
				if (k == 0) {
					*w++ += *d * learning_rate * -1.0;
				}
				else {
					*w++ += *d * learning_rate * i[k - 1];
				}
			}

			++d;
		}

		assert(w - ann->weight == ann->total_weights);
	}


	/* Train the hidden layers. */
	for (h = ann->hidden_layers - 1; h >= 0; --h) {

		/* Find first delta in this layer. */
		double const *d = ann->delta + (h * ann->hidden);

		/* Find first input to this layer. */
		double const *i = ann->output + (h
			? (ann->inputs + ann->hidden * (h - 1))
			: 0);

		/* Find first weight to this layer. */
		double *w = ann->weight + (h
			? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * (ann->hidden) * (h - 1))
			: 0);


		for (j = 0; j < ann->hidden; ++j) {
			for (k = 0; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
				if (k == 0) {
					*w++ += *d * learning_rate * -1.0;
				}
				else {
					*w++ += *d * learning_rate * i[k - 1];
				}
			}
			++d;
		}

	}

}


void genann_write(genann const *ann, FILE *out) {
	fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		fprintf(out, " %.20e", ann->weight[i]);
	}
}
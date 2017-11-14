#include <stdio.h>
#include <time.h>
#include "genann.h"



double calc_time(struct timespec start, struct timespec end){ //This function comes from a professor for another class
       double start_sec = (double)start.tv_sec*1000000000.0 + (double)start.tv_nsec;
       double end_sec = (double)end.tv_sec*1000000000.0 + (double)end.tv_nsec;
       if(end_sec < start_sec){
           return 0;
	   }
	      else{
	          return end_sec - start_sec;
		    }
}

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
        i = (int)((a-min)/interval+0.5);
	    if (i <= 0) return lookup[0];
	        if (i >= LOOKUP_SIZE) return lookup[LOOKUP_SIZE-1];
		    return lookup[i];
		    }


void genann_randomize(genann *ann) {
    int i;
        for (i = 0; i < ann->total_weights; ++i) {
	        double r = GENANN_RANDOM();
		        /* Sets weights from -0.5 to 0.5. */
			        ann->weight[i] = r - 0.5;
				    }
				    }
				    

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
        if (inputs < 1) return 0;
	    if (outputs < 1) return 0;
	        if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
        const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
	    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
        const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
	    genann *ret = (genann *)malloc(size);
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

__global__
void genann_run(genann const *ann, double const *inputs) {
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
					                sum += *w++ * i[k];}
		*o++ = act(sum);
}
        i += (h == 0 ? ann->inputs : ann->hidden);
	    }

    //double const *ret = o;

    /* Figure output layer. */
        for (j = 0; j < ann->outputs; ++j) {
	        double sum = *w++ * -1.0;
		        for (k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs); ++k) {
			            sum += *w++ * i[k];
				            }
					            *o++ = acto(sum);
						        }}
							//return ret;}
										

int main(void) {
  printf("Testing Ben's genann example\n");
    struct timespec start_time, end_time;
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      genann * ann = genann_init(2,1,2,1);
      const double input[2] = {0,0};
      genann_run<<<500, 500>>>(ann, input);
      clock_gettime(CLOCK_MONOTONIC, &end_time);
      printf("Time is %lf\n", calc_time(start_time, end_time));
      return 0;
  }
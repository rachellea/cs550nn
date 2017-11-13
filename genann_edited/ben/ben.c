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

int main(void) {
  printf("Testing Ben's genann example\n");
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  genann * ann = genann_init(2,1,2,1);
  const double input[2] = {0,0};
  for(int i = 0; i < 1000; i++){
    genann_run(ann, input);
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double time = calc_time(start_time, end_time);
  printf("Time elapsed is %lf\n", time);
  return 0;
}

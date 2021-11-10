/*
  OpenMP implementation to find the closest pairs of waypoints
  in each of a set of METAL TMG graph files.

  An OpenMP team of worker threads is used to parallelize the outer
  loop of the closest pairs computation for each graph (one at a
  time).

  argv[1] is expected to be a parallelization mode within one graph,
  and can be:

  "fine", which would let OpenMP schedule the outer for loop as it
  sees fit, and update the global closest pair after each is computed.

  "coarse" which would break the points up among the threads, have each
  compute its local closest pair, and only agree on the global result
  at the end, analogous to the pthreads version.

  The tasks to complete are to find the closest pair of points in
  METAL TMG files given as command-line parameters in argv[2] through
  argv[argc-1].

  Jim Teresco, Fall 2021
  Siena College
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "timer.h"
#include "tmggraph.h"

/* closest pairs of vertices with OpenMP, with less contention on the
   shared distance variable because updates to the shared result
   variables are done only at the end */
void tmg_closest_pair_omp_coarse(tmg_graph *g, int *v1, int *v2, double *distance) {

  *v1 = -1;
  *v2 = -1;
  *distance = 10000;  // larger than earth diameter

#pragma omp parallel shared(v1, v2, distance, g)
  {
    // local variables to each thread for our local leaders
    int local_v1 = -1;
    int local_v2 = -1;
    double local_distance = 10000;    // larger than earth diameter
    
    int vert1, vert2;
    double this_dist;

    int thread_num = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    // compute the local leaders for our subset of the rows
    for (vert1 = thread_num; vert1 < g->num_vertices - 1;
	 vert1+=num_threads) {
      for (vert2 = vert1 + 1; vert2 < g->num_vertices; vert2++) {
	this_dist = tmg_distance_latlng(&(g->vertices[vert1]->w.coords),
					&(g->vertices[vert2]->w.coords));
	if (this_dist < local_distance) {
	  local_distance = this_dist;
	  local_v1 = vert1;
	  local_v2 = vert2;
	}
      }
    }
  // contribute our local result to the overall result
#pragma omp critical(mutex)
    if (local_distance < *distance) {
      *distance = local_distance;
      *v1 = local_v1;
      *v2 = local_v2;
    }
  }
}

/* closest pairs of vertices with OpenMP, with potential contention on the shared
   distance variable because of frequent updates to the shared result variables */
void tmg_closest_pair_omp_fine(tmg_graph *g, int *v1, int *v2, double *distance) {

  *v1 = -1;
  *v2 = -1;
  *distance = 10000;  // larger than earth diameter

  int vert1, vert2;
  double this_dist;
#pragma omp parallel for private(vert1,vert2,this_dist) shared(g,distance,v1,v2)
  for (vert1 = 0; vert1 < g->num_vertices - 1; vert1++) {
    for (vert2 = vert1 + 1; vert2 < g->num_vertices; vert2++) {
      this_dist = tmg_distance_latlng(&(g->vertices[vert1]->w.coords),
				      &(g->vertices[vert2]->w.coords));
#pragma omp critical(mutex)
      if (this_dist < *distance) {
	*distance = this_dist;
	*v1 = vert1;
	*v2 = vert2;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  
  // about how many distance calculations?
  long dcalcs = 0;

  int worker_rank;
  int num_tasks;

  int i;

  struct timeval start_time, stop_time;
  double active_time;

  // all parameters except argv[0] (program name) and argv[1]
  // (parallelization strategy) will be filenames to load, so the
  // number of tasks is argc - 2
  num_tasks = argc - 2;
  
  if (argc < 3) {
    fprintf(stderr, "Usage: %s mode filenames\n", argv[0]);
    exit(1);
  }

  if (strcmp("fine", argv[1]) && strcmp("coarse", argv[1])) {
    fprintf(stderr, "Invalid mode %s\n", argv[1]);
    fprintf(stderr, "Usage: %s mode filenames\n", argv[0]);
    exit(1);
  }

  printf("Have %d tasks to be done\n", num_tasks);
  
  // start the timer
  gettimeofday(&start_time, NULL);
  
  // go through the files
  for (int task_pos = 2; task_pos < argc; task_pos++) {
    gettimeofday(&stop_time, NULL);
    active_time = diffgettime(start_time, stop_time);
    printf("Starting task %d %s at elapsed time %.6f\n", (task_pos-1),
	   argv[task_pos], active_time);

    tmg_graph *g = tmg_load_graph(argv[task_pos]);
    if (g == NULL) {
      fprintf(stderr, "Could not create graph from file %s, SKIPPING\n",
	      argv[task_pos]);
      continue;
    }
    
    int v1, v2;
    double distance;
    
    // do it
    if (strcmp(argv[1], "fine") == 0) {
      tmg_closest_pair_omp_fine(g, &v1, &v2, &distance);
    }
    else {
      tmg_closest_pair_omp_coarse(g, &v1, &v2, &distance);
    }    
    
    printf("%s closest pair #%d %s (%.6f,%.6f) and #%d %s (%.6f,%.6f) distance %.15f\n",
	   argv[task_pos], v1, g->vertices[v1]->w.label,
	   g->vertices[v1]->w.coords.lat, g->vertices[v1]->w.coords.lng,
	   v2, g->vertices[v2]->w.label,
	   g->vertices[v2]->w.coords.lat, g->vertices[v2]->w.coords.lng,
	   distance);
    
    tmg_graph_destroy(g);
  }
  
  // get main thread's elapsed time
  gettimeofday(&stop_time, NULL);
  active_time = diffgettime(start_time, stop_time);


  printf("Main thread was active for %.4f seconds\n", active_time);
  return 0;
}

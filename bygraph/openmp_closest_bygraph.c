/*
  Bag of Tasks OpenMP implementation to find the closest pairs of waypoints
  in each of a set of METAL TMG graph files.

  The tasks to complete are to find the closest pair of points in
  METAL TMG files given as command-line parameters in argv[2] through
  argv[argc-1].

  The tasks are distributed in an order based on the string passed as
  argv[1], which is one of:

      "orig": the order that the files are presented on the command line
      "alpha": alphabetical order by filename
      "size": from largest to smallest number of points in the file
      "random": randomized order

  Jim Teresco, Fall 2021
  Siena College
*/

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "timer.h"
#include "tmggraph.h"

// struct to encapsulate info about the tasks in the bag
typedef struct cptask {
  int num_vertices;
  char *filename;
} cptask;

// helper function to read only up to the number of vertices from a
// TMG file and return that number
int read_tmg_vertex_count(char *filename) {

  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Cannot open file %s for reading.\n", filename);
    exit(1);
  }

  // read over first line
  char temp[100];
  fscanf(fp, "%s %s %s", temp, temp, temp);
  
  // read number of vertices
  int nv;
  fscanf(fp, "%d", &nv);

  // that's all we need for now
  fclose(fp);

  return nv;
}

int main(int argc, char *argv[]) {
  
  int num_threads;
  
  // about how many distance calculations?
  long dcalcs = 0;

  int worker_rank;
  int num_tasks;

  int i;

  struct timeval start_time, stop_time;
  double active_time;

  // all parameters except argv[0] (program name) and argv[1] (input
  // ordering) will be filenames to load, so the number of tasks is
  // argc - 2
  num_tasks = argc - 2;
  
  if (argc < 3) {
    fprintf(stderr, "Usage: %s orig|alpha|size|random filenames\n", argv[0]);
    exit(1);
  }

  // check for a valid ordering in argv[1];
  char *orderings[] = {
    "orig",
    "alpha",
    "size",
    "random"
  };
  int ordering = -1;
  for (i = 0; i < 4; i++) {
    if (strcmp(argv[1], orderings[i]) == 0) {
      ordering = i;
      break;
    }
  }
  if (ordering == -1) {
    fprintf(stderr, "Usage: %s orig|alpha|size|random filenames\n", argv[0]);
    exit(1);
  }      

  printf("Have %d tasks to be done\n", num_tasks);
  
  // start the timer
  gettimeofday(&start_time, NULL);
  
  // allocate and populate our "bag of tasks" array
  cptask **tasks = (cptask **)malloc(num_tasks*sizeof(cptask *));
  
  // add the first at pos 0, since we know there's at least one and
  // this will eliminate some special cases in our code below.
  tasks[0] = (cptask *)malloc(sizeof(cptask));
  tasks[0]->filename = argv[2];
  if (ordering == 2) {
    tasks[0]->num_vertices = read_tmg_vertex_count(argv[3]);
  }
  
  // get them all in
  for (i = 1; i < num_tasks; i++) {
    cptask *taski = (cptask *)malloc(sizeof(cptask));
    taski->filename = argv[i+2];
    int pos = i;
    int insertat;
    switch (ordering) {
      
    case 0:
      // original ordering as specified by argv
      tasks[i] = taski;
      break;
      
      
    case 1:
      // alphabetical order by filename
      while (pos > 0 && strcmp(taski->filename, tasks[pos-1]->filename) < 0) {
	tasks[pos] = tasks[pos-1];
	pos--;
      }
      tasks[pos] = taski;
      
      break;
      
    case 2:
      // order by size largest to smallest number of vertices
      taski->num_vertices = read_tmg_vertex_count(taski->filename);
      while (pos > 0 && taski->num_vertices >= tasks[pos-1]->num_vertices) {
	tasks[pos] = tasks[pos-1];
	pos--;
      }
      tasks[pos] = taski;
      
      break;
      
    case 3:
      // order randomly
      insertat = random()%(pos+1);
      while (pos > insertat) {
	tasks[pos] = tasks[pos-1];
	pos--;
      }
      tasks[pos] = taski;
      break;
    }
  }

  // for thread stats
  int minjobs = num_tasks+1;
  int maxjobs = 0;
  long mincalcs = LONG_MAX;
  long maxcalcs = 0L;
  long totalcalcs = 0;
  double mintime = DBL_MAX;
  double maxtime = 0.0;
  
  // what's the next task available in the bag of tasks (index into array)
  int next_task = 0;
  
#pragma omp parallel shared(tasks, next_task, minjobs, maxjobs, mincalcs, maxcalcs, totalcalcs, mintime, maxtime, num_threads)
  {
    struct timeval start_time, stop_time;

    // start the timer
    gettimeofday(&start_time, NULL);

    int my_task = -1;
    int jobs_done = 0;
    long dcalcs = 0L;
    int thread_num = omp_get_thread_num();
    num_threads = omp_get_num_threads();

    while (1) {
      
      // grab a task from the bag
#pragma omp critical(mutex)
      my_task = next_task++;
      
      if (my_task >= num_tasks) break;
      
      // this thread can process this one
      printf("[%d] working on %s\n", thread_num, tasks[my_task]->filename);
      tmg_graph *g = tmg_load_graph(tasks[my_task]->filename);
      if (g == NULL) {
	fprintf(stderr, "Could not create graph from file %s\n",
		tasks[my_task]->filename);
	exit(1);
      }
    
      int v1, v2;
      double distance;
      
      // do it
      tmg_closest_pair(g, &v1, &v2, &distance);
      
      jobs_done++;
      long job_calcs = g->num_vertices;
      job_calcs *= g->num_vertices;
      job_calcs /= 2;
      dcalcs += job_calcs;
      
      printf("[%d] %s closest pair #%d %s (%.6f,%.6f) and #%d %s (%.6f,%.6f) distance %.15f\n",
	     thread_num, tasks[my_task]->filename, v1,
	     g->vertices[v1]->w.label,
	     g->vertices[v1]->w.coords.lat, g->vertices[v1]->w.coords.lng,
	     v2, g->vertices[v2]->w.label,
	     g->vertices[v2]->w.coords.lat, g->vertices[v2]->w.coords.lng,
	     distance);
      
      tmg_graph_destroy(g);
    }
    
    gettimeofday(&stop_time, NULL);

    double thread_elapsed_time = diffgettime(start_time, stop_time);

    // separate critical section for accumulation and update of
    // simulation stats
#pragma omp critical(stats)
    {
    if (jobs_done < minjobs)
      minjobs = jobs_done;
    if (jobs_done > maxjobs)
      maxjobs = jobs_done;
    if (dcalcs < mincalcs)
      mincalcs = dcalcs;
    if (dcalcs > maxcalcs)
      maxcalcs = dcalcs;
    totalcalcs += dcalcs;
    if (thread_elapsed_time < mintime)
      mintime = thread_elapsed_time;
    if (thread_elapsed_time > maxtime)
      maxtime = thread_elapsed_time;
    }
    printf("[%d] terminating\n", thread_num);
  }

  // get main thread's elapsed time
  gettimeofday(&stop_time, NULL);
  active_time = diffgettime(start_time, stop_time);

  double avgjobs = 1.0*num_tasks/num_threads;

  printf("Main thread was active for %.4f seconds\n", active_time);
  printf("%d workers processed %d jobs with about %ld distance calculations\n",
	 num_threads, num_tasks, totalcalcs);
  printf("Job balance: min %d, max %d, avg: %.2f\n", minjobs, maxjobs,
	 avgjobs);
  printf("Distance calculation balance: min %ld, max %ld, avg: %.2f\n",
	 mincalcs, maxcalcs, ((1.0*totalcalcs)/num_threads));
  printf("Active time balance: min %.4f, max %.4f\n", mintime, maxtime);

  for (i = 0; i < num_tasks; i++) {
    free(tasks[i]);
  }
  free(tasks);
  return 0;
}

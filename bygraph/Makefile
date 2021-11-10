# Makefile for C program to read and process a TMG file

PROGRAM=openmp_closest_bygraph
CFILES=tmggraph.c sll.c timer.c $(PROGRAM).c
OFILES=$(CFILES:.c=.o)
CC=gcc
CFLAGS=-fopenmp

$(PROGRAM):	$(OFILES)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OFILES) -lreadline -lm

clean::
	/bin/rm $(PROGRAM) $(OFILES)

# Makefile for C program to read and process a TMG file

PROGRAM=openmp_closest_byvertex
CFILES=tmggraph.c sll.c timer.c $(PROGRAM).c
OFILES=$(CFILES:.c=.o)
CC=clang
CFLAGS=-fopenmp=libomp

$(PROGRAM):	$(OFILES)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OFILES) -lm

clean::
	/bin/rm $(PROGRAM) $(OFILES)

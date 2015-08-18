#
# Requirement: Intel compiler and Intel MKL library
#

CC = icc  
CFLAGS=-mkl -Wall -O2 -g

TARGETS=mpi-mkldgemm

all: $(TARGETS)

mpi-mkldgemm: mpi-mkldgemm.c
	mpiicc  -mkl=sequential  -Wall -O2  -g -o $@ $<

clean:
	rm -f $(TARGETS)

distclean: clean
	rm -f *~




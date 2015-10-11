#
# Requirement: Intel compiler and Intel MKL library
#

CC=icc
MPICC=mpiicc
CFLAGS=-mkl -Wall -O2 -g
DOCOPT=python docopt_c.py

TARGETS=mpi-mkldgemm mpidgemmio

all: $(TARGETS)

mpi-mkldgemm: mpi-mkldgemm.c
	$(MPICC) $(CFLAGS) -o $@ $<

mpidgemmio: mpidgemmio.c docopt.c
	$(MPICC) $(CFLAGS) -o $@ $<

docopt.c: mpidgemmio.docopt
	$(DOCOPT) $< -o $@

clean:
	rm -f $(TARGETS)

distclean: clean
	rm -f *~




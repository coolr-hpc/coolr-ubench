#
# Requirement: Intel compiler and Intel MKL library
#

CC=icc
MPICC=mpiicc
CFLAGS=-mkl -Wall -O2 -g
DOCOPT=python docopt_c.py

TARGETS=mpi-mkldgemm mpidgemmio hpbench mpi-hpbench

all: $(TARGETS)

mpi-mkldgemm: mpi-mkldgemm.c
	$(MPICC) $(CFLAGS) -o $@ $<

mpidgemmio: mpidgemmio.c docopt.c
	$(MPICC) $(CFLAGS) -o $@ $<

docopt.c: mpidgemmio.docopt
	$(DOCOPT) $< -o $@

hpbench : hpbench.c raplreader_qh.c
	$(CC) -O3 -axCORE-AVX2 -Wall -o $@ $^

mpi-hpbench : hpbench.c raplreader_qh.c
	$(MPICC) -O3 -axCORE-AVX2 -Wall -DENABLE_MPI -o $@ $^

clean:
	rm -f $(TARGETS)

distclean: clean
	rm -f *~




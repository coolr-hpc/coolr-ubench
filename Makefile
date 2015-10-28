#
# Requirement: Intel compiler and Intel MKL library
#

CC=icc
MPICC=mpiicc

RAPLREADER_DIR=../intercoolr/
RAPLREADER_DIR=../intercoolr/

CFLAGS=-mkl -Wall -O2 -g -I$(RAPLREADER_DIR)
LDFLAGS=-L$(RAPLREADER_DIR) -lintercoolr

DOCOPT=python docopt_c.py

TARGETS=mpi-mkldgemm mpidgemmio hpbench mpi-hpbench

all: $(TARGETS)

mpi-mkldgemm: mpi-mkldgemm.c
	$(MPICC) $(CFLAGS) -o $@ $<

mpidgemmio: mpidgemmio.c docopt.c
	$(MPICC) $(CFLAGS) -o $@ $<

docopt.c: mpidgemmio.docopt
	$(DOCOPT) $< -o $@

hpbench : hpbench.c
	$(CC) $(CFLAGS) -axCORE-AVX2 -o $@ $^ $(LDFLAGS)

mpi-hpbench : hpbench.c
	$(MPICC) $(CFLAGS) -axCORE-AVX2  -DENABLE_MPI -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGETS)

distclean: clean
	rm -f *~




#define _GNU_SOURCE
#include <sched.h>

#include <mpi.h>
#include <mkl.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>


static int size;
static int rank;

/* common parameters */
#define N_INNER  (400)
#define N_OUTTER (2000)

static double alpha = 1.0;
static double beta = 1.0;

#define M_M  (384)
#define M_K  (384)
#define M_N  (384)
static int m_m = M_M;
static int m_k = M_K;
static int m_n = M_N;

static char transa = 'N';
static char transb = 'N';

static double flopsval = (2.0*M_M*M_K*M_N + M_K*M_N)*(double)N_INNER;

struct perrank_data {
	double *A, *B, *C;
	int  cpuid;
	double elapsedtime;
	double gflops;
};

static struct perrank_data  prd;

static void initprd(void)
{
	int i;

	prd.A = calloc(m_m*m_k, sizeof(double));
	prd.B = calloc(m_k*m_n, sizeof(double));
	prd.C = calloc(m_m*m_n, sizeof(double));
	assert(prd.A);
	assert(prd.B);
	assert(prd.C);
	prd.cpuid = sched_getcpu();
	prd.elapsedtime = 0.0;
}


static void dosomething(int rank, int seq_rank)
{
	int i;
	double t1, t2;

	if (seq_rank == -1 ||
	    (seq_rank >= 0 && seq_rank == rank)) {
		for (i = 0; i < m_m*m_k ; ++i)
			prd.A[i] = 1.0;
		for (i = 0; i < m_k*m_n ; ++i)
			prd.B[i] = 2.0;
		for (i = 0; i < m_m*m_n ; ++i)
			prd.C[i] = 1.0;

		t1 = MPI_Wtime();
		for (i = 0; i < N_INNER; i++) {
			dgemm(&transa, &transb, &m_m, &m_n, &m_k, &alpha,
			      prd.A, &m_m, prd.B, &m_k, &beta,
			      prd.C, &m_m);
		}
		t2 = MPI_Wtime();
		prd.elapsedtime = t2-t1;
		prd.gflops = (flopsval * 1e-9)/prd.elapsedtime;
	} else {
		prd.gflops = 0.0;
		//sleep(1);
	}
}


void set_strict_affinity(int size, int rank)
{
	cpu_set_t  cpuset_mask;

	CPU_ZERO(&cpuset_mask);
	CPU_SET(rank, &cpuset_mask);
	if (sched_setaffinity(0, sizeof(cpuset_mask), &cpuset_mask) == -1) {
		printf("sched_setaffinity() failed\n");
		exit(1);
	}
}

static double getuptime(void)
{
	FILE *fp;
	double ret;

	fp = fopen("/proc/uptime", "r");
	if (fp) {
		char buf[80];
		int i, l;

		fgets(buf, sizeof(buf), fp);
		l = strlen(buf);
		if (l > 0) {
			for (i = 0; i < l; i++) {
				if (buf[i] == ' ') {
					buf[i] = 0;
					break;
				}
			}
			ret = atof(buf);
		}
	}
	return ret;
}

static void  usage(const char *prog)
{
	printf("Usage: %s [options]\n", prog);
	printf("\n");
	printf("[options]\n");
	printf("\n");
	printf("-t int : timeout [sec]\n");
	printf("\n");
}

int main(int argc, char *argv[])
{
	int i;
	double *rbuf;
	FILE *fp = stdout;
	char fn[BUFSIZ];
	/* for an output in the json format */
	char strbuf[BUFSIZ];
	char *ptr;
	int pos, remain;
	int len;
	double timeout_sec = 600.0, st;
	int opt;
	/* If seq_rank >=0, only seq_rank does a real work
	 * other ranks do nothing; waiting until the next sync point
	 */
	int seq_rank = -1;
	int cnt = 0;

	fn[0] = 0;

	MPI_Init(NULL, NULL);

	while ((opt = getopt(argc, argv, "ht:s:o:")) != -1) {
		switch (opt) {
		case 'h':
			usage(argv[0]);
			exit(0);
		case 't':
			timeout_sec = atoi(optarg);
			break;
		case 's':
			seq_rank = atoi(optarg);
			break;
		case 'o':
			snprintf(fn, BUFSIZ, "%s", optarg);
			break;
		}
	}


	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		printf("# timeout_sec=%lf\n", timeout_sec);
		printf("# mpisize=%d\n", size);

		if (strlen(fn) > 0) {
			fp = fopen(fn, "w");
			if (!fp) {
				fprintf(stderr, "Unable to open %s", fn);
				exit(1);
			}
		}
	}

	set_strict_affinity(size, rank);
	if (rank == 0)
		printf("# set affinity\n");

	if (rank == 0) {
		rbuf = (double *)malloc(size*sizeof(double));
		assert(rbuf);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	initprd();
	st = MPI_Wtime();

	while (1) {
		MPI_Barrier(MPI_COMM_WORLD);

		if ((MPI_Wtime()-st) >= timeout_sec)
			break;

		dosomething(rank, seq_rank);

		MPI_Gather(&prd.gflops, 1, MPI_DOUBLE,
			   rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


		if (rank == 0) {
			int j;
			double agg;

			ptr = strbuf;
			remain = BUFSIZ - 1;

			snprintf(ptr, remain, "{\"sample\":\"dgemm\",\"time\":%.2lf",
				 MPI_Wtime());
			len = strlen(ptr);
			ptr += len;
			remain -= len;

			snprintf(ptr, remain, ", \"mpisize\":%d ", size);
			len = strlen(ptr);
			ptr += len;
			remain -= len;

			/* */
			agg = 0.0;
			for (j = 0; j < size; j++) {
				agg += rbuf[j];
				snprintf(ptr, remain, ", \"dgemm%d\":%.3lf ",
					 j, rbuf[j]);
				len = strlen(ptr);
				ptr += len;
				remain -= len;
			}

			snprintf(ptr, remain, ", \"dgemm_agg\":%.3lf ", agg);
			len = strlen(ptr);
			ptr += len;
			remain -= len;

			snprintf(ptr, remain, "}");
			fprintf(fp, "%s\n", strbuf);
			fflush(fp);
			if (0) {
				FILE *fp;

				fp = fopen("dgemmlast.log", "w");
				if (fp) {
					fprintf(fp, "%s\n", strbuf);
					fflush(fp);
					fclose(fp);
				}
			}
		}
	}


	if (rank == 0) {
		if (fp != stdout)
			fclose(fp);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

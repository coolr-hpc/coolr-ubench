#define _GNU_SOURCE
#include <sched.h>

#include <mpi.h>
#include <mkl.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <time.h>

#include "docopt.c"
#define VERSION "1.0"

static int size;
static int rank;
static int cpus;
static int ioboost;
static int mode_read;

#define BLOCKSIZE   (4096)
static unsigned int inner, outer;
static unsigned long blocks;
static char *tmpdir;
static char *affinity = NULL;
#define POLICY_RANK         "rank"
#define POLICY_RANK_INV     "rank_inv"
#define POLICY_NONE         "none"
#define POLICY_RANDOM       "random"
#define POLICY_IOLOWTEMP    "iolowtemp"
#define POLICY_COMPLOWTEMP  "complowtemp"
#define POLICY_INTERLEAVE   "interleave"

static double alpha = 1.0;
static double beta = 1.0;

#define M_M  (384)
#define M_K  (384)
#define M_N  (384)
static int m_m = M_M;
static int m_k = M_K;
static int m_n = M_N;

static char transa='N';
static char transb='N';

static double flopsval;

struct perrank_data
{
    double *A, *B, *C;
    int  cpuid;
    double elapsedtime;
    double gflops;
    int f;
};

static struct perrank_data  prd;

static void initprd_io(rank)
{
    int f;
    prd.cpuid = sched_getcpu();
    prd.elapsedtime = 0.0;

    if (ioboost) {
        f = open("/sys/devices/system/cpu/intel_pstate/task_boost", O_WRONLY);
        if (f < 0) {
            printf("cannot open /sys/devices/system/cpu/intel_pstate/task_boost\n");
            exit(0);
        }
        write(f, "1\n", 2);
        close(f);
    }
}

static void cleanprd_io(int rank)
{
    int f;
    if (ioboost) {
        f = open("/sys/devices/system/cpu/intel_pstate/task_boost", O_WRONLY);
        if (f < 0) {
            printf("cannot open /sys/devices/system/cpu/intel_pstate/task_boost\n");
            exit(0);
        }
        write(f, "0\n", 2);
        close(f);
    }
}

static void cleanprd_compute(int rank)
{
}

static void initprd_compute(int rank)
{
    prd.A = calloc( m_m*m_k, sizeof(double) );
    prd.B = calloc( m_k*m_n, sizeof(double) );
    prd.C = calloc( m_m*m_n, sizeof(double) );
    assert(prd.A);
    assert(prd.B);
    assert(prd.C);
    prd.cpuid = sched_getcpu();
    prd.elapsedtime = 0.0;
}

static int is_io(int rank, int num_io)
{
    return (rank < num_io);
}

static void initprd(int rank, int num_io)
{
    if (is_io(rank, num_io))
        return initprd_io(rank);
    else
        return initprd_compute(rank);
}

static void cleanprd(int rank, int num_io)
{
    if (is_io(rank, num_io))
        return cleanprd_io(rank);
    else
        return cleanprd_compute(rank);
}

static void dosomething_io(int rank)
{
    char buf[BLOCKSIZE];
    char path[200];
    int f;
    ssize_t i;
    double t1, t2;

    sprintf(path, "%s/io.%d", tmpdir, rank);
    if (mode_read) {
        f = open(path, O_RDONLY);
        if (f < 0) {
            perror("open O_RDONLY");
            exit(0);
        }
    } else {
        f = open(path, O_WRONLY | O_CREAT, 0666);
        if (f < 0) {
            perror("open O_WRONLY");
            exit(0);
        }
    }

    t1 = MPI_Wtime();
    if (mode_read) {
        for (i=0; i<blocks; i++) {
            int r = read(f, buf, BLOCKSIZE);
            if (r != BLOCKSIZE) {
                perror("read()");
                exit(0);
            }
        }
    } else {
        for (i=0; i<blocks; i++) {
            int r = write(f, buf, BLOCKSIZE);
            if (r != BLOCKSIZE) {
                perror("write()");
                exit(0);
            }
        }
        fsync(f);
    }
    t2 = MPI_Wtime();

    prd.elapsedtime = t2-t1;
    close(f);
}

static void dosomething_compute(int rank)
{
    int i;
    double t1,t2;

    for (i=0; i<m_m*m_k ; ++i) prd.A[i] = 1.0;
    for (i=0; i<m_k*m_n ; ++i) prd.B[i] = 2.0;
    for (i=0; i<m_m*m_n ; ++i) prd.C[i] = 1.0;

    t1 = MPI_Wtime();
    for(i=0; i<inner; i ++ ) { 
        dgemm(&transa, &transb, &m_m, &m_n, &m_k, &alpha, 
                prd.A, &m_m, prd.B, &m_k, &beta, prd.C, &m_m);
    }
    t2 = MPI_Wtime();
    prd.elapsedtime = t2-t1;
    prd.gflops = (flopsval * 1e-9)/prd.elapsedtime;

}

static void dosomething(int rank, int num_io)
{
    if (is_io(rank, num_io))
        return dosomething_io(rank);
    else
        return dosomething_compute(rank);
}

void set_rank_affinity(int size, int rank, int reversed)
{
    cpu_set_t  cpuset_mask;
    int cpu_target;

    CPU_ZERO(&cpuset_mask);
    if (reversed)
        cpu_target = cpus-rank-1;
    else
        cpu_target = rank;
    CPU_SET(cpu_target, &cpuset_mask);
    if ( sched_setaffinity(0, sizeof(cpuset_mask), &cpuset_mask) == -1 ) {
        printf("sched_setaffinity() failed\n");
        exit(1);
    }
    //printf("[%d] on cpu %d\n", rank, cpu_target); 
}

int min(int a, int b)
{
    if (a <= b)
        return a;
    return b;
}

void set_interleave_affinity(int size, int rank, int num_io)
{
    cpu_set_t  cpuset_mask;
    int cpu_target;
    int num_interleave;

    num_interleave = min(size-num_io, num_io);

    CPU_ZERO(&cpuset_mask);
    if (rank < num_interleave) {
        cpu_target = rank + rank + 1;
    } else if ((rank >= num_interleave) && (rank < num_interleave*2)) {
        cpu_target = rank - (num_interleave*2 - rank); 
    } else {
        cpu_target = rank;
    }

    //printf("[%d] on cpu %d\n", rank, cpu_target); 
    CPU_SET(cpu_target, &cpuset_mask);
    if ( sched_setaffinity(0, sizeof(cpuset_mask), &cpuset_mask) == -1 ) {
        printf("sched_setaffinity() failed\n");
        exit(1);
    }
}

//void read_cpu_temp(struct temp_map *tm, unsigned int size)
//{
//    memset((char*)tm->assigned, 0, size);
//}
//
///* find lowest temp that isn't assigned yet. Assign when found */
//unsigned int index_next_lowest_temp(struct temp_map *tm, unsigned int size)
//{
//    int i;
//    unsigned int lowest = UINT_MAX;
//    int lowest_idx = -1;
//
//    for (i=0; i<size; i++) {
//        if (tm->assigned[i])
//            continue;
//        if (tm->temps[i] <= lowest)
//            lowest_idx = i;
//    }
//    assert(lowest_idx >= 0);
//    tm->assigned[i] = 1;
//    return i;
//}
//
//void print_temps(struct temp_map *tm, int size) {
//    int i;
//    for(i=0; i<size; i++) {
//        printf("%d ", tm->temp[i]);
//    }
//    printf("\n");
//}
//
//void set_iolowtemp_affinity(int size, int rank, int numio)
//{
//    struct temp_map tm;
//    unsigned int skip;
//    unsigned int i;
//    unsigned int lowest_index;
//    
//    if (rank == 0) {
//        read_cpu_temp(&tm, size);
//        print_temps(&tm, size);
//    }
//
//    MPI_Bcast((char*)tm, sizeof(tm), MPI_BYTE, 0, MPI_COMM_WORLD);
//
//    /* Rank 0 gets the lowest temp, rank 1 the second lowest, etc.
//     * Each rank gets a copy of the temps, so skip the first rank lowest temps.
//     */
//    skip = rank; /* skip this many cpus */
//    while(skip) {
//        lowest_index = index_next_lowest_temp(&tm, size);
//        skip--;
//    }
//    set_rank_affinity(size, lowest_index, 0);
//    printf("rank %d -> cpu %d\n", rank, lowest_index);
//}

void setup_affinity(char *policy, int size, int rank, int num_io)
{
    if (strncmp(policy, POLICY_RANK, strlen(policy)) == 0) {
        set_rank_affinity(size, rank, 0);
    } else if (strncmp(policy, POLICY_RANK_INV, strlen(policy)) == 0) {
        set_rank_affinity(size, rank, 1);
    } else if (strncmp(policy, POLICY_INTERLEAVE, strlen(policy)) == 0) {
        set_interleave_affinity(size, rank, num_io);
    } else if (strncmp(policy, POLICY_IOLOWTEMP, strlen(policy)) == 0) {
        printf("WARNING: Policy: %s not implemented. Setting to none.\n", policy);
        return; //set_iolowtemp_affinity(size, rank, num_io);
    } else if (strncmp(policy, POLICY_COMPLOWTEMP, strlen(policy)) == 0) {
        printf("WARNING: Policy: %s not implemented. Setting to none.\n", policy);
        return;
    } else if (strncmp(policy, POLICY_RANDOM, strlen(policy)) == 0) {
        printf("WARNING: Policy: %s not implemented. Setting to none.\n", policy);
        return;
    } else if (strncmp(policy, POLICY_NONE, strlen(policy)) == 0) {
        return;
    } else {
        set_rank_affinity(size, rank, 0);
    }
}


static double getuptime(void)
{
    FILE *fp;
    double ret;
    char buf[80];
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    ret = (double)(ts.tv_sec) + (double)(ts.tv_nsec)/(double)1000000000;
    return ret;
}

static unsigned long get_num_cpus(void)
{
    return sysconf(_SC_NPROCESSORS_ONLN);
}

int main(int argc, char *argv[])
{
    int i;
    double *rbuf;
    // for an output in the json format
    char strbuf[BUFSIZ];
    char *ptr;
    int pos, remain;
    int len;

    MPI_Group world_gid, compute_gid = MPI_GROUP_EMPTY, io_gid = MPI_GROUP_EMPTY;
    MPI_Comm compute_comm = MPI_COMM_NULL, io_comm = MPI_COMM_NULL;
    int io_ranges[2][3];
    int compute_ranges[2][3];
    int group_rank, group_size;

    DocoptArgs args = docopt(argc, argv, 1, VERSION);
    int verbose = !args.quiet; /* this seems stupid but it's best for having verbose be the default */
    unsigned int num_io = atoi(args.numio);
    inner = atoi(args.inner);
    outer = atoi(args.outer);
    blocks = atol(args.blocks);
    affinity = args.affinity;
    tmpdir = args.tmpdir;
    ioboost = args.ioboost;
    mode_read = args.read;
    flopsval = (2.0*M_M*M_K*M_N + M_K*M_N)*(double)inner;
    cpus = get_num_cpus();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && verbose)
        printf("total %d io %d compute %d\n", size, num_io, size-num_io);

    if (num_io > 0) {
        io_ranges[0][0] = 0;
        io_ranges[0][1] = num_io-1;
        io_ranges[0][2] = 1;
    }
    if (num_io < size) {
        compute_ranges[0][0] = num_io;
        compute_ranges[0][1] = size-1;
        compute_ranges[0][2] = 1;
    }

    MPI_Comm_group(MPI_COMM_WORLD, &world_gid);

    if (num_io > 0) {
        MPI_Group_range_incl(world_gid, 1, io_ranges, &io_gid);
        MPI_Comm_create(MPI_COMM_WORLD, io_gid, &io_comm);
    }
    if (num_io < size) {
        MPI_Group_range_incl(world_gid, 1, compute_ranges, &compute_gid);
        MPI_Comm_create(MPI_COMM_WORLD, compute_gid, &compute_comm);
    }
    if (MPI_COMM_NULL == compute_comm) {
        MPI_Group_size(io_gid, &group_size);
        MPI_Group_rank(io_gid, &group_rank);
    } else {
        MPI_Group_size(compute_gid, &group_size);
        MPI_Group_rank(compute_gid, &group_rank);
    }

    setup_affinity(args.affinity, size, rank, num_io);

    if(group_rank == 0) {
        rbuf = (double*)malloc(size*sizeof(double));
        assert(rbuf);
    }

    FILE *fio = fopen(args.io_outfile, "w");
    FILE *fcompute = fopen(args.compute_outfile, "w");
    initprd(rank, num_io);
    MPI_Barrier(MPI_COMM_WORLD);

    for(i=0; i<outer; i++ ) {
        dosomething(rank, num_io);
        if (MPI_COMM_NULL != compute_comm) { // compute
            if (args.gflops)
                MPI_Gather(&prd.gflops, 1, MPI_DOUBLE, 
                        rbuf, 1, MPI_DOUBLE, 0, compute_comm); 
            else
                MPI_Gather(&prd.elapsedtime, 1, MPI_DOUBLE, 
                        rbuf, 1, MPI_DOUBLE, 0, compute_comm); 
            if(group_rank==0) {
                int j;
                fprintf(fcompute, "{ \"uptime\":%.2lf ", getuptime() );
                if (verbose) printf("%d/%d ", i, outer-1);

                for(j=0; j<group_size; j++) {
                    if (verbose) printf("dgemm%d %.3lf ", num_io+j, rbuf[j]);
                    fprintf(fcompute, ", \"dgemm%d\":%.3lf ", num_io+j, rbuf[j] );
                }
                fprintf(fcompute, "}\n");
                if (verbose) printf("\n");
                fflush(fcompute);
            }
        } else { // I/O
            MPI_Gather(&prd.elapsedtime, 1, MPI_DOUBLE, 
                    rbuf, 1, MPI_DOUBLE, 0, io_comm); 
            if (group_rank == 0) {
                int j;
                fprintf(fio, "{ \"uptime\":%.2lf ", getuptime() );

                for(j=0; j<group_size; j++) {
                    if (verbose) printf("io%d %.3lf ", j, rbuf[j]);
                    fprintf(fio, ", \"dgemm%d\":%.3lf ", j, rbuf[j] );
                }
                fprintf(fio, "}\n");
                if (verbose) printf("\n");
                fflush(fio);
            }
        }
    }

    fclose(fio);
    fclose(fcompute);
    if (group_rank == 0)
        free(rbuf);

    MPI_Barrier(MPI_COMM_WORLD);
    cleanprd(rank, num_io);


out:
    if (io_gid != MPI_GROUP_EMPTY)
        MPI_Group_free(&io_gid);
    if (compute_gid != MPI_GROUP_EMPTY)
        MPI_Group_free(&compute_gid);
    if (io_comm != MPI_COMM_NULL)
        MPI_Comm_free(&io_comm);
    if (compute_comm != MPI_COMM_NULL)
        MPI_Comm_free(&compute_comm);

    MPI_Finalize();
    return 0;
}


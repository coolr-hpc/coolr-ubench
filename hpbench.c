/*
  benchmark for half-precision data type 

  Kazutomo Yoshii <ky@anl.gov>
 */
#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

#include "rdtsc.h"

#include "raplreader_qh.h"


typedef uint16_t  hp_t;

// assune both s and h have eight elements
void conv_s2h(float *s, hp_t *h)
{
	__m256 m;
	m = _mm256_load_ps( s );
	_mm_store_si128((__m128i*)h, _mm256_cvtps_ph(m, 0 ));
}

void conv_h2s(hp_t *h, float *s)
{
	__m256 m;

	m = _mm256_cvtph_ps(_mm_load_si128((__m128i*)h));

	_mm256_store_ps(s, m);
}

void bench_hp(hp_t *ha, hp_t *hb, unsigned int n)
{
	unsigned int i;
	__m256 m1, m2;

	for (i = 0; i < n; i += 8) {
		m1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)(ha + i)));
		m2 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)(hb + i)));

		m1 = _mm256_add_ps(m1, m2);

		_mm_store_si128((__m128i*)(ha + i), _mm256_cvtps_ph(m1, 0 ));
	}
}

void bench_float(float *sa, float *sb, unsigned int n)
{
	unsigned int i;
	__m256 m1, m2;

	for (i = 0; i < n; i += 8) {
		m1 = _mm256_load_ps(sa + i);
		m2 = _mm256_load_ps(sb + i);

		m1 = _mm256_add_ps(m1, m2);

		_mm256_store_ps(sa + i, m1);
	}

}


void bench_double(double *da, double *db, unsigned int n)
{
	unsigned int i;
	__m256d m1, m2;

	for (i = 0; i < n; i += 4) {
		m1 = _mm256_load_pd(da + i);
		m2 = _mm256_load_pd(db + i);

		m1 = _mm256_add_pd(m1, m2);

		_mm256_store_pd(da + i, m1);
	}
}





int main(int argc, char *argv[])
{
	int n_order = 26; // 64M elements
	double *da, *db;
	float *sa, *sb;
	hp_t  *ha, *hb;
	unsigned int n, i;
	int rc;
	int ntry;
	int size = 1, rank = 0;

#ifdef ENABLE_MPI
	MPI_Init(0,0);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
	if (argc > 1) {
		n_order = atoi(argv[1]);
	}
	n = 1 << n_order;

	ntry = (3ULL*1000*1000*1000/n);

	if (rank == 0) {
#ifdef ENABLE_MPI
		printf("mpi.size=%d\n", size);
#endif
		printf("n=%d (n_order=%d), ntry =%d\n", n, n_order, ntry);
	}

	/* allocate */
	rc = posix_memalign(&da, 32, n*sizeof(double));
	assert(rc == 0);
	rc = posix_memalign(&db, 32, n*sizeof(double));
	assert(rc == 0);
	rc = posix_memalign(&sa, 32, n*sizeof(float));
	assert(rc == 0);
	rc = posix_memalign(&sb, 32, n*sizeof(float));
	assert(rc == 0);
	rc = posix_memalign(&ha, 32, n*sizeof(hp_t));
	assert(rc == 0);
	rc = posix_memalign(&hb, 32, n*sizeof(hp_t));
	assert(rc == 0);
	
	/* init single precision data */
	for (i = 0; i < n; i++ ) {
		sa[i] = (float)(i%512);
		sb[i] = (float)100;
		da[i] = sa[i];
		db[i] = sb[i];
	}
	/* convert to half precision data */
	for (i = 0; i < n; i+=8 ) {
		conv_s2h(sa+i, ha+i);
		conv_s2h(sb+i, hb+i);
	}

	/* */
	{
		int i;
		uint64_t st, et;
		struct raplsample rs1, rs2;
		double j[2], e, dt;
		const double nJ = 1000.0*1000.0*1000.0;

		/* half */
#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		read_rapl(&rs1); st = rdtsc();
		for (i = 0; i < ntry; i++)
			bench_hp(ha,hb,n);
#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		et = rdtsc() - st; read_rapl(&rs2); diff_energy(&rs1, &rs2, j);
		e = j[0] + j[1];
		dt = rs2.t-rs1.t;
		if (rank == 0)
			printf("half  : %f [cycles/op]  e=%lf [nJ/op]  t=%lf [sec]  p=%lf [watt]\n",
				et/((float)n*ntry*size),
				e * nJ / ((float)n*ntry*size), dt, e/dt);

#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		read_rapl(&rs1); st = rdtsc();
		for (i = 0; i < ntry; i++)
			bench_float(sa,sb,n);
#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		et = rdtsc() - st; read_rapl(&rs2); diff_energy(&rs1, &rs2, j);
		e = j[0] + j[1];
		dt = rs2.t-rs1.t;
		if (rank == 0)
			printf("float : %f [cycles/op]  e=%lf [nJ/op]  t=%lf [sec]  p=%lf [watt]\n",
				et/((float)n*ntry*size),
				e * nJ / ((float)n*ntry*size), dt, e/dt);
#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		read_rapl(&rs1); st = rdtsc();
		for (i = 0; i < ntry; i++)
			bench_double(da,db,n);

#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		et = rdtsc() - st; read_rapl(&rs2); diff_energy(&rs1, &rs2, j);
		e = j[0] + j[1];
		dt = rs2.t-rs1.t;
		if (rank == 0)
			printf("double: %f [cycles/op]  e=%lf [nJ/op]  t=%lf [sec]  p=%lf [watt]\n",
				et/((float)n*ntry*size),
				e * nJ / ((float)n*ntry*size), dt, e/dt);

#ifdef ENABLE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}


	/* validation. */
	if (ntry == 1) {
		for (i = 0; i < n; i++ ) {
			float k = (i % 512) + 100;
			if (sa[i] != k) {
				printf("corrupted: sa[%d]=%lf  %lf\n", i, sa[i], k);
				exit(1);
			}
		}
		for (i = 0; i < n; i+=8 ) {
			float tmp[8];
			int j;
			conv_h2s(ha+i, tmp);
			for (j = 0; j < 8; j++) {
				float k = ((i + j) % 512) + 100;
				if (tmp[j] != k) {
					printf("corrupted: ha[%d]=%lf  %lf\n", i+j, tmp[j], (float)k);
					exit(1);
				}
			}
		}
	}

	if (rank == 0)
		printf("done\n");

	return 0;
}


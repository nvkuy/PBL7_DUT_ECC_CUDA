#include <iostream>
#include <tbb/task_group.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <chrono>

#include "kernel.h"

const unsigned MOD = 65537;
const unsigned SPECIAL = MOD - 1;
const unsigned ROOT = 3;
const unsigned ROOT_INV = 21846;
const unsigned MAX_LOG = 16;

const unsigned ALGO_N_2_CUTOFF = 64;

const unsigned LOG_DATA = 10;
const unsigned LOG_SYMBOL = LOG_DATA - 1;
const unsigned LOG_SEG = LOG_SYMBOL - 1;
const unsigned SYMBOL_PER_PACKET = 1 << LOG_SYMBOL;
const unsigned NUM_OF_PACKET = 1 << (MAX_LOG - LOG_SYMBOL);
const unsigned NUM_OF_NEED_PACKET = NUM_OF_PACKET >> 1;
const unsigned SEG_PER_PACKET = 1 << LOG_SEG;
const unsigned SEG_DIFF = 1 << (MAX_LOG - 1);
const unsigned NUM_OF_NEED_SYMBOL = 1 << (MAX_LOG - 1);

const unsigned LEN_ROOT_LAYER_POW = (1 << MAX_LOG) - 1;
const unsigned LEN_ROOT_LAYER_POW_2 = LEN_ROOT_LAYER_POW << 1;
const unsigned LEN_N_POS = ((1 << (MAX_LOG + 1)) - 1);
const unsigned LEN_PACKET_PRODUCT = NUM_OF_PACKET * (SYMBOL_PER_PACKET << 1);
const unsigned LEN_ONE_PACKET_PRODUCT = 1 << (LOG_SYMBOL + 1);

const unsigned LEN_SMALL = NUM_OF_NEED_SYMBOL;
const unsigned LEN_LARGE = LEN_SMALL << 1;

const unsigned ALIGNMENT_SIZE = 32;

unsigned *n_pos;
unsigned *inv;
unsigned *root_layer_pow;
unsigned *packet_product;

inline unsigned fast_mod(unsigned x) {

	// if (x < MOD) return x;
	unsigned res = MOD + (x & 0xFFFFU) - (x >> 16);
	if (res >= MOD) res -= MOD;
	return res;

}

inline unsigned mul_mod(unsigned a, unsigned b)
{
	return (a == SPECIAL && b == SPECIAL) ? 1 : fast_mod(a * b);
}

inline unsigned div_mod(unsigned a, unsigned b, unsigned *d_inv)
{
	return mul_mod(a, d_inv[b]);
}

inline unsigned add_mod(unsigned a, unsigned b)
{
	return fast_mod(a + b);
}

inline unsigned sub_mod(unsigned a, unsigned b)
{
	return fast_mod(a - b + MOD);
}

inline unsigned pow_mod(unsigned a, unsigned b)
{
	unsigned res = 1;
	while (b > 0)
	{
		if (b & 1)
			res = mul_mod(res, a);
		a = mul_mod(a, a);
		b >>= 1;
	}
	return res;
}

void init()
{
	// offline process

	unsigned size_n_pos = LEN_N_POS * ALIGNMENT_SIZE;
	n_pos = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_n_pos);

	for (unsigned i = 1; i <= MAX_LOG; i++)
	{
		unsigned n = 1 << i, st = n - 1;
		for (unsigned j = 0; j < n; j++)
			n_pos[st + j] = j;
	}

	for (unsigned i = 1; i <= MAX_LOG; i++)
	{
		unsigned n = 1 << i, st = n - 1;
		for (unsigned j = 0; j < n; j++)
		{
			unsigned rev_num = 0;
			for (unsigned k = 0; k < i; k++)
			{
				if (j & (1 << k))
					rev_num |= (1 << (i - 1 - k));
			}
			if (j < rev_num)
				std::swap(n_pos[st + j], n_pos[st + rev_num]);
		}
	}

	unsigned size_root = MOD * ALIGNMENT_SIZE;
	unsigned *root_pow = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_root);
	unsigned *root_inv_pow = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_root);
	inv = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_root);

	root_pow[0] = 1, root_inv_pow[0] = 1, inv[0] = 0;
	for (unsigned i = 1; i < MOD; i++)
	{
		root_pow[i] = mul_mod(root_pow[i - 1], ROOT);
		root_inv_pow[i] = mul_mod(root_inv_pow[i - 1], ROOT_INV);
		inv[i] = pow_mod(i, MOD - 2);
	}

	unsigned size_root_layer_pow = LEN_ROOT_LAYER_POW_2 * ALIGNMENT_SIZE;
	root_layer_pow = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_root_layer_pow);

	for (unsigned i = 0; i < 2; i++)
	{
		unsigned st_i = LEN_ROOT_LAYER_POW * i;
		for (unsigned j = 0; j < MAX_LOG; j++)
		{
			unsigned haft_len = 1 << j;
			unsigned st_j = haft_len - 1;
			unsigned ang = 1 << (MAX_LOG - j - 1);
			unsigned wn = i ? root_inv_pow[ang] : root_pow[ang], w = 1;
			for (unsigned k = 0; k < haft_len; k++)
			{
				root_layer_pow[st_i + st_j + k] = w;
				w = mul_mod(w, wn);
			}
		}
	}

	unsigned size_packet_product = LEN_PACKET_PRODUCT * ALIGNMENT_SIZE;
	packet_product = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_packet_product);

	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		for (unsigned j = 0; j < SEG_PER_PACKET; j++)
		{
			unsigned k = (i << LOG_SEG) + j;
			packet_product[st + (j << 1)] = sub_mod(0, root_pow[k]);
			packet_product[st + ((j << 1) | 1)] = 1;
			packet_product[st + ((j + SEG_PER_PACKET) << 1)] = sub_mod(0, root_pow[k + SEG_DIFF]);
			packet_product[st + (((j + SEG_PER_PACKET) << 1) | 1)] = 1;
		}
	}
	unsigned size_tmp = (LEN_ONE_PACKET_PRODUCT << 1) * ALIGNMENT_SIZE;
	unsigned *tmp = (unsigned *)aligned_alloc(ALIGNMENT_SIZE, size_tmp);
	// std::vector<unsigned> tmp(LEN_ONE_PACKET_PRODUCT << 1);
	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		ispc::build_product(packet_product + st, tmp, tmp + LEN_ONE_PACKET_PRODUCT, 1, LOG_SYMBOL + 1, n_pos, root_layer_pow, inv);
	}
	free(root_pow);
	free(root_inv_pow);
	free(tmp);

	std::cout << "Init process completed!" << std::endl;
}

void fin()
{
	// clear cuda memory

	free(n_pos);
	free(inv);
	free(root_layer_pow);
	free(packet_product);
	
}

inline void encode(unsigned *p, unsigned *y) {

	ispc::encode(p, y, n_pos, root_layer_pow, inv);

}

inline void decode(unsigned *x, unsigned *y, unsigned *p) {

	std::vector<unsigned> t1(LEN_LARGE), t2(LEN_LARGE), ax(LEN_LARGE), dax(LEN_SMALL);
	std::vector<unsigned> vdax(LEN_LARGE), n1(LEN_SMALL), n2(LEN_LARGE), n3(LEN_SMALL);
	ispc::decode(x, y, p, 
		t1.data(), t2.data(), ax.data(), dax.data(), vdax.data(), 
		n1.data(), n2.data(), n3.data(),
		packet_product, n_pos, root_layer_pow, inv);

}

// void test_fnt();

// void test_poly_mul();

void test_build_init_product();

void test_encode_decode_performance();

int main() {

    init();

	// test_fnt();

	// test_poly_mul();

	test_build_init_product();

	test_encode_decode_performance();

	fin();

    return 0;
}

// void test_fnt() {

// 	// test correctness of fnt()

// 	unsigned N_test = 32;

// 	for (unsigned tt = 0; tt < N_test; tt++)
// 	{
// 		unsigned log_nc = 15, log_nv = 16, nc = 1 << log_nc, nv = 1 << log_nv;
// 		unsigned size_nc = nc * sizeof(unsigned), size_nv = nv * sizeof(unsigned);
// 		std::vector<unsigned> c1(nc, 0), c2(nv, 0), c3(nv, 0);

// 		for (unsigned i = 0; i < nc; i++)
// 			c1[i] = rand() % (MOD - 1);
// 		shuffle(c1.begin(), c1.end(), std::default_random_engine(time(NULL)));

// 		ispc::fnt(c1.data(), c3.data(), log_nc, log_nv, 0, n_pos, root_layer_pow, inv);
// 		ispc::fnt(c3.data(), c2.data(), log_nv, log_nv, 3, n_pos, root_layer_pow, inv);

// 		for (unsigned i = 0; i < nc; i++)
// 			assert(c1[i] == c2[i]);

// 		// std::cout << "FNT test " << tt << " passed!" << std::endl;
// 	}

// 	std::cout << "FNT test passed!" << std::endl;

// }

// void test_poly_mul() {

// 	// test correctness of poly_mul()

// 	srand(time(NULL));

// 	unsigned N_test = 32;

// 	for (unsigned tt = 0; tt < N_test; tt++)
// 	{

// 		unsigned log_n = 11;
// 		unsigned n = 1 << log_n, size_n = n * sizeof(unsigned);

// 		std::vector<unsigned> a(n), b(n), c1(n << 1, 0), c2(n << 1, 0);

// 		for (unsigned i = 0; i < n; i++)
// 		{
// 			a[i] = rand() % (MOD - 1); // 2 bytes
// 			b[i] = rand() % (MOD - 1); // 2 bytes
// 		}

// 		std::vector<unsigned> t1(n << 1), t2(n << 1);
// 		ispc::poly_mul(a.data(), b.data(), t1.data(), t2.data(), c2.data(), log_n, n_pos, root_layer_pow, inv);

// 		for (unsigned i = 0; i < n; i++)
// 			for (unsigned j = 0; j < n; j++)
// 				c1[i + j] = add_mod(c1[i + j], mul_mod(a[i], b[j]));

// 		for (unsigned i = 0; i < (n << 1); i++)
// 			assert(c1[i] == c2[i]);

// 		// std::cout << "Poly mul test " << tt << " passed!" << std::endl;

// 	}

// 	std::cout << "Poly mul test passed!" << std::endl;

// }

void test_build_init_product() {

	// first 10 element..
	std::vector<unsigned> a1 = {64375, 0, 52012, 0, 2347, 0, 23649, 0, 30899, 0};

	for (unsigned i = 0; i < 10; i++)
		assert(a1[i] == packet_product[i]);

	// first 10 element of next packet..
	std::vector<unsigned> a2 = {64375, 0, 31561, 0, 12153, 0, 31103, 0, 20714, 0};

	for (unsigned i = 0; i < 10; i++)
		assert(a2[i] == packet_product[i + (1 << (LOG_SYMBOL + 1))]);

	std::cout << "Test packet_product passed!" << std::endl;

}

void test_encode_decode_performance()
{

	// test encode(), decode() performance full flow (without prepare memory in device)

	using namespace std;
	srand(time(NULL));

	tbb::task_group tg;

	const unsigned N_test = 128 * 1024 / 64;
	// const unsigned N_test = 1; // use when need profile one..
	const long long symbol_bytes = 2;
	const double size_test_gb = 1.0 * symbol_bytes * NUM_OF_NEED_SYMBOL * N_test / (1024 * 1024 * 1024);

	vector<vector<unsigned>> a(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> b(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL << 1));
	vector<vector<unsigned>> c(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> x(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> y(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	// vector<vector<unsigned>> t1(N_test, vector<unsigned>(LEN_LARGE));
	// vector<vector<unsigned>> t2(N_test, vector<unsigned>(LEN_LARGE));
	// vector<vector<unsigned>> ax(N_test, vector<unsigned>(LEN_LARGE));
	// vector<vector<unsigned>> dax(N_test, vector<unsigned>(LEN_SMALL));
	// vector<vector<unsigned>> vdax(N_test, vector<unsigned>(LEN_LARGE));
	// vector<vector<unsigned>> n1(N_test, vector<unsigned>(LEN_SMALL));
	// vector<vector<unsigned>> n2(N_test, vector<unsigned>(LEN_LARGE));
	// vector<vector<unsigned>> n3(N_test, vector<unsigned>(LEN_SMALL));

	for (unsigned tt = 0; tt < N_test; tt++)
		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			a[tt][i] = rand() % (MOD - 1); // 2 bytes


	cout << "Encode performance test start" << endl;

	auto start1 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++)
	{
		// ispc::encode(a[tt].data(), b[tt].data(), n_pos, root_layer_pow, inv);
		tg.run([tt, &a, &b]{encode(a[tt].data(), b[tt].data());});
	}
	
	tg.wait();

	auto stop1 = chrono::high_resolution_clock::now();
	auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1).count();

	cout << "Encode " << N_test << " 64kb chunks in " << duration1 << "ms" << endl;
	cout << "Encode " << (1.0 * size_test_gb) / (1.0 * duration1 / 1000.0) << " GB/s" << endl;

	for (unsigned tt = 0; tt < N_test; tt++)
	{
		for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
		{
			unsigned stx = i * SYMBOL_PER_PACKET;
			for (unsigned j = 0; j < SEG_PER_PACKET; j++)
			{
				x[tt][stx + j] = stx + j;
				x[tt][stx + j + SEG_PER_PACKET] = stx + j + SEG_DIFF;
				y[tt][stx + j] = b[tt][stx + j];
				y[tt][stx + j + SEG_PER_PACKET] = b[tt][stx + j + SEG_DIFF];
			}
		}
	}

	cout << "Decode performance test start" << endl;

	auto start2 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++)
	{
		// vector<unsigned> t1(LEN_LARGE), t2(LEN_LARGE), ax(LEN_LARGE), dax(LEN_SMALL);
		// vector<unsigned> vdax(LEN_LARGE), n1(LEN_SMALL), n2(LEN_LARGE), n3(LEN_SMALL);

		// ispc::decode(x[tt].data(), y[tt].data(), c[tt].data(), 
		// t1[tt].data(), t2[tt].data(), ax[tt].data(), dax[tt].data(), vdax[tt].data(), 
		// n1[tt].data(), n2[tt].data(), n3[tt].data(),
		// packet_product, n_pos, root_layer_pow, inv);

		// ispc::decode(x[tt].data(), y[tt].data(), c[tt].data(), 
		// t1.data(), t2.data(), ax.data(), dax.data(), vdax.data(), 
		// n1.data(), n2.data(), n3.data(),
		// packet_product, n_pos, root_layer_pow, inv);

		tg.run([tt, &x, &y, &c] {decode(x[tt].data(), y[tt].data(), c[tt].data());});
	}
	
	tg.wait();

	auto stop2 = chrono::high_resolution_clock::now();
	auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();

	cout << "Decode " << N_test << " 64kb chunks in " << duration2 << "ms" << endl;
	cout << "Decode " << (1.0 * size_test_gb) / (1.0 * duration2 / 1000.0) << " GB/s" << endl;

	for (unsigned tt = 0; tt < N_test; tt++)
	{
		for (unsigned i = 0; i < c[tt].size(); i++)
			assert(a[tt][i] == c[tt][i]);
	}

}
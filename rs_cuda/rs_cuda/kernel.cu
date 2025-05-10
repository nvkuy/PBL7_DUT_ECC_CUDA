#ifdef __CUDACC__
#define CUDA_KERNEL(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define CUDA_SYNCTHREADS() __syncthreads()
#else
#define CUDA_KERNEL(grid, block, sh_mem, stream)
#define CUDA_SYNCTHREADS()
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <math.h>


#include <iostream>
#include <time.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <chrono>

const unsigned MOD = 65537;
const unsigned SPECIAL = MOD - 1;
const unsigned ROOT = 3;
const unsigned ROOT_INV = 21846;
const unsigned MAX_LOG = 16;

const unsigned WARP_SIZE = 32;
const unsigned ALGO_N_2_CUTOFF = WARP_SIZE << 1;

const unsigned ALGO_N_1_KERNEL_CUTOFF = 512;

const unsigned ALGO_N_1_LOW_WPT = 512; // poly_mul_i
const unsigned ALGO_N_1_HIGH_WPT = 128; // fnt
const unsigned ALGO_N_1_EXTR_WPT = 1; // build_product

const unsigned ALGO_N_1_NTH_SMALL = WARP_SIZE;
const unsigned ALGO_N_1_NTH_MED = WARP_SIZE << 1;
const unsigned ALGO_N_1_NTH_LARGE = WARP_SIZE << 2;

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

const unsigned SIZE_SMALL = NUM_OF_NEED_SYMBOL * sizeof(unsigned);
const unsigned SIZE_LARGE = SIZE_SMALL << 1;
const unsigned SIZE_ONE_PACKET_PRODUCT = LEN_ONE_PACKET_PRODUCT * sizeof(unsigned);

unsigned* d_N_pos;
unsigned* d_root_pow;
unsigned* d_root_inv_pow;
unsigned* d_inv;
unsigned* d_root_layer_pow;
unsigned* d_packet_product;

__host__ __device__ __forceinline__ inline void build_launch_param(unsigned n, unsigned& wpt, unsigned& n_th) {
	
	if (n < WARP_SIZE)
		n_th = n;
	else {
		n_th = WARP_SIZE;
		unsigned t_nth = n / wpt;
		if (t_nth >= ALGO_N_1_NTH_SMALL)
			n_th = ALGO_N_1_NTH_SMALL;
		if (t_nth >= ALGO_N_1_NTH_MED)
			n_th = ALGO_N_1_NTH_MED;
		if (t_nth >= ALGO_N_1_NTH_LARGE)
			n_th = ALGO_N_1_NTH_LARGE;
	}
	wpt = n / n_th;
	//assert(n == wpt * n_th);
	//assert(n_th < 1024);
}

__host__ __device__ __forceinline__ inline unsigned mul_mod(unsigned a, unsigned b)
{
	// TODO: use 32-bit simd
	if (a == SPECIAL && b == SPECIAL)
		return 1; // overflow
	return (a * b) % MOD;
}

__device__ __forceinline__ inline unsigned div_mod(unsigned a, unsigned b,
	unsigned* d_inv)
{
	return mul_mod(a, d_inv[b]);
}

__host__ __device__ __forceinline__ inline unsigned add_mod(unsigned a, unsigned b)
{
	return (a + b) % MOD;
}

__host__ __device__ __forceinline__ inline unsigned sub_mod(unsigned a, unsigned b)
{
	return (a - b + MOD) % MOD;
}

__host__ __device__ __forceinline__ inline unsigned pow_mod(unsigned a, unsigned b)
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

__global__ void g_fnt(unsigned* a, unsigned* b, unsigned log_na, unsigned log_nb, unsigned opt,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, unsigned wpt)
{

	/*
	opt 2 bit: x1 x2
	 - x1: w_n or 1/w_n
	 - x2: need result * 1/n
	*/

	// size_b >= size_a;
	// need memset *b before use unless size_a == size_b

	// have size_b/2 tasks
	// only use when have atleast 2^10 task

	unsigned id = threadIdx.x;
	unsigned na = 1 << log_na, nb = 1 << log_nb, wp = (opt & 2) >> 1, st = nb - 1;
	unsigned os1 = nb >> 1, os2 = LEN_ROOT_LAYER_POW * wp;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		if (k < na) b[d_N_pos[st + k]] = a[k];
		if (log_na == log_nb) b[d_N_pos[st + k + os1]] = a[k + os1];
	}
	
	CUDA_SYNCTHREADS();

	for (unsigned i = 0; i < log_nb; i++) {

		unsigned haft_len = 1 << i;
		for (unsigned j = 0; j < wpt; j++) {
			unsigned k = id * wpt + j;
			unsigned bl_st = ((k >> i) << (i + 1)), th_id = (k & (haft_len - 1));
			unsigned pos = bl_st + th_id;
			unsigned u = b[pos];
			unsigned v = mul_mod(b[pos + haft_len], d_root_layer_pow[os2 + haft_len - 1 + th_id]);
			b[pos] = add_mod(u, v);
			b[pos + haft_len] = sub_mod(u, v);
		}

		CUDA_SYNCTHREADS();

	}

	if (opt & 1) {
		for (unsigned j = 0; j < wpt; j++) {
			unsigned k = id * wpt + j;
			b[k] = div_mod(b[k], nb, d_inv);
			b[k + os1] = div_mod(b[k + os1], nb, d_inv);
		}

	}


}

__device__ void d_fnt(unsigned* a, unsigned* b, unsigned log_na, unsigned log_nb, unsigned opt,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	// sequence version of g_fnt(), use when nb/2 < ALGO_N_1_KERNEL_CUTOFF

	unsigned na = 1 << log_na, nb = 1 << log_nb, wp = (opt & 2) >> 1, st = nb - 1;
	unsigned os = LEN_ROOT_LAYER_POW * wp;

	for (unsigned i = 0; i < na; i++)
		b[d_N_pos[st + i]] = a[i];

	for (unsigned i = 0; i < log_nb; i++) {
		unsigned haft_len = 1 << i, len = haft_len << 1;
		unsigned wlen = d_root_layer_pow[os + haft_len];
		for (unsigned j = 0; j < nb; j += len) {
			unsigned w = 1;
			for (unsigned k = 0; k < haft_len; k++) {
				unsigned u = b[j + k], v = mul_mod(b[j + k + haft_len], w);
				b[j + k] = add_mod(u, v);
				b[j + k + haft_len] = sub_mod(u, v);
				w = mul_mod(w, wlen);
			}
		}
	}

	if (opt & 1) {
		for (unsigned i = 0; i < nb; i++)
			b[i] = div_mod(b[i], nb, d_inv);
	}

}

__device__ __forceinline__ void d_vector_mul_i(unsigned* a, unsigned* b, unsigned* c, unsigned n) {

	for (unsigned i = 0; i < n; i++)
		c[i] = mul_mod(a[i], b[i]);

}

__global__ void g_vector_mul_i(unsigned* a, unsigned* b, unsigned* c, unsigned wpt)
{

	unsigned id = threadIdx.x;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		c[k] = mul_mod(a[k], b[k]);
	}

}

__device__ __forceinline__ void d_poly_mul(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	// 2 ^ log_n == size_a && size_a == size_b
	// *c == *a && *a + na == *b (allow)

	unsigned na = 1 << log_n, nc = na << 1, size_nc = nc * sizeof(unsigned);

	if (na <= ALGO_N_2_CUTOFF)
	{
		for (unsigned i = 0; i < na; i++)
		{
			t1[i] = a[i];
			t2[i] = b[i];
		}
		memset(c, 0, size_nc);
		for (unsigned i = 0; i < na; i++)
			for (unsigned j = 0; j < na; j++)
				c[i + j] = add_mod(c[i + j], mul_mod(t1[i], t2[j]));
	}
	else
	{
		memset(t1, 0, size_nc);
		memset(t2, 0, size_nc);

		if (na <= ALGO_N_1_KERNEL_CUTOFF) {

			d_fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv);
			d_fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv);

			d_vector_mul_i(t1, t2, t1, nc);

			d_fnt(t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv);

		}
		else {

			
			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

			unsigned n_th1, wpt1 = ALGO_N_1_HIGH_WPT;
			build_launch_param(na, wpt1, n_th1);
			g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, wpt1);
			g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, wpt1);

			unsigned n_th2, wpt2 = ALGO_N_1_LOW_WPT;
			build_launch_param(nc, wpt2, n_th2);
			g_vector_mul_i CUDA_KERNEL(1, n_th2, NULL, stream) (t1, t2, t1, wpt2);

			g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, wpt1);

			cudaStreamDestroy(stream);

		}

	}
}

inline void h_poly_mul(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, cudaStream_t stream) {

	// only use with large poly

	unsigned na = 1 << log_n, nc = na << 1, size_nc = nc * sizeof(unsigned);
	cudaMemsetAsync(t1, 0, size_nc, stream);
	cudaMemsetAsync(t2, 0, size_nc, stream);

	unsigned n_th1, wpt1 = ALGO_N_1_HIGH_WPT;
	build_launch_param(na, wpt1, n_th1);
	g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, wpt1);
	g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, wpt1);

	unsigned n_th2, wpt2 = ALGO_N_1_LOW_WPT;
	build_launch_param(nc, wpt2, n_th2);
	g_vector_mul_i CUDA_KERNEL(1, n_th2, NULL, stream) (t1, t2, t1, wpt2);

	g_fnt CUDA_KERNEL(1, n_th1, NULL, stream) (t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, wpt1);

}

__global__ void poly_deriv(unsigned* p1, unsigned* p2, unsigned wpt)
{

	unsigned id = threadIdx.x;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		p2[k] = mul_mod(p1[k + 1], k + 1);
	}
}

__global__ void build_product_i(unsigned* p, unsigned* t1, unsigned* t2, unsigned i,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, unsigned wpt)
{

	unsigned id = threadIdx.x;
	unsigned len = 1 << i;

	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j, st = k << (i + 1);
		d_poly_mul(p + st, p + st + len, t1 + st, t2 + st, p + st, i, d_N_pos, d_root_layer_pow, d_inv);
	}

}

void build_product(unsigned* p, unsigned* t1, unsigned* t2, unsigned log_n1, unsigned log_n2, cudaStream_t stream)
{

	// p, t1, t2 in device

	unsigned n = 1 << log_n2;
	for (unsigned i = log_n1; i < log_n2; i++)
	{
		unsigned m = n >> (i + 1);
		unsigned n_th, wpt = ALGO_N_1_EXTR_WPT;
		build_launch_param(m, wpt, n_th);
		build_product_i CUDA_KERNEL(1, n_th, NULL, stream)(p, t1, t2, i, d_N_pos, d_root_layer_pow, d_inv, wpt);
	}
}

inline void build_ax(unsigned* x, unsigned* p, unsigned* t1, unsigned* t2, cudaStream_t stream)
{

	// p, t1, t2 in device
	// x in host

	for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		unsigned st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 2;
		cudaMemcpyAsync(p + st_p1, d_packet_product + st_p2, SIZE_ONE_PACKET_PRODUCT, cudaMemcpyDeviceToDevice, stream);
	}
	build_product(p, t1, t2, LOG_SYMBOL + 1, MAX_LOG, stream);
}

__global__ void build_n1(unsigned* n1, unsigned* vdax, unsigned* x, unsigned* y,
	unsigned* d_inv, unsigned wpt)
{

	unsigned id = threadIdx.x;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		n1[k] = div_mod(y[k], vdax[x[k]], d_inv);
	}

}


__global__ void build_n2(unsigned* n2, unsigned* n1, unsigned* x, unsigned wpt) {

	// need to memset n2 first
	// have NUM_OF_NEED_SYMBOL tasks

	unsigned id = threadIdx.x;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		n2[x[k]] = n1[k];
	}

}

__global__ void build_n3(unsigned* n3, unsigned* p_n3, unsigned wpt) {

	// have NUM_OF_NEED_SYMBOL tasks

	unsigned id = threadIdx.x;
	for (unsigned j = 0; j < wpt; j++) {
		unsigned k = id * wpt + j;
		n3[k] = sub_mod(0, p_n3[k + 1]);
	}

}

inline void build_px(unsigned* p, unsigned* ax, unsigned* n3, unsigned* t1, unsigned* t2,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, cudaStream_t stream) {

	h_poly_mul(ax, n3, t1, t2, p, MAX_LOG - 1, d_N_pos, d_root_layer_pow, d_inv, stream);

}

void encode(unsigned* p, unsigned* y)
{

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	unsigned* d_p;
	unsigned* d_y;
	cudaMallocAsync(&d_p, SIZE_SMALL, stream);
	cudaMallocAsync(&d_y, SIZE_LARGE, stream);
	cudaMemcpyAsync(d_p, p, SIZE_SMALL, cudaMemcpyHostToDevice, stream);
	cudaMemsetAsync(d_y, 0, SIZE_LARGE, stream);

	unsigned n_th, wpt = ALGO_N_1_HIGH_WPT;
	build_launch_param(1 << (MAX_LOG - 1), wpt, n_th);
	g_fnt CUDA_KERNEL(1, n_th, NULL, stream) (d_p, d_y, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, wpt);

	cudaMemcpyAsync(y, d_y, SIZE_LARGE, cudaMemcpyDeviceToHost, stream);

	cudaFreeAsync(d_p, stream);
	cudaFreeAsync(d_y, stream);

	cudaStreamDestroy(stream);

}

void decode(unsigned* x, unsigned* y, unsigned* p)
{

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	unsigned n_th_hi, wpt_hi = ALGO_N_1_HIGH_WPT;
	build_launch_param(NUM_OF_NEED_SYMBOL, wpt_hi, n_th_hi);
	unsigned n_th_lo, wpt_lo = ALGO_N_1_LOW_WPT;
	build_launch_param(NUM_OF_NEED_SYMBOL, wpt_lo, n_th_lo);

	unsigned* t1;
	unsigned* t2;
	cudaMallocAsync(&t1, SIZE_LARGE, stream);
	cudaMallocAsync(&t2, SIZE_LARGE, stream);

	unsigned* ax;
	cudaMallocAsync(&ax, SIZE_LARGE, stream);
	build_ax(x, ax, t1, t2, stream);

	unsigned* dax;
	cudaMallocAsync(&dax, SIZE_SMALL, stream);
	poly_deriv CUDA_KERNEL(1, n_th_lo, NULL, stream)(ax, dax, wpt_lo);

	unsigned* vdax;
	cudaMallocAsync(&vdax, SIZE_LARGE, stream);
	cudaMemsetAsync(vdax, 0, SIZE_LARGE, stream);
	g_fnt CUDA_KERNEL(1, n_th_hi, NULL, stream) (dax, vdax, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, wpt_hi);

	unsigned* n1;
	cudaMallocAsync(&n1, SIZE_SMALL, stream);
	cudaMemcpyAsync(t1, x, SIZE_SMALL, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(t2, y, SIZE_SMALL, cudaMemcpyHostToDevice, stream);
	build_n1 CUDA_KERNEL(1, n_th_lo, NULL, stream)(n1, vdax, t1, t2, d_inv, wpt_lo);

	unsigned* n2;
	unsigned* n3;
	cudaMallocAsync(&n2, SIZE_LARGE, stream);
	cudaMallocAsync(&n3, SIZE_SMALL, stream);

	cudaMemsetAsync(n2, 0, SIZE_LARGE, stream);
	build_n2 CUDA_KERNEL(1, n_th_lo, NULL, stream)(n2, n1, t1, wpt_lo);

	g_fnt CUDA_KERNEL(1, n_th_hi, NULL, stream) (n2, t2, MAX_LOG, MAX_LOG, 2, d_N_pos, d_root_layer_pow, d_inv, wpt_hi);
	build_n3 CUDA_KERNEL(1, n_th_lo, NULL, stream)(n3, t2, wpt_lo);

	build_px(n2, ax, n3, t1, t2, d_N_pos, d_root_layer_pow, d_inv, stream);
	cudaMemcpyAsync(p, n2, SIZE_SMALL, cudaMemcpyDeviceToHost, stream);

	cudaFreeAsync(t1, stream);
	cudaFreeAsync(t2, stream);
	cudaFreeAsync(ax, stream);
	cudaFreeAsync(dax, stream);
	cudaFreeAsync(vdax, stream);
	cudaFreeAsync(n1, stream);
	cudaFreeAsync(n2, stream);
	cudaFreeAsync(n3, stream);

	cudaStreamDestroy(stream);

}

void init()
{
	// offline process
	// TODO: change runtime limit later..

	cudaDeviceSynchronize();
	cudaDeviceReset();
	cudaDeviceSynchronize();

	unsigned size_N_pos = LEN_N_POS * sizeof(unsigned);
	unsigned* N_pos = (unsigned*)malloc(size_N_pos);
	cudaMalloc(&d_N_pos, size_N_pos);

	for (unsigned i = 1; i <= MAX_LOG; i++)
	{
		unsigned n = 1 << i, st = n - 1;
		for (unsigned j = 0; j < n; j++)
			N_pos[st + j] = j;
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
				std::swap(N_pos[st + j], N_pos[st + rev_num]);
		}
	}

	cudaMemcpy(d_N_pos, N_pos, size_N_pos, cudaMemcpyHostToDevice);
	free(N_pos);

	unsigned size_root = MOD * sizeof(unsigned);
	unsigned* root_pow = (unsigned*)malloc(size_root);
	unsigned* root_inv_pow = (unsigned*)malloc(size_root);
	unsigned* inv = (unsigned*)malloc(size_root);
	cudaMalloc(&d_root_pow, size_root);
	cudaMalloc(&d_root_inv_pow, size_root);
	cudaMalloc(&d_inv, size_root);

	root_pow[0] = 1, root_inv_pow[0] = 1, inv[0] = 0;
	for (unsigned i = 1; i < MOD; i++)
	{
		root_pow[i] = mul_mod(root_pow[i - 1], ROOT);
		root_inv_pow[i] = mul_mod(root_inv_pow[i - 1], ROOT_INV);
		inv[i] = pow_mod(i, MOD - 2);
	}

	cudaMemcpy(d_root_pow, root_pow, size_root, cudaMemcpyHostToDevice);
	cudaMemcpy(d_root_inv_pow, root_inv_pow, size_root, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inv, inv, size_root, cudaMemcpyHostToDevice);

	unsigned size_root_layer_pow = LEN_ROOT_LAYER_POW_2 * sizeof(unsigned);
	unsigned* root_layer_pow = (unsigned*)malloc(size_root_layer_pow);
	cudaMalloc(&d_root_layer_pow, size_root_layer_pow);

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

	cudaMemcpy(d_root_layer_pow, root_layer_pow, size_root_layer_pow, cudaMemcpyHostToDevice);
	free(root_layer_pow);

	unsigned size_packet_product = LEN_PACKET_PRODUCT * sizeof(unsigned);
	unsigned* packet_product = (unsigned*)malloc(size_packet_product);
	cudaMalloc(&d_packet_product, size_packet_product);

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
	cudaMemcpy(d_packet_product, packet_product, size_packet_product, cudaMemcpyHostToDevice);
	free(packet_product);
	unsigned* tmp;
	cudaMalloc(&tmp, (LEN_ONE_PACKET_PRODUCT << 1) * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		build_product(d_packet_product + st, tmp, tmp + LEN_ONE_PACKET_PRODUCT, 1, LOG_SYMBOL + 1, NULL);
	}
	cudaFree(tmp);
	free(inv);
	free(root_pow);
	free(root_inv_pow);

	cudaDeviceSynchronize();
	std::cout << "Init process completed!" << std::endl;

}

void fin()
{
	// clear cuda memory

	cudaDeviceSynchronize();
	cudaDeviceReset();
	cudaDeviceSynchronize();

}

void test_fnt();

void test_poly_mul();

void test_build_init_product();

void test_encode_decode();

void test_fnt_performance();

void test_encode_decode_performance();

int main()
{

	init();

	//test_fnt();

	//test_poly_mul();

	//test_build_init_product();

	//test_encode_decode();

	test_fnt_performance();

	test_encode_decode_performance();

	fin();

	return 0;
}

void test_fnt() {

	// test correctness of fnt()

	unsigned N_test = 32;

	for (unsigned tt = 0; tt < N_test; tt++) {
		unsigned log_nc = 15, log_nv = 16, nc = 1 << log_nc, nv = 1 << log_nv;
		unsigned size_nc = nc * sizeof(unsigned), size_nv = nv * sizeof(unsigned);
		std::vector<unsigned> c1(nc), c2(nc);
		unsigned* d_c1, * d_c2, * d_v;
		cudaMalloc(&d_c1, size_nc);
		cudaMemset(d_c1, 0, size_nc);
		cudaMalloc(&d_c2, size_nv);
		cudaMemset(d_c2, 0, size_nv);
		cudaMalloc(&d_v, size_nv);
		cudaMemset(d_v, 0, size_nv);

		for (unsigned i = 0; i < nc; i++)
			c1[i] = i;
		shuffle(c1.begin(), c1.end(), std::default_random_engine(time(NULL)));
		cudaMemcpy(d_c1, c1.data(), size_nc, cudaMemcpyHostToDevice);

		unsigned n_th, wpt = ALGO_N_1_HIGH_WPT;
		build_launch_param(nv >> 1, wpt, n_th);
		g_fnt CUDA_KERNEL(1, n_th, NULL, NULL) (d_c1, d_v, log_nc, log_nv, 0, d_N_pos, d_root_layer_pow, d_inv, wpt);
		g_fnt CUDA_KERNEL(1, n_th, NULL, NULL) (d_v, d_c2, log_nv, log_nv, 3, d_N_pos, d_root_layer_pow, d_inv, wpt);

		cudaMemcpy(c2.data(), d_c2, size_nc, cudaMemcpyDeviceToHost);
		for (unsigned i = 0; i < nc; i++)
			assert(c1[i] == c2[i]);

		cudaFree(d_c1);
		cudaFree(d_c2);
		cudaFree(d_v);

		std::cout << "FNT test " << tt << " passed!" << std::endl;

	}

}

void test_build_init_product() {

	// first 10 element..
	std::vector<unsigned> a1 = { 64375, 0, 52012, 0, 2347, 0, 23649, 0, 30899, 0 }, b1(10);
	cudaMemcpy(b1.data(), d_packet_product, 10 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < 10; i++)
		assert(a1[i] == b1[i]);

	// first 10 element of next packet..
	std::vector<unsigned> a2 = { 64375, 0, 31561, 0, 12153, 0, 31103, 0, 20714, 0 }, b2(10);
	cudaMemcpy(b2.data(), d_packet_product + (1 << (LOG_SYMBOL + 1)), 10 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < 10; i++)
		assert(a2[i] == b2[i]);

	std::cout << "Test packet_product passed!" << std::endl;

}

//__global__ void poly_mul_wrapper_for_test(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
//	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv) {
//
//	// launch with 1 thread only
//	d_poly_mul(a, b, t1, t2, c, log_n, d_N_pos, d_root_layer_pow, d_inv);
//
//}

void test_poly_mul() {

	// test correctness of poly_mul()

	srand(time(NULL));

	unsigned N_test = 32;

	for (unsigned tt = 0; tt < N_test; tt++) {

		unsigned log_n = 11;
		unsigned n = 1 << log_n, size_n = n * sizeof(unsigned);

		std::vector<unsigned> a(n), b(n), c1(n << 1, 0), c2(n << 1, 0);

		for (unsigned i = 0; i < n; i++) {
			a[i] = rand() % (MOD - 1); // 2 bytes
			b[i] = rand() % (MOD - 1); // 2 bytes
		}

		unsigned* t1, * t2, * d_c;
		cudaMalloc(&t1, size_n << 1);
		cudaMalloc(&t2, size_n << 1);
		cudaMalloc(&d_c, size_n << 1);
		cudaMemcpy(d_c, a.data(), size_n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c + n, b.data(), size_n, cudaMemcpyHostToDevice);
		h_poly_mul(d_c, d_c + n, t1, t2, d_c, log_n, d_N_pos, d_root_layer_pow, d_inv, NULL);
		//poly_mul_wrapper_for_test CUDA_KERNEL(1, 1, NULL, NULL) (d_c, d_c + n, t1, t2, d_c, log_n, d_N_pos, d_root_layer_pow, d_inv);

		for (unsigned i = 0; i < n; i++)
			for (unsigned j = 0; j < n; j++)
				c1[i + j] = add_mod(c1[i + j], mul_mod(a[i], b[j]));

		cudaMemcpy(c2.data(), d_c, size_n << 1, cudaMemcpyDeviceToHost);

		for (unsigned i = 0; i < (n << 1); i++)
			assert(c1[i] == c2[i]);

		std::cout << "Poly mul test " << tt << " passed!" << std::endl;

		cudaFree(t1);
		cudaFree(t2);
		cudaFree(d_c);

	}

}

void test_encode_decode() {

	// test correctness of encode() and decode()

	srand(time(NULL));

	unsigned N_test = 32;

	for (unsigned tt = 0; tt < N_test; tt++) {
		std::vector<unsigned> a(NUM_OF_NEED_SYMBOL), b(NUM_OF_NEED_SYMBOL << 1), c(NUM_OF_NEED_SYMBOL);

		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			a[i] = rand() % (MOD - 1); // 2 bytes

		encode(a.data(), b.data());

		std::vector<unsigned> x(NUM_OF_NEED_SYMBOL), y(NUM_OF_NEED_SYMBOL);

		for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++) {
			unsigned stx = i * SYMBOL_PER_PACKET;
			for (unsigned j = 0; j < SEG_PER_PACKET; j++) {
				x[stx + j] = stx + j;
				x[stx + j + SEG_PER_PACKET] = stx + j + SEG_DIFF;
				y[stx + j] = b[stx + j];
				y[stx + j + SEG_PER_PACKET] = b[stx + j + SEG_DIFF];
			}
		}

		decode(x.data(), y.data(), c.data());
		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			assert(a[i] == c[i]);
		std::cout << "Encode decode test " << tt << " passed!" << std::endl;
	}

}

void test_fnt_performance() {

	using namespace std;

	const unsigned N_test = 512 * 1024 / 64;
	unsigned log_n = 16, n = 1 << log_n;
	unsigned size_n = n * sizeof(unsigned);
	vector<vector<unsigned>> a(N_test, vector<unsigned>(n));
	cudaStream_t stream[N_test];
	vector<unsigned*> d_a(N_test), d_b(N_test);

	for (unsigned tt = 0; tt < N_test; tt++) {
		for (unsigned i = 0; i < n; i++)
			a[tt][i] = i;
		cudaStreamCreate(&stream[tt]);
		cudaMallocAsync(&d_a[tt], size_n, stream[tt]);
		cudaMallocAsync(&d_b[tt], size_n, stream[tt]);
		cudaMemcpyAsync(d_a[tt], a[tt].data(), size_n, cudaMemcpyHostToDevice, stream[tt]);
	}

	cudaDeviceSynchronize();
	auto start = chrono::high_resolution_clock::now();

	unsigned n_th, wpt = ALGO_N_1_HIGH_WPT;
	build_launch_param(n >> 1, wpt, n_th);

	for (unsigned tt = 0; tt < N_test; tt++) {
		g_fnt CUDA_KERNEL(1, n_th, NULL, stream[tt]) (d_a[tt], d_b[tt], log_n, log_n, 0, d_N_pos, d_root_layer_pow, d_inv, wpt);
	}

	cudaDeviceSynchronize();
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();

	cout << "FNT " << N_test << " in " << duration << "ms" << endl;

	for (unsigned tt = 0; tt < N_test; tt++) {
		cudaFreeAsync(d_a[tt], stream[tt]);
		cudaFreeAsync(d_b[tt], stream[tt]);
		cudaStreamDestroy(stream[tt]);
	}

}

void test_encode_decode_performance() {

	using namespace std;
	srand(time(NULL));

	unsigned N_test = 10 * 1024 / 64;
	vector<vector<unsigned>> a(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> b(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL << 1));
	vector<vector<unsigned>> c(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));

	for (unsigned tt = 0; tt < N_test; tt++)
		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			a[tt][i] = rand() % (MOD - 1); // 2 bytes
	
	cudaDeviceSynchronize();
	auto start1 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++)
		encode(a[tt].data(), b[tt].data());

	cudaDeviceSynchronize();
	auto stop1 = chrono::high_resolution_clock::now();
	auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1).count();

	cout << "Encode " << N_test << " chunks in " << duration1 << "ms" << endl;

	vector<vector<unsigned>> x(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> y(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));

	for (unsigned tt = 0; tt < N_test; tt++) {
		for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++) {
			unsigned stx = i * SYMBOL_PER_PACKET;
			for (unsigned j = 0; j < SEG_PER_PACKET; j++) {
				x[tt][stx + j] = stx + j;
				x[tt][stx + j + SEG_PER_PACKET] = stx + j + SEG_DIFF;
				y[tt][stx + j] = b[tt][stx + j];
				y[tt][stx + j + SEG_PER_PACKET] = b[tt][stx + j + SEG_DIFF];
			}
		}
	}

	cudaDeviceSynchronize();
	auto start2 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++)
		decode(x[tt].data(), y[tt].data(), c[tt].data());

	cudaDeviceSynchronize();
	auto stop2 = chrono::high_resolution_clock::now();
	auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();

	cout << "Decode " << N_test << " chunks in " << duration2 << "ms" << endl;

}
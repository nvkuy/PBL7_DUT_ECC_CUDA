#ifdef __CUDACC__
#define CUDA_KERNEL(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define CUDA_KERNEL(grid, block, sh_mem, stream)
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>


#include <iostream>
#include <time.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>

const unsigned MOD = 65537;
const unsigned SPECIAL = MOD - 1;
const unsigned ROOT = 3;
const unsigned ROOT_INV = 21846;
const unsigned MAX_LOG = 16;

const unsigned WARP_SIZE = 32;
const unsigned ALGO_N_1_CUTOFF = WARP_SIZE * WARP_SIZE;
const unsigned ALGO_N_2_CUTOFF = WARP_SIZE << 1;

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

unsigned *d_N_pos;
unsigned *d_root_pow;
unsigned *d_root_inv_pow;
unsigned *d_inv;
unsigned *d_root_layer_pow;
unsigned *d_packet_product;

__host__ __device__ __forceinline__ inline void build_launch_param(unsigned n, unsigned &n_th, unsigned &n_bl) {
	n_th = n > 256 ? 256 : n;
	n_bl = n / n_th;
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

__global__ void pre_fnt(unsigned *a, unsigned *b, unsigned st, unsigned* d_N_pos)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	b[d_N_pos[st + id]] = a[id];
	
}

__global__ void end_fnt(unsigned *a, unsigned *b, unsigned n, unsigned *d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	b[id << 1] = div_mod(b[id << 1], n, d_inv);
	b[(id << 1) | 1] = div_mod(b[(id << 1) | 1], n, d_inv);
}

__global__ void fnt_i(unsigned *a, unsigned *b, unsigned i, bool inv,
	unsigned *d_root_layer_pow)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned haft_len = 1 << i;
	unsigned bl_st = ((id >> i) << (i + 1)), th_id = (id & (haft_len - 1));
	unsigned pos = bl_st + th_id;
	unsigned u = b[pos];
	unsigned v = mul_mod(b[pos + haft_len], d_root_layer_pow[(LEN_ROOT_LAYER_POW * inv) + haft_len - 1 + th_id]);
	b[pos] = add_mod(u, v);
	b[pos + haft_len] = sub_mod(u, v);
}

__host__ __forceinline__ __device__ inline void fnt(unsigned *a, unsigned *b, unsigned log_na, unsigned log_nb, unsigned opt,
	unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv)
{

	/*
	opt 2 bit: x1 x2
	 - x1: w_n or 1/w_n
	 - x2: need result * 1/n
	*/

	// size_b >= size_a;
	// may need memset *b before use

	unsigned na = 1 << log_na, nb = 1 << log_nb, wp = (opt & 2) >> 1;
	unsigned n_th, n_bl;

	build_launch_param(na, n_th, n_bl);
	pre_fnt CUDA_KERNEL(n_bl, n_th, NULL, NULL)(a, b, nb - 1, d_N_pos);

	build_launch_param(nb >> 1, n_th, n_bl);
	for (unsigned i = 0; i < log_nb; i++)
		fnt_i CUDA_KERNEL(n_bl, n_th, NULL, NULL)(a, b, i, wp, d_root_layer_pow);

	if (opt & 1)
		end_fnt CUDA_KERNEL(n_bl, n_th, NULL, NULL)(a, b, nb, d_inv);
}

__global__ void vector_add(unsigned *a, unsigned *b, unsigned *c)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	c[id] = add_mod(a[id], b[id]);

}

__global__ void vector_mul_i(unsigned *a, unsigned *b, unsigned *c)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	c[id] = mul_mod(a[id], b[id]);

}

__forceinline__ __device__ void poly_add(unsigned *a, unsigned *b, unsigned *c, unsigned n)
{
	if (n <= ALGO_N_1_CUTOFF)
	{
		for (unsigned i = 0; i < n; i++)
			c[i] = add_mod(a[i], b[i]);
	}
	else
	{
		unsigned n_th, n_bl;
		build_launch_param(n, n_th, n_bl);
		vector_add CUDA_KERNEL(n_bl, n_th, NULL, NULL)(a, b, c);
	}
}

__forceinline__ __device__ void poly_mul_i(unsigned *a, unsigned *b, unsigned *c, unsigned n)
{
	if (n <= ALGO_N_1_CUTOFF)
	{
		for (unsigned i = 0; i < n; i++)
			c[i] = mul_mod(a[i], b[i]);
	}
	else
	{
		unsigned n_th, n_bl;
		build_launch_param(n, n_th, n_bl);
		vector_mul_i CUDA_KERNEL(n_bl, n_th, NULL, NULL)(a, b, c);
	}
}

__global__ void end_poly_mul_large(unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv) {

	// launch with 1 thread only

	poly_mul_i(t1, t2, c, 2 << log_n);

	fnt(c, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv);

}

__forceinline__ __device__ void poly_mul(unsigned *a, unsigned *b, unsigned *t1, unsigned *t2, unsigned *c, unsigned log_n,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	// 2 ^ log_n == size_a && size_a == size_b
	// *c == *a && *a + na == *b

	unsigned na = 1 << log_n, size_nc = (na << 1) * sizeof(unsigned);

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

		fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv);
		fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv);

		end_poly_mul_large CUDA_KERNEL(1, 1, NULL, NULL)(t1, t2, c, log_n, d_N_pos, d_root_layer_pow, d_inv);

	}
}

__global__ void poly_deriv(unsigned *p1, unsigned *p2)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	p2[id] = mul_mod(p1[id + 1], id + 1);
}

__global__ void build_product_i(unsigned *p, unsigned *t1, unsigned *t2, unsigned i, 
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned st = id << (i + 1), len = 1 << i;
	poly_mul(p + st, p + st + len, t1 + st, t2 + st, p + st, i, d_N_pos, d_root_layer_pow, d_inv);
}

void build_product(unsigned *p, unsigned *t1, unsigned *t2, unsigned log_n1, unsigned log_n2)
{

	// p, t1, t2 in device

	unsigned n = 1 << log_n2;
	for (unsigned i = log_n1; i < log_n2; i++)
	{
		unsigned m = n >> (i + 1);
		unsigned n_th, n_bl;
		build_launch_param(m, n_th, n_bl);
		build_product_i CUDA_KERNEL(n_bl, n_th, NULL, NULL)(p, t1, t2, i, d_N_pos, d_root_layer_pow, d_inv);
	}
}

void build_ax(unsigned *x, unsigned *p, unsigned *t1, unsigned *t2)
{

	// p, t1, t2 in device
	// x in host

	for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		unsigned st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 2;
		cudaMemcpy(p + st_p1, d_packet_product + st_p2, SIZE_ONE_PACKET_PRODUCT, cudaMemcpyDeviceToDevice);
	}
	build_product(p, t1, t2, LOG_SYMBOL + 1, MAX_LOG);
}

__global__ void build_n1(unsigned *n1, unsigned *vdax, unsigned *x, unsigned *y, 
	unsigned *d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	n1[id] = div_mod(y[id], vdax[x[id]], d_inv);

}

__global__ void build_n2n3(unsigned *n2, unsigned *n3, unsigned *n1, unsigned *x, 
	unsigned *d_root_pow)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned j = id << 1;
	n2[j] = n1[id];
	n2[j | 1] = 0;
	n3[j] = sub_mod(0, d_root_pow[x[id]]);
	n3[j | 1] = 1;

}

__global__ void build_num_large_i(unsigned* t1, unsigned* t2, unsigned* n2, unsigned n_len) {

	// launch with 1 thread only
	poly_add(t1, t2, n2, n_len);

}

__global__ void build_num_small_i(unsigned* t1, unsigned* t3, unsigned* t4, unsigned* n2, unsigned* n3, 
	unsigned i, unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv) {

	// launch with 1 thread only
	unsigned len = 1 << i;
	unsigned size_len = len * sizeof(unsigned);
	memcpy(t1, n2, size_len);
	memcpy(t1 + len, n3, size_len);
	poly_mul(t1, t1 + len, t3, t4, t1, i, d_N_pos, d_root_layer_pow, d_inv);

}

__global__ void build_den_i(unsigned* t1, unsigned* t2, unsigned* n3, 
	unsigned i, unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv) {

	// launch with 1 thread only
	unsigned len = 1 << i;
	poly_mul(n3, n3 + len, t1, t2, n3, i, d_N_pos, d_root_layer_pow, d_inv);

}

__global__ void build_px_i(unsigned *n2, unsigned *n3, unsigned *t1, unsigned *t2, unsigned *t3, unsigned *t4, 
	unsigned i, unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned len = 1 << i, n_len = len << 1;
	unsigned st = id << (i + 1);

	if (len > ALGO_N_2_CUTOFF) {

		build_num_small_i CUDA_KERNEL(1, 1, NULL, NULL)(t1 + st, t3 + st, t4 + st, n2 + st, n3 + st + len,
			i, d_N_pos, d_root_layer_pow, d_inv);

		build_num_small_i CUDA_KERNEL(1, 1, NULL, NULL)(t2 + st, t3 + st, t4 + st, n2 + st + len, n3 + st,
			i, d_N_pos, d_root_layer_pow, d_inv);

		build_num_large_i CUDA_KERNEL(1, 1, NULL, NULL)(t1 + st, t2 + st, n2 + st, n_len);
		
		build_den_i CUDA_KERNEL(1, 1, NULL, NULL)(t1 + st, t2 + st, n3 + st,
			i, d_N_pos, d_root_layer_pow, d_inv);

	}
	else {
		
		// TODO: make another func later..

		unsigned size_len = len * sizeof(unsigned);

		memcpy(t1 + st, n2 + st, size_len);
		memcpy(t1 + st + len, n3 + st + len, size_len);
		poly_mul(t1 + st, t1 + st + len, t3 + st, t4 + st, t1 + st, i, d_N_pos, d_root_layer_pow, d_inv);

		memcpy(t2 + st, n2 + st + len, size_len);
		memcpy(t2 + st + len, n3 + st, size_len);
		poly_mul(t2 + st, t2 + st + len, t3 + st, t4 + st, t2 + st, i, d_N_pos, d_root_layer_pow, d_inv);

		poly_add(t1 + st, t2 + st, n2 + st, n_len);

		poly_mul(n3 + st, n3 + st + len, t1 + st, t2 + st, n3 + st, i, d_N_pos, d_root_layer_pow, d_inv);

	}

}

void build_px(unsigned *n2, unsigned *n3, unsigned *t1, unsigned *t2, unsigned *t3, unsigned *t4)
{

	for (unsigned i = 1; i < MAX_LOG; i++)
	{
		unsigned m = NUM_OF_NEED_SYMBOL >> i;
		unsigned n_th, n_bl;
		build_launch_param(m, n_th, n_bl);
		build_px_i CUDA_KERNEL(n_bl, n_th, NULL, NULL)(n2, n3, t1, t2, t3, t4, i, d_N_pos, d_root_layer_pow, d_inv);
	}
}

void encode(unsigned *p, unsigned *y)
{

	unsigned *d_p;
	unsigned *d_y;
	cudaMalloc(&d_p, SIZE_SMALL);
	cudaMalloc(&d_y, SIZE_LARGE);
	cudaMemcpy(d_p, p, SIZE_SMALL, cudaMemcpyHostToDevice);
	cudaMemset(d_y, 0, SIZE_LARGE);

	fnt(d_p, d_y, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv);

	cudaMemcpy(y, d_y, SIZE_LARGE, cudaMemcpyDeviceToHost);

	cudaFree(d_p);
	cudaFree(d_y);
}

void decode(unsigned *x, unsigned *y, unsigned *p)
{

	unsigned n_th, n_bl;
	build_launch_param(NUM_OF_NEED_SYMBOL, n_th, n_bl);

	unsigned *t1;
	unsigned *t2;
	cudaMalloc(&t1, SIZE_LARGE);
	cudaMalloc(&t2, SIZE_LARGE);

	unsigned *ax;
	cudaMalloc(&ax, SIZE_LARGE);
	build_ax(x, ax, t1, t2);

	unsigned *dax;
	cudaMalloc(&dax, SIZE_SMALL);
	poly_deriv CUDA_KERNEL(n_bl, n_th, NULL, NULL)(ax, dax);

	unsigned *vdax;
	cudaMalloc(&vdax, SIZE_LARGE);
	cudaMemset(vdax, 0, SIZE_LARGE);
	fnt(dax, vdax, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv);

	unsigned *n1;
	cudaMalloc(&n1, SIZE_SMALL);
	cudaMemcpy(t1, x, SIZE_SMALL, cudaMemcpyHostToDevice);
	cudaMemcpy(t2, y, SIZE_SMALL, cudaMemcpyHostToDevice);
	build_n1 CUDA_KERNEL(n_bl, n_th, NULL, NULL)(n1, vdax, t1, t2, d_inv);

	unsigned *n2;
	unsigned *n3;
	cudaMalloc(&n2, SIZE_LARGE);
	cudaMalloc(&n3, SIZE_LARGE);
	build_n2n3 CUDA_KERNEL(n_bl, n_th, NULL, NULL)(n2, n3, n1, t1, d_root_pow);

	build_px(n2, n3, t1, t2, vdax, ax);

	cudaMemcpy(p, n2, SIZE_SMALL, cudaMemcpyDeviceToHost);

	cudaFree(t1);
	cudaFree(t2);
	cudaFree(ax);
	cudaFree(dax);
	cudaFree(vdax);
	cudaFree(n1);
	cudaFree(n2);
	cudaFree(n3);
}

void init()
{
	// offline process
	// TODO: async mem malloc and copy
	// TODO: change runtime limit later..

	cudaDeviceReset();
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, NUM_OF_NEED_SYMBOL << 3);

	unsigned size_N_pos = LEN_N_POS * sizeof(unsigned);
	unsigned *N_pos = (unsigned *)malloc(size_N_pos);
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
	unsigned *root_pow = (unsigned *)malloc(size_root);
	unsigned *root_inv_pow = (unsigned *)malloc(size_root);
	unsigned *inv = (unsigned *)malloc(size_root);
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
	unsigned *root_layer_pow = (unsigned *)malloc(size_root_layer_pow);
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
	unsigned *packet_product = (unsigned *)malloc(size_packet_product);
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
	unsigned *tmp;
	cudaMalloc(&tmp, (LEN_ONE_PACKET_PRODUCT << 1) * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		build_product(d_packet_product + st, tmp, tmp + LEN_ONE_PACKET_PRODUCT, 1, LOG_SYMBOL + 1);
	}
	cudaFree(tmp);
	free(inv);
	free(root_pow);
	free(root_inv_pow);
}

void fin()
{
	// clear cuda memory

	/*cudaFree(d_N_pos);

	cudaFree(d_root_pow);
	cudaFree(d_root_inv_pow);
	cudaFree(d_inv);



	cudaFree(d_root_layer_pow);
	cudaFree(d_packet_product);*/

	cudaDeviceReset();

}

void test_fnt();

void test_poly_mul();

void test_build_init_product();

void test_encode_decode();

int main()
{

	init();

	test_fnt();

	test_poly_mul();

	test_build_init_product();

	test_encode_decode();

	fin();

	return 0;
}

void test_fnt() {

	unsigned N_test = 20;

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

		fnt(d_c1, d_v, log_nc, log_nv, 0, d_N_pos, d_root_layer_pow, d_inv);
		fnt(d_v, d_c2, log_nv, log_nv, 3, d_N_pos, d_root_layer_pow, d_inv);

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

__global__ void poly_mul_wrapper(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n, 
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv) {

	unsigned n = 1 << log_n;
	assert(a == c);
	assert(b == c + n);
	// wrapper to test poly_mul(), launch with 1 thread only
	poly_mul(a, b, t1, t2, c, log_n, d_N_pos, d_root_layer_pow, d_inv);

}

void test_poly_mul() {

	srand(time(NULL));

	unsigned N_test = 10;

	for (unsigned tt = 0; tt < N_test; tt++) {

		unsigned log_n = 10;
		unsigned n = 1 << log_n, size_n = n * sizeof(unsigned);

		std::vector<unsigned> a(n), b(n), c1(n << 1, 0), c2(n << 1, 0);

		for (unsigned i = 0; i < n; i++) {
			a[i] = rand() % (MOD - 1); // 2 bytes
			b[i] = rand() % (MOD - 1); // 2 bytes
		}

		unsigned* t1, *t2, * d_c;
		cudaMalloc(&t1, size_n << 1);
		cudaMalloc(&t2, size_n << 1);
		cudaMalloc(&d_c, size_n << 1);
		cudaMemcpy(d_c, a.data(), size_n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c + n, b.data(), size_n, cudaMemcpyHostToDevice);
		poly_mul_wrapper CUDA_KERNEL(1, 1, NULL, NULL)(d_c, d_c + n, t1, t2, d_c, log_n, d_N_pos, d_root_layer_pow, d_inv);

		for (unsigned i = 0; i < n; i++)
			for (unsigned j = 0; j < n; j++)
				c1[i + j] = add_mod(c1[i + j], mul_mod(a[i], b[j]));

		cudaMemcpy(c2.data(), d_c, size_n << 1, cudaMemcpyDeviceToHost);

		for (unsigned i = 0; i < (n << 1); i++)
			assert(c1[i] == c2[i]);
		
		std::cout << "Poly mul test " << tt << " passed!" << std::endl;
	}

}

void test_encode_decode() {

	srand(time(NULL));

	unsigned N_test = 10;

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
		std::cout << "EncodeDecode test " << tt << " passed!" << std::endl;
	}

}
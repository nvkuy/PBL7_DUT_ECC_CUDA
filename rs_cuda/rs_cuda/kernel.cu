#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


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
const unsigned ALGO_N_2_CUTOFF = WARP_SIZE;

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

const unsigned SIZE_SMALL = NUM_OF_NEED_SYMBOL * sizeof(unsigned);
const unsigned SIZE_LARGE = SIZE_SMALL << 1;

__device__ unsigned *d_N_pos;
__device__ unsigned *d_root_pow;
__device__ unsigned *d_root_inv_pow;
__device__ unsigned *d_inv;
__device__ unsigned *d_root_layer_pow;
__device__ unsigned *d_packet_product;
unsigned* inv;

__host__ __device__ __forceinline__ inline unsigned mul_mod(unsigned a, unsigned b)
{
	// TODO: use 32-bit simd
	if (a == SPECIAL && b == SPECIAL)
		return 1; // overflow
	return (a * b) % MOD;
}

__host__ __device__ __forceinline__ inline unsigned div_mod(unsigned a, unsigned b)
{
	#ifdef __CUDA_ARCH__
		return mul_mod(a, d_inv[b]);
	#else
		return mul_mod(a, inv[b]);
	#endif
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

__global__ void pre_fnt(unsigned *a, unsigned *b, unsigned n)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	b[d_N_pos[n + id - 1]] = a[id];
}

__global__ void end_fnt(unsigned *a, unsigned *b, unsigned n)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	b[id << 1] = div_mod(b[id << 1], n);
	b[(id << 1) & 1] = div_mod(b[(id << 1) & 1], n);
}

__global__ void fnt_i(unsigned *a, unsigned *b, unsigned i, bool inv)
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

__host__ __device__ void fnt(unsigned *a, unsigned *b, unsigned log_n, bool inv)
{

	// 2 ^ log_n == size_b && size_b == size_a * 2 && num_all_thread == size_a

	/*
	opt 2 bit: x1 x2
	 - x1: w_n or 1 / w_n
	 - x2: need result / n
	*/

	unsigned n = 1 << log_n;
	unsigned n_th = log_n > 9 ? 256 : 32;
	unsigned n_bl = (n >> 1) / n_th;

	pre_fnt<<<n_bl, n_th, NULL, cudaStreamTailLaunch>>>(a, b, n);

	for (unsigned i = 0; i < log_n; i++)
		fnt_i<<<n_bl, n_th, NULL, cudaStreamTailLaunch>>>(a, b, i, inv);

	if (inv)
		end_fnt<<<n_bl, n_th, NULL, cudaStreamTailLaunch>>>(a, b, n);
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

__device__ void poly_add(unsigned *a, unsigned *b, unsigned *c, unsigned n)
{
	if (n < ALGO_N_1_CUTOFF)
	{
		for (unsigned i = 0; i < n; i++)
			c[i] = add_mod(a[i], b[i]);
	}
	else
	{
		vector_add<<<n / 256, 256, NULL, cudaStreamTailLaunch>>>(a, b, c);
	}
}

__device__ void poly_mul_i(unsigned *a, unsigned *b, unsigned *c, unsigned n)
{
	if (n < ALGO_N_1_CUTOFF)
	{
		for (unsigned i = 0; i < n; i++)
			c[i] = mul_mod(a[i], b[i]);
	}
	else
	{
		vector_mul_i<<<n / 256, 256, NULL, cudaStreamGraphTailLaunch>>>(a, b, c);
	}
}

__device__ void poly_mul(unsigned *a, unsigned *b, unsigned *t1, unsigned *t2, unsigned *c, unsigned log_n)
{

	// 2 ^ log_n == size_a && size_a == size_b
	unsigned na = 1 << log_n, nc = na << 1, size_nc = nc * sizeof(unsigned);

	if (na < ALGO_N_2_CUTOFF)
	{
		memset(c, 0, size_nc);
		for (unsigned i = 0; i < na; i++)
		{
			t1[i] = a[i];
			t2[i] = b[i];
		}
		for (unsigned i = 0; i < na; i++)
			for (unsigned j = 0; j < na; j++)
				c[i + j] = add_mod(c[i + j], mul_mod(t1[i], t2[j]));
	}
	else
	{
		memset(t1, 0, size_nc);
		memset(t2, 0, size_nc);

		fnt(a, t1, log_n + 1, 0);
		fnt(b, t2, log_n + 1, 0);

		poly_mul_i(t1, t2, c, nc);

		fnt(c, c, log_n + 1, 1);
	}
}

__global__ void poly_deriv(unsigned *p1, unsigned *p2)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	p2[id] = mul_mod(p1[id + 1], id + 1);
}

__global__ void build_product_i(unsigned *p, unsigned *t1, unsigned *t2, unsigned i)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned st = id << (i + 1), len = 1 << i;
	poly_mul(p + st, p + st + len, t1 + st, t2 + st, p + st, i);
}

void build_product(unsigned *p, unsigned *t1, unsigned *t2, unsigned log_n1, unsigned log_n2)
{

	// p, t1, t2 in device

	unsigned n = 1 << log_n2;
	for (unsigned i = log_n1; i < log_n2; i++)
	{
		unsigned m = n >> (i + 1);
		unsigned n_th = m > 256 ? 256 : m;
		unsigned n_bl = m / n_th;
		build_product_i<<<n_bl, n_th>>>(p, t1, t2, i);
	}
}

void build_ax(unsigned *x, unsigned *p, unsigned *t1, unsigned *t2)
{

	// p, t1, t2 in device
	// x in host

	unsigned size_product = (1 << (LOG_SYMBOL + 1)) * sizeof(unsigned);
	for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		unsigned st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 1;
		cudaMemcpy(p + st_p1, d_packet_product + st_p2, size_product, cudaMemcpyDeviceToDevice);
	}
	build_product(p, t1, t2, LOG_SYMBOL + 1, MAX_LOG);
}

__global__ void build_n1(unsigned *n1, unsigned *vdax, unsigned *x, unsigned *y)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	n1[id] = div_mod(y[id], vdax[x[id]]);
}

__global__ void build_n2n3(unsigned *n2, unsigned *n3, unsigned *n1, unsigned *x)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned j = id << 1;
	n2[j] = n1[id];
	n2[j & 1] = 0;
	n3[j] = sub_mod(MOD, d_root_pow[x[id]]);
	n3[j & 1] = 1;
}

__global__ void build_px_i(unsigned *n2, unsigned *n3, unsigned *t1, unsigned *t2, unsigned *t3, unsigned *t4, unsigned i)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned len = 1 << i, n_len = len << 1;
	unsigned st = id << (i + 1);
	unsigned size_len = len * sizeof(unsigned);

	memcpy(t1 + st, n2 + st, size_len);
	memcpy(t1 + st + len, n3 + st + len, size_len);
	poly_mul(t1 + st, t1 + st + len, t3, t4, t1 + st, i);

	memcpy(t2 + st, n2 + st + len, size_len);
	memcpy(t2 + st + len, n3 + st, size_len);
	poly_mul(t2 + st, t2 + st + len, t3, t4, t1 + st, i);

	poly_add(t1 + st, t2 + st, n2 + st, n_len);

	poly_mul(n3 + st, n3 + st + len, t1, t2, n3 + st, i);
}

void build_px(unsigned *n2, unsigned *n3, unsigned *t1, unsigned *t2, unsigned *t3, unsigned *t4)
{

	// TODO: optimize space later..

	for (unsigned i = 1; i < MAX_LOG; i++)
	{
		unsigned m = NUM_OF_NEED_SYMBOL >> i;
		unsigned n_th = m > 256 ? 256 : m;
		unsigned n_bl = m / n_th;
		build_px_i<<<n_bl, n_th>>>(n2, n3, t1, t2, t3, t4, i);
	}
}

void encode(unsigned *p, unsigned *y)
{

	unsigned *d_p;
	unsigned *d_y;
	cudaMalloc(&d_p, SIZE_SMALL);
	cudaMalloc(&d_y, SIZE_LARGE);
	cudaMemcpy(d_p, p, SIZE_SMALL, cudaMemcpyHostToDevice);

	fnt(d_p, d_y, MAX_LOG, 0);

	cudaMemcpy(y, d_y, SIZE_LARGE, cudaMemcpyDeviceToHost);

	cudaFree(d_p);
	cudaFree(d_y);
}

void decode(unsigned *x, unsigned *y, unsigned *p)
{

	unsigned n_th = 512, n_bl = NUM_OF_NEED_SYMBOL / n_th;

	unsigned *t1;
	unsigned *t2;
	cudaMalloc(&t1, SIZE_LARGE);
	cudaMalloc(&t2, SIZE_LARGE);

	unsigned *ax;
	cudaMalloc(&ax, SIZE_LARGE);
	build_ax(x, ax, t1, t2);

	unsigned *dax;
	cudaMalloc(&dax, SIZE_SMALL);
	poly_deriv<<<n_bl, n_th>>>(ax, dax);

	unsigned *vdax;
	cudaMalloc(&vdax, SIZE_LARGE);
	fnt(dax, vdax, MAX_LOG, 0);

	unsigned *n1;
	cudaMalloc(&n1, SIZE_SMALL);
	cudaMemcpy(t1, x, SIZE_SMALL, cudaMemcpyHostToDevice);
	cudaMemcpy(t2, y, SIZE_SMALL, cudaMemcpyHostToDevice);
	build_n1<<<n_bl, n_th>>>(n1, vdax, t1, t2);

	unsigned *n2;
	unsigned *n3;
	cudaMalloc(&n2, SIZE_LARGE);
	cudaMalloc(&n3, SIZE_LARGE);
	build_n2n3<<<n_bl, n_th>>>(n2, n3, n1, t1);

	build_px(n2, n3, t1, t2, vdax, ax);

	cudaMemcpy(n2, p, SIZE_SMALL, cudaMemcpyDeviceToHost);

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

	unsigned *t1, *t2, *t3;

	unsigned size_N_pos = LEN_N_POS * sizeof(unsigned);
	unsigned *N_pos = (unsigned *)malloc(size_N_pos);
	cudaMalloc(&t1, size_N_pos);

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

	cudaMemcpy(t1, N_pos, size_N_pos, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_N_pos, &t1, sizeof(t1));
	cudaFree(t1);
	free(N_pos);

	unsigned size_root = MOD * sizeof(unsigned);
	unsigned *root_pow = (unsigned *)malloc(size_root);
	unsigned *root_inv_pow = (unsigned *)malloc(size_root);
	inv = (unsigned *)malloc(size_root);
	cudaMalloc(&t1, size_root);
	cudaMalloc(&t2, size_root);
	cudaMalloc(&t3, size_root);

	root_pow[0] = 1, root_inv_pow[0] = 1;
	for (unsigned i = 1; i < MOD; i++)
	{
		root_pow[i] = mul_mod(root_pow[i - 1], ROOT);
		root_inv_pow[i] = mul_mod(root_inv_pow[i - 1], ROOT_INV);
		inv[i] = pow_mod(i, MOD - 2);
	}

	cudaMemcpy(t1, root_pow, size_root, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_root_pow, &t1, sizeof(t1));
	cudaMemcpy(t2, root_inv_pow, size_root, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_root_inv_pow, &t2, sizeof(t2));
	cudaMemcpy(t3, inv, size_root, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_inv, &t3, sizeof(t3));
	cudaFree(t1);
	cudaFree(t2);
	cudaFree(t3);

	unsigned size_root_layer_pow = LEN_ROOT_LAYER_POW_2 * sizeof(unsigned);
	unsigned *root_layer_pow = (unsigned *)malloc(size_root_layer_pow);
	cudaMalloc(&t1, size_root_layer_pow);

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

	cudaMemcpy(t1, root_layer_pow, size_root_layer_pow, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_root_layer_pow, &t1, sizeof(t1));
	cudaFree(t1);
	free(root_layer_pow);

	unsigned size_packet_product = LEN_PACKET_PRODUCT * sizeof(unsigned);
	unsigned *packet_product = (unsigned *)malloc(size_packet_product);
	cudaMalloc(&t1, size_packet_product);

	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		for (unsigned j = 0; j < SEG_PER_PACKET; j++)
		{
			unsigned k = (i << LOG_SEG) + j;
			packet_product[st + (j << 1)] = sub_mod(MOD, root_pow[k]);
			packet_product[st + ((j << 1) & 1)] = 1;
			packet_product[st + ((j + SEG_PER_PACKET) << 1)] = sub_mod(MOD, root_pow[k + SEG_DIFF]);
			packet_product[st + (((j + SEG_PER_PACKET) << 1) & 1)] = 1;
		}
	}
	cudaMemcpy(t1, packet_product, size_packet_product, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_packet_product, &t1, sizeof(t1));
	cudaFree(t1);
	free(packet_product);
	unsigned len_product = 1 << (LOG_SYMBOL + 1);
	unsigned *tmp;
	cudaMalloc(&tmp, (len_product << 1) * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		build_product(d_packet_product + st, tmp, tmp + len_product, 1, LOG_SYMBOL + 1);
	}
	cudaFree(tmp);
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

void test_encode_decode();

int main()
{

	init();

	test_encode_decode();

	fin();

	return 0;
}

void test_encode_decode() {

	srand(time(0));

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
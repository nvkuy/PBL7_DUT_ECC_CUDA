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

#include <iostream>
#include <time.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK_LAST() check_last(__FILE__, __LINE__)
inline void check_last(const char* const file, const int line)
{
	cudaError_t const err{ cudaPeekAtLastError() };
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

const unsigned MOD = 65537;
const unsigned SPECIAL = MOD - 1;
const unsigned ROOT = 3;
const unsigned ROOT_INV = 21846;
const unsigned MAX_LOG = 16;

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

const unsigned SIZE_SMALL = LEN_SMALL * sizeof(unsigned);
const unsigned SIZE_LARGE = LEN_LARGE * sizeof(unsigned);
const unsigned SIZE_ONE_PACKET_PRODUCT = LEN_ONE_PACKET_PRODUCT * sizeof(unsigned);

const unsigned LOG_LEN_ENCODE_P = MAX_LOG - 1;
const unsigned LOG_LEN_ENCODE_Y = MAX_LOG;
const unsigned LOG_LEN_DECODE_X = MAX_LOG - 1;
const unsigned LOG_LEN_DECODE_Y = MAX_LOG - 1;
const unsigned LOG_LEN_DECODE_P = MAX_LOG - 1;

const unsigned LEN_ENCODE_P = 1 << LOG_LEN_ENCODE_P;
const unsigned LEN_ENCODE_Y = 1 << LOG_LEN_ENCODE_Y;
const unsigned LEN_DECODE_X = 1 << LOG_LEN_DECODE_X;
const unsigned LEN_DECODE_Y = 1 << LOG_LEN_DECODE_Y;
const unsigned LEN_DECODE_P = 1 << LOG_LEN_DECODE_P;

const unsigned SIZE_ENCODE_P = LEN_ENCODE_P * sizeof(unsigned);
const unsigned SIZE_ENCODE_Y = LEN_ENCODE_Y * sizeof(unsigned);
const unsigned SIZE_DECODE_X = LEN_DECODE_X * sizeof(unsigned);
const unsigned SIZE_DECODE_Y = LEN_DECODE_Y * sizeof(unsigned);
const unsigned SIZE_DECODE_P = LEN_DECODE_P * sizeof(unsigned);

const unsigned MAX_ACTIVE_ENCODE = 256;
const unsigned MAX_ACTIVE_DECODE = 256;
const unsigned MAX_ENCODE_LAUNCH_CNT = 16;
const unsigned MAX_DECODE_LAUNCH_CNT = 128;

const unsigned LOG_ALGO_LOW_WPT = 3;
const unsigned LOG_ALGO_MED_WPT = 2;
const unsigned LOG_ALGO_HIGH_WPT = 2;
const unsigned LOG_ALGO_EXTR_WPT = 0;

const unsigned LOG_LEN_WARP = 5;
const unsigned LEN_WARP = 1 << LOG_LEN_WARP;
const unsigned ALGO_N_2_CUTOFF = 64;

const unsigned ALGO_LOW_WPT = 1 << LOG_ALGO_LOW_WPT;
const unsigned ALGO_MED_WPT = 1 << LOG_ALGO_MED_WPT;
const unsigned ALGO_HIGH_WPT = 1 << LOG_ALGO_HIGH_WPT; 
const unsigned ALGO_EXTR_WPT = 1 << LOG_ALGO_EXTR_WPT; 

const size_t SIZE_ENCODE_P_SLOT = sizeof(unsigned) * LEN_ENCODE_P * MAX_ACTIVE_ENCODE;
const size_t SIZE_ENCODE_Y_SLOT = sizeof(unsigned) * LEN_ENCODE_Y * MAX_ACTIVE_ENCODE;

const size_t SIZE_DECODE_X_SLOT = sizeof(unsigned) * LEN_DECODE_X * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_Y_SLOT = sizeof(unsigned) * LEN_DECODE_Y * MAX_ACTIVE_DECODE;
//const size_t SIZE_DECODE_P_SLOT = sizeof(unsigned) * LEN_DECODE_P * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_T1_SLOT = sizeof(unsigned) * LEN_LARGE * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_T2_SLOT = sizeof(unsigned) * LEN_LARGE * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_AX_SLOT = sizeof(unsigned) * LEN_LARGE * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_DAX_SLOT = sizeof(unsigned) * LEN_SMALL * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_VDAX_SLOT = sizeof(unsigned) * LEN_LARGE * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_N1_SLOT = sizeof(unsigned) * LEN_SMALL * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_N2_SLOT = sizeof(unsigned) * LEN_LARGE * MAX_ACTIVE_DECODE;
const size_t SIZE_DECODE_N3_SLOT = sizeof(unsigned) * LEN_SMALL * MAX_ACTIVE_DECODE;

unsigned** h_encode_p_slot;
unsigned** h_encode_y_slot;
unsigned** h_decode_x_slot;
unsigned** h_decode_y_slot;
unsigned** h_decode_p_slot;

unsigned* d_encode_p_slot;
unsigned* d_encode_y_slot;
unsigned* d_decode_x_slot;
unsigned* d_decode_y_slot;
//unsigned* d_decode_p_slot;
unsigned* d_decode_t1_slot;
unsigned* d_decode_t2_slot;
unsigned* d_decode_ax_slot;
unsigned* d_decode_dax_slot;
unsigned* d_decode_vdax_slot;
unsigned* d_decode_n1_slot;
unsigned* d_decode_n2_slot;
unsigned* d_decode_n3_slot;

unsigned* d_N_pos;
unsigned* d_root_pow;
unsigned* d_root_inv_pow;
unsigned* d_inv;
unsigned* d_root_layer_pow;
unsigned* d_packet_product;

struct CB_DATA {
	unsigned slot_id;
	unsigned* dst;
	unsigned* src;
	size_t size_res;
	std::queue<unsigned>& slot; 
	std::mutex& mt; 
	std::condition_variable& cv;
};

std::queue<unsigned> encode_slot, decode_slot;
std::mutex mt_encode_slot, mt_decode_slot;
std::condition_variable cv_encode_slot, cv_decode_slot;

inline unsigned pop_slot(std::queue<unsigned> &slot, std::mutex &mt, std::condition_variable &cv) {
	std::unique_lock<std::mutex> lock(mt);
	cv.wait(lock, [&] { return !slot.empty(); });
	unsigned id = slot.front();
	slot.pop();
	return id;
}

inline void push_slot(unsigned id, std::queue<unsigned>& slot, std::mutex& mt, std::condition_variable& cv) {
	{
		std::lock_guard<std::mutex> lock(mt);
		slot.push(id);
	}
	cv.notify_one();
}

void init_batch_slot() {

	for (unsigned i = 0; i < MAX_ACTIVE_ENCODE; i++)
		push_slot(i, encode_slot, mt_encode_slot, cv_encode_slot);

	for (unsigned i = 0; i < MAX_ACTIVE_DECODE; i++)
		push_slot(i, decode_slot, mt_decode_slot, cv_decode_slot);

}

void CUDART_CB h_end_batch_slot(void* data) {

	CB_DATA* dat = static_cast<CB_DATA*>(data);

	memcpy(dat->dst, dat->src, dat->size_res);
	push_slot(dat->slot_id, dat->slot, dat->mt, dat->cv);

	delete dat;
}

__host__ __device__ __forceinline__ inline void build_launch_param(unsigned log_n, unsigned& n_th, unsigned& n_bl) {
	if (log_n <= LOG_LEN_WARP) {
		n_th = 1 << log_n;
		n_bl = 1;
	}
	else {
		unsigned c_l2_sqrt_n = (log_n >> 1) + (log_n & 1);
		n_th = 1 << c_l2_sqrt_n;
		n_bl = 1 << (log_n - c_l2_sqrt_n);
	}
}

__host__ __device__ __forceinline__ inline unsigned mul_mod(unsigned a, unsigned b)
{
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

__global__ void g_pre_fnt(unsigned* a, unsigned* b, unsigned st, unsigned* d_N_pos)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_HIGH_WPT, id_r = id_l + ALGO_HIGH_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		b[d_N_pos[st + k]] = a[k];

}

__global__ void g_end_fnt(unsigned* b, unsigned n, unsigned* d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_HIGH_WPT, id_r = id_l + ALGO_HIGH_WPT;
	for (unsigned k = id_l; k < id_r; k++) {
		b[k << 1] = div_mod(b[k << 1], n, d_inv);
		b[(k << 1) | 1] = div_mod(b[(k << 1) | 1], n, d_inv);
	}
}

__global__ void g_fnt_i(unsigned* b, unsigned i, bool inv,
	unsigned* d_root_layer_pow)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_HIGH_WPT, id_r = id_l + ALGO_HIGH_WPT;

	unsigned haft_len = 1 << i;
	for (unsigned k = id_l; k < id_r; k++) {
		unsigned bl_st = ((k >> i) << (i + 1)), th_id = (k & (haft_len - 1));
		unsigned pos = bl_st + th_id;
		unsigned u = b[pos];
		unsigned v = mul_mod(b[pos + haft_len], d_root_layer_pow[(LEN_ROOT_LAYER_POW * inv) + haft_len - 1 + th_id]);
		b[pos] = add_mod(u, v);
		b[pos + haft_len] = sub_mod(u, v);
	}

}

__host__ __forceinline__ __device__ inline void fnt(unsigned* a, unsigned* b, unsigned log_na, unsigned log_nb, unsigned opt,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, cudaStream_t stream)
{

	/*
	opt 2 bit: x1 x2
	 - x1: w_n or 1/w_n
	 - x2: need result * 1/n
	*/

	// size_b >= size_a;
	// need memset *b before use unless size_a == size_b

	unsigned nb = 1 << log_nb, wp = (opt & 2) >> 1;
	unsigned n_bl, n_th;

	build_launch_param(log_na - LOG_ALGO_HIGH_WPT, n_th, n_bl);
	g_pre_fnt CUDA_KERNEL(n_bl, n_th, NULL, stream)(a, b, nb - 1, d_N_pos);

	build_launch_param(log_nb - 1 - LOG_ALGO_HIGH_WPT, n_th, n_bl);
	for (unsigned i = 0; i < log_nb; i++)
		g_fnt_i CUDA_KERNEL(n_bl, n_th, NULL, stream)(b, i, wp, d_root_layer_pow);

	if (opt & 1)
		g_end_fnt CUDA_KERNEL(n_bl, n_th, NULL, stream)(b, nb, d_inv);
}

__global__ void g_vector_mul_i(unsigned* a, unsigned* b, unsigned* c)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_LOW_WPT, id_r = id_l + ALGO_LOW_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		c[k] = mul_mod(a[k], b[k]);


}

__global__ void g_fill(unsigned* a, unsigned val) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_LOW_WPT, id_r = id_l + ALGO_LOW_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		a[k] = val;

}

__forceinline__ __device__ void d_poly_mul(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
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

		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

		unsigned n_bl, n_th;
		build_launch_param(log_n + 1 - LOG_ALGO_LOW_WPT, n_th, n_bl);
		g_fill CUDA_KERNEL(n_bl, n_th, NULL, stream) (t1, 0);
		g_fill CUDA_KERNEL(n_bl, n_th, NULL, stream) (t2, 0);

		fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
		fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);

		g_vector_mul_i CUDA_KERNEL(n_bl, n_th, NULL, stream)(t1, t2, t1);

		fnt(t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, stream);

		cudaStreamDestroy(stream);

	}
}

inline void h_poly_mul(unsigned* a, unsigned* b, unsigned* t1, unsigned* t2, unsigned* c, unsigned log_n,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, cudaStream_t stream) {

	// only use with large poly

	unsigned nc = 1 << (log_n + 1), size_nc = nc * sizeof(unsigned);

	unsigned n_bl, n_th;
	build_launch_param(log_n + 1 - LOG_ALGO_LOW_WPT, n_th, n_bl);
	g_fill CUDA_KERNEL(n_bl, n_th, NULL, stream) (t1, 0);
	g_fill CUDA_KERNEL(n_bl, n_th, NULL, stream) (t2, 0);

	fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);

	g_vector_mul_i CUDA_KERNEL(n_bl, n_th, NULL, stream)(t1, t2, t1);

	fnt(t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, stream);

}

__global__ void g_poly_deriv(unsigned* ax, unsigned* dax)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_MED_WPT, id_r = id_l + ALGO_MED_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		dax[k] = mul_mod(ax[k + 1], k + 1);
}

__global__ void g_build_product_i(unsigned* p, unsigned* t1, unsigned* t2, unsigned i,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned len = 1 << i;

	unsigned id_l = id * ALGO_EXTR_WPT, id_r = id_l + ALGO_EXTR_WPT;
	for (unsigned k = id_l; k < id_r; k++) {
		unsigned st = k << (i + 1);
		d_poly_mul(p + st, p + st + len, t1 + st, t2 + st, p + st, i, d_N_pos, d_root_layer_pow, d_inv);
	}

}

inline void h_build_product(unsigned* p, unsigned* t1, unsigned* t2, unsigned log_n1, unsigned log_n2, cudaStream_t stream)
{

	// p, t1, t2 in device

	for (unsigned i = log_n1; i < log_n2; i++)
	{
		unsigned n_th, n_bl;
		build_launch_param(log_n2 - i - 1 - LOG_ALGO_EXTR_WPT, n_th, n_bl);
		g_build_product_i CUDA_KERNEL(n_bl, n_th, NULL, stream)(p, t1, t2, i, d_N_pos, d_root_layer_pow, d_inv);
		CUDA_CHECK_LAST();
	}
}

inline void h_build_ax(unsigned* x, unsigned* p, unsigned* t1, unsigned* t2, cudaStream_t stream)
{

	// p, t1, t2 in device
	// x in host

	for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		unsigned st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 2;
		CUDA_CHECK(cudaMemcpyAsync(p + st_p1, d_packet_product + st_p2, SIZE_ONE_PACKET_PRODUCT, cudaMemcpyDeviceToDevice, stream));
	}
	h_build_product(p, t1, t2, LOG_SYMBOL + 1, MAX_LOG, stream);
}

__global__ void g_build_n1(unsigned* n1, unsigned* vdax, unsigned* x, unsigned* y,
	unsigned* d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_MED_WPT, id_r = id_l + ALGO_MED_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		n1[k] = div_mod(y[k], vdax[x[k]], d_inv);

}

__global__ void g_build_n2(unsigned* n2, unsigned* n1, unsigned* x) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_MED_WPT, id_r = id_l + ALGO_MED_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		n2[x[k]] = n1[k];

}

__global__ void g_build_n3(unsigned* n3, unsigned* p_n3) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id_l = id * ALGO_MED_WPT, id_r = id_l + ALGO_MED_WPT;
	for (unsigned k = id_l; k < id_r; k++)
		n3[k] = sub_mod(0, p_n3[k + 1]);

}

inline void h_build_px(unsigned* p, unsigned* ax, unsigned* n3, unsigned* t1, unsigned* t2,
	unsigned* d_N_pos, unsigned* d_root_layer_pow, unsigned* d_inv, cudaStream_t stream) {

	h_poly_mul(ax, n3, t1, t2, p, MAX_LOG - 1, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();

}

void h_encode(unsigned* p, unsigned* y)
{

	unsigned slot_id = pop_slot(encode_slot, mt_encode_slot, cv_encode_slot);

	unsigned* sl_p = h_encode_p_slot[slot_id];
	unsigned* sl_y = h_encode_y_slot[slot_id];

	memcpy(sl_p, p, SIZE_ENCODE_P);

	unsigned* d_p = d_encode_p_slot + 1LL * slot_id * LEN_ENCODE_P;
	unsigned* d_y = d_encode_y_slot + 1LL * slot_id * LEN_ENCODE_Y;

	unsigned n_th, n_bl;
	build_launch_param(LOG_LEN_ENCODE_Y - LOG_ALGO_LOW_WPT, n_th, n_bl);

	CB_DATA* data = new CB_DATA{ slot_id, y, sl_y, SIZE_ENCODE_Y, encode_slot, mt_encode_slot, cv_encode_slot };

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	CUDA_CHECK(cudaMemcpyAsync(d_p, sl_p, SIZE_ENCODE_P, cudaMemcpyHostToDevice, stream));
	
	g_fill CUDA_KERNEL(n_bl, n_th, NULL, stream)(d_y, 0);

	fnt(d_p, d_y, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();

	CUDA_CHECK(cudaMemcpyAsync(sl_y, d_y, SIZE_ENCODE_Y, cudaMemcpyDeviceToHost, stream));
	
	CUDA_CHECK(cudaLaunchHostFunc(stream, h_end_batch_slot, data));

	CUDA_CHECK(cudaStreamDestroy(stream));

}

void h_decode(unsigned* x, unsigned* y, unsigned* p)
{

	unsigned slot_id = pop_slot(decode_slot, mt_decode_slot, cv_decode_slot);

	unsigned* sl_x = h_decode_x_slot[slot_id];
	unsigned* sl_y = h_decode_y_slot[slot_id];
	unsigned* sl_p = h_decode_p_slot[slot_id];

	memcpy(sl_x, x, SIZE_DECODE_X);
	memcpy(sl_y, y, SIZE_DECODE_Y);

	unsigned* d_x = d_decode_x_slot + 1LL * slot_id * LEN_DECODE_X;
	unsigned* d_y = d_decode_y_slot + 1LL * slot_id * LEN_DECODE_Y;
	unsigned* d_t1 = d_decode_t1_slot + 1LL * slot_id * LEN_LARGE;
	unsigned* d_t2 = d_decode_t2_slot + 1LL * slot_id * LEN_LARGE;
	unsigned* d_ax = d_decode_ax_slot + 1LL * slot_id * LEN_LARGE;
	unsigned* d_dax = d_decode_dax_slot + 1LL * slot_id * LEN_SMALL;
	unsigned* d_vdax = d_decode_vdax_slot + 1LL * slot_id * LEN_LARGE;
	unsigned* d_n1 = d_decode_n1_slot + 1LL * slot_id * LEN_SMALL;
	unsigned* d_n2 = d_decode_n2_slot + 1LL * slot_id * LEN_LARGE;
	unsigned* d_n3 = d_decode_n3_slot + 1LL * slot_id * LEN_SMALL;

	unsigned n_th1, n_bl1, n_th2, n_bl2;
	build_launch_param(MAX_LOG - LOG_ALGO_MED_WPT - 1, n_th1, n_bl1);
	build_launch_param(MAX_LOG - LOG_ALGO_LOW_WPT, n_th2, n_bl2);

	CB_DATA* data = new CB_DATA{ slot_id, p, sl_p, SIZE_DECODE_P, decode_slot, mt_decode_slot, cv_decode_slot };

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	CUDA_CHECK(cudaMemcpyAsync(d_x, sl_x, SIZE_DECODE_X, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_y, sl_y, SIZE_DECODE_Y, cudaMemcpyHostToDevice, stream));

	g_fill CUDA_KERNEL(n_bl2, n_th2, NULL, stream)(d_vdax, 0);
	g_fill CUDA_KERNEL(n_bl2, n_th2, NULL, stream)(d_n2, 0);

	h_build_ax(x, d_ax, d_t1, d_t2, stream);

	g_poly_deriv CUDA_KERNEL(n_bl1, n_th1, NULL, stream)(d_ax, d_dax);
	CUDA_CHECK_LAST();
	
	fnt(d_dax, d_vdax, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
	
	g_build_n1 CUDA_KERNEL(n_bl1, n_th1, NULL, stream)(d_n1, d_vdax, d_x, d_y, d_inv);
	CUDA_CHECK_LAST();
	
	g_build_n2 CUDA_KERNEL(n_bl1, n_th1, NULL, stream)(d_n2, d_n1, d_x);
	CUDA_CHECK_LAST();
	
	fnt(d_n2, d_t2, MAX_LOG, MAX_LOG, 2, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
	g_build_n3 CUDA_KERNEL(n_bl1, n_th1, NULL, stream)(d_n3, d_t2);
	CUDA_CHECK_LAST();
	
	h_build_px(d_n2, d_ax, d_n3, d_t1, d_t2, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK(cudaMemcpyAsync(sl_p, d_n2, SIZE_SMALL, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaLaunchHostFunc(stream, h_end_batch_slot, data));

	CUDA_CHECK(cudaStreamDestroy(stream));

}

void init()
{
	// offline process

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaDeviceReset());
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 
		std::max(MAX_ACTIVE_DECODE * MAX_DECODE_LAUNCH_CNT, MAX_ACTIVE_ENCODE * MAX_ENCODE_LAUNCH_CNT)));

	h_encode_p_slot = (unsigned**)malloc(MAX_ACTIVE_ENCODE * sizeof(unsigned*));
	h_encode_y_slot = (unsigned**)malloc(MAX_ACTIVE_ENCODE * sizeof(unsigned*));
	h_decode_x_slot = (unsigned**)malloc(MAX_ACTIVE_DECODE * sizeof(unsigned*));
	h_decode_y_slot = (unsigned**)malloc(MAX_ACTIVE_DECODE * sizeof(unsigned*));
	h_decode_p_slot = (unsigned**)malloc(MAX_ACTIVE_DECODE * sizeof(unsigned*));

	for (unsigned i = 0; i < MAX_ACTIVE_ENCODE; i++) {
		CUDA_CHECK(cudaMallocHost(&(h_encode_p_slot[i]), SIZE_ENCODE_P));
		CUDA_CHECK(cudaMallocHost(&(h_encode_y_slot[i]), SIZE_ENCODE_Y));
	}

	for (unsigned i = 0; i < MAX_ACTIVE_DECODE; i++) {
		CUDA_CHECK(cudaMallocHost(&(h_decode_x_slot[i]), SIZE_DECODE_X));
		CUDA_CHECK(cudaMallocHost(&(h_decode_y_slot[i]), SIZE_DECODE_Y));
		CUDA_CHECK(cudaMallocHost(&(h_decode_p_slot[i]), SIZE_DECODE_P));
	}

	CUDA_CHECK(cudaMalloc(&d_encode_p_slot, SIZE_ENCODE_P_SLOT));
	CUDA_CHECK(cudaMalloc(&d_encode_y_slot, SIZE_ENCODE_Y_SLOT));

	CUDA_CHECK(cudaMalloc(&d_decode_x_slot, SIZE_DECODE_X_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_y_slot, SIZE_DECODE_Y_SLOT));
	//CUDA_CHECK(cudaMalloc(&d_decode_p_slot, SIZE_DECODE_P_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_t1_slot, SIZE_DECODE_T1_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_t2_slot, SIZE_DECODE_T2_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_ax_slot, SIZE_DECODE_AX_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_dax_slot, SIZE_DECODE_DAX_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_vdax_slot, SIZE_DECODE_VDAX_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_n1_slot, SIZE_DECODE_N1_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_n2_slot, SIZE_DECODE_N2_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_n3_slot, SIZE_DECODE_N3_SLOT));

	init_batch_slot();

	unsigned size_N_pos = LEN_N_POS * sizeof(unsigned);
	unsigned* N_pos = (unsigned*)malloc(size_N_pos);
	CUDA_CHECK(cudaMalloc(&d_N_pos, size_N_pos));

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

	CUDA_CHECK(cudaMemcpy(d_N_pos, N_pos, size_N_pos, cudaMemcpyHostToDevice));
	free(N_pos);

	unsigned size_root = MOD * sizeof(unsigned);
	unsigned* root_pow = (unsigned*)malloc(size_root);
	unsigned* root_inv_pow = (unsigned*)malloc(size_root);
	unsigned* inv = (unsigned*)malloc(size_root);
	CUDA_CHECK(cudaMalloc(&d_root_pow, size_root));
	CUDA_CHECK(cudaMalloc(&d_root_inv_pow, size_root));
	CUDA_CHECK(cudaMalloc(&d_inv, size_root));

	root_pow[0] = 1, root_inv_pow[0] = 1, inv[0] = 0;
	for (unsigned i = 1; i < MOD; i++)
	{
		root_pow[i] = mul_mod(root_pow[i - 1], ROOT);
		root_inv_pow[i] = mul_mod(root_inv_pow[i - 1], ROOT_INV);
		inv[i] = pow_mod(i, MOD - 2);
	}

	CUDA_CHECK(cudaMemcpy(d_root_pow, root_pow, size_root, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_root_inv_pow, root_inv_pow, size_root, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_inv, inv, size_root, cudaMemcpyHostToDevice));

	unsigned size_root_layer_pow = LEN_ROOT_LAYER_POW_2 * sizeof(unsigned);
	unsigned* root_layer_pow = (unsigned*)malloc(size_root_layer_pow);
	CUDA_CHECK(cudaMalloc(&d_root_layer_pow, size_root_layer_pow));

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

	CUDA_CHECK(cudaMemcpy(d_root_layer_pow, root_layer_pow, size_root_layer_pow, cudaMemcpyHostToDevice));
	free(root_layer_pow);

	unsigned size_packet_product = LEN_PACKET_PRODUCT * sizeof(unsigned);
	unsigned* packet_product = (unsigned*)malloc(size_packet_product);
	CUDA_CHECK(cudaMalloc(&d_packet_product, size_packet_product));

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
	CUDA_CHECK(cudaMemcpy(d_packet_product, packet_product, size_packet_product, cudaMemcpyHostToDevice));
	free(packet_product);
	unsigned* tmp;
	CUDA_CHECK(cudaMalloc(&tmp, (LEN_ONE_PACKET_PRODUCT << 1) * sizeof(unsigned)));
	for (unsigned i = 0; i < NUM_OF_PACKET; i++)
	{
		unsigned st = i << (LOG_SYMBOL + 1);
		h_build_product(d_packet_product + st, tmp, tmp + LEN_ONE_PACKET_PRODUCT, 1, LOG_SYMBOL + 1, NULL);
	}
	CUDA_CHECK(cudaFree(tmp));
	free(inv);
	free(root_pow);
	free(root_inv_pow);

	CUDA_CHECK(cudaDeviceSynchronize());
	std::cout << "Init process completed!" << std::endl;

}

void fin()
{
	// clear cuda memory

	CUDA_CHECK(cudaDeviceSynchronize());

	for (unsigned i = 0; i < MAX_ACTIVE_ENCODE; i++) {
		CUDA_CHECK(cudaFreeHost(h_encode_p_slot[i]));
		CUDA_CHECK(cudaFreeHost(h_encode_y_slot[i]));
	}

	for (unsigned i = 0; i < MAX_ACTIVE_DECODE; i++) {
		CUDA_CHECK(cudaFreeHost(h_decode_x_slot[i]));
		CUDA_CHECK(cudaFreeHost(h_decode_y_slot[i]));
		CUDA_CHECK(cudaFreeHost(h_decode_p_slot[i]));
	}

	free(h_encode_p_slot);
	free(h_encode_y_slot);
	free(h_decode_x_slot);
	free(h_decode_y_slot);
	free(h_decode_p_slot);

	CUDA_CHECK_LAST();

	CUDA_CHECK(cudaDeviceReset());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK_LAST();

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
	//
	//test_poly_mul();
	//
	//test_build_init_product();
	//
	//test_encode_decode();
	//
	//test_fnt_performance();

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
			c1[i] = rand() % (MOD - 1);
		shuffle(c1.begin(), c1.end(), std::default_random_engine(time(NULL)));
		cudaMemcpy(d_c1, c1.data(), size_nc, cudaMemcpyHostToDevice);

		fnt(d_c1, d_v, log_nc, log_nv, 0, d_N_pos, d_root_layer_pow, d_inv, NULL);
		fnt(d_v, d_c2, log_nv, log_nv, 3, d_N_pos, d_root_layer_pow, d_inv, NULL);

		cudaMemcpy(c2.data(), d_c2, size_nc, cudaMemcpyDeviceToHost);
		for (unsigned i = 0; i < nc; i++)
			assert(c1[i] == c2[i]);

		cudaFree(d_c1);
		cudaFree(d_c2);
		cudaFree(d_v);

		//std::cout << "FNT test " << tt << " passed!" << std::endl;

	}

	std::cout << "FNT test passed!" << std::endl;

	CUDA_CHECK_LAST();

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

	CUDA_CHECK_LAST();

}

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

		for (unsigned i = 0; i < n; i++)
			for (unsigned j = 0; j < n; j++)
				c1[i + j] = add_mod(c1[i + j], mul_mod(a[i], b[j]));

		/*unsigned* d_a, * d_b;
		cudaMalloc(&d_a, size_n);
		cudaMalloc(&d_b, size_n);
		cudaMemcpy(d_a, a.data(), size_n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b.data(), size_n, cudaMemcpyHostToDevice);
		poly_mul_wrapper CUDA_KERNEL(1, 1, NULL, NULL)(d_a, d_b, t1, t2, d_c, log_n, d_N_pos, d_root_layer_pow, d_inv);*/

		cudaMemcpy(c2.data(), d_c, size_n << 1, cudaMemcpyDeviceToHost);

		for (unsigned i = 0; i < (n << 1); i++)
			assert(c1[i] == c2[i]);

		//std::cout << "Poly mul test " << tt << " passed!" << std::endl;

		cudaFree(t1);
		cudaFree(t2);
		cudaFree(d_c);

	}

	std::cout << "Poly mul test passed!" << std::endl;

	CUDA_CHECK_LAST();

}

void test_encode_decode() {

	// test correctness of encode() and decode()

	srand(time(NULL));

	unsigned N_test = 32;

	for (unsigned tt = 0; tt < N_test; tt++) {
		std::vector<unsigned> a(NUM_OF_NEED_SYMBOL), b(NUM_OF_NEED_SYMBOL << 1), c(NUM_OF_NEED_SYMBOL);

		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			a[i] = rand() % (MOD - 1); // 2 bytes

		h_encode(a.data(), b.data());
		cudaDeviceSynchronize();

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

		h_decode(x.data(), y.data(), c.data());
		cudaDeviceSynchronize();

		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			assert(a[i] == c[i]);
		//std::cout << "Encode decode test " << tt << " passed!" << std::endl;
	}

	std::cout << "Encode decode test passed!" << std::endl;

	CUDA_CHECK_LAST();

}

void test_fnt_performance() {

	// fnt() performance with memory already prepare in device

	using namespace std;

	const unsigned N_test = 1024 * 1024 / 64;
	//const unsigned N_test = 1; // use when need profile one..
	unsigned log_n = 16, n = 1 << log_n;
	unsigned size_n = n * sizeof(unsigned);
	vector<vector<unsigned>> a(N_test, vector<unsigned>(n));
	cudaStream_t stream[N_test];
	vector<unsigned*> d_a(N_test), d_b(N_test);

	for (unsigned tt = 0; tt < N_test; tt++) {
		for (unsigned i = 0; i < n; i++)
			a[tt][i] = rand() % (MOD - 1);
		CUDA_CHECK(cudaStreamCreate(&stream[tt]));
		CUDA_CHECK(cudaMallocAsync(&d_a[tt], size_n, stream[tt]));
		CUDA_CHECK(cudaMallocAsync(&d_b[tt], size_n, stream[tt]));
		CUDA_CHECK(cudaMemcpyAsync(d_a[tt], a[tt].data(), size_n, cudaMemcpyHostToDevice, stream[tt]));
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	cout << "FNT test start" << endl;

	auto start = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++) {
		//cudaStreamCreate(&stream[tt]);
		//cudaMallocAsync(&d_a[tt], size_n, stream[tt]);
		//cudaMallocAsync(&d_b[tt], size_n, stream[tt]);
		//cudaMemcpyAsync(d_a[tt], a[tt].data(), size_n, cudaMemcpyHostToDevice, stream[tt]);
		fnt(d_a[tt], d_b[tt], log_n, log_n, 0, d_N_pos, d_root_layer_pow, d_inv, stream[tt]);
		CUDA_CHECK_LAST();
		//cudaFreeAsync(d_a[tt], stream[tt]);
		//cudaFreeAsync(d_b[tt], stream[tt]);
		//cudaStreamDestroy(stream[tt]);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();

	cout << "FNT " << N_test << " chunks in " << duration << "ms" << endl;

	for (unsigned tt = 0; tt < N_test; tt++) {
		CUDA_CHECK(cudaFreeAsync(d_a[tt], stream[tt]));
		CUDA_CHECK(cudaFreeAsync(d_b[tt], stream[tt]));
		CUDA_CHECK(cudaStreamDestroy(stream[tt]));
	}

}

void test_encode_decode_performance() {

	// test encode(), decode() performance full flow (without prepare memory in device)

	using namespace std;
	srand(time(NULL));

	const unsigned N_test = 128 * 1024 / 64;
	//const unsigned N_test = 1; // use when need profile one..
	const long long symbol_bytes = 2;
	const double size_test_gb = 1.0 * symbol_bytes * NUM_OF_NEED_SYMBOL * N_test / (1024 * 1024 * 1024);

	vector<vector<unsigned>> a(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));
	vector<vector<unsigned>> b(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL << 1));
	vector<vector<unsigned>> c(N_test, vector<unsigned>(NUM_OF_NEED_SYMBOL));

	for (unsigned tt = 0; tt < N_test; tt++)
		for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
			a[tt][i] = rand() % (MOD - 1); // 2 bytes

	CUDA_CHECK(cudaDeviceSynchronize());

	cout << "Encode performance test start" << endl;

	auto start1 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++) {
		h_encode(a[tt].data(), b[tt].data());
		CUDA_CHECK_LAST();
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto stop1 = chrono::high_resolution_clock::now();
	auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1).count();

	cout << "Encode " << N_test << " 64kb chunks in " << duration1 << "ms" << endl;
	cout << "Encode " << (1.0 * size_test_gb) / (1.0 * duration1 / 1000.0) << " GB/s" << endl;

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

	CUDA_CHECK(cudaDeviceSynchronize());

	cout << "Decode performance test start" << endl;

	auto start2 = chrono::high_resolution_clock::now();

	for (unsigned tt = 0; tt < N_test; tt++) {
		h_decode(x[tt].data(), y[tt].data(), c[tt].data());
		CUDA_CHECK_LAST();
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto stop2 = chrono::high_resolution_clock::now();
	auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();

	cout << "Decode " << N_test << " 64kb chunks in " << duration2 << "ms" << endl;
	cout << "Decode " << (1.0 * size_test_gb) / (1.0 * duration2 / 1000.0) << " GB/s" << endl;

	for (unsigned tt = 0; tt < N_test; tt++) {
		for (unsigned i = 0; i < c[tt].size(); i++)
			assert(a[tt][i] == c[tt][i]);
	}

}

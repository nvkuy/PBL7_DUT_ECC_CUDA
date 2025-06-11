#ifdef __CUDACC__
#define CUDA_KERNEL(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define CUDA_SYNCTHREADS() __syncthreads()
#else
#define CUDA_KERNEL(grid, block, sh_mem, stream)
#define CUDA_SYNCTHREADS()
#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b
#endif

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "rs_java_RS_Native.h"
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

#define JNI_CHECK_LAST(env) \
    if ((env)->ExceptionCheck()) { \
        std::cerr << "!! JNI EXCEPTION PENDING at " << __FILE__ << ":" << __LINE__ << std::endl; \
        (env)->ExceptionDescribe(); \
        return; \
    }

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK_LAST() check_last(__FILE__, __LINE__)
inline void check_last(const char *const file, const int line)
{
	cudaError_t const err{cudaPeekAtLastError()};
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

// const unsigned SM_CNT = 28;
// const unsigned MAX_WARP = 48 * 4 * SM_CNT;

// const unsigned MAX_ENCODE_LAUNCH_CNT = 32;
// const unsigned MAX_DECODE_LAUNCH_CNT = 256;

// const unsigned LOG_THREAD_PER_OP = 10;
// const unsigned THREAD_PER_OP = 1 << 10;
const unsigned MAX_NUM_BLOCK_PER_OP = 32;

const unsigned LOG_THREAD_PER_BLOCK = 8;
const unsigned THREAD_PER_BLOCK = 1 << LOG_THREAD_PER_BLOCK;
// const unsigned N_TH = 256, N_BL = THREAD_PER_OP / N_TH;

const unsigned LOG_THREAD_PER_EXTR_OP = 5;
const unsigned THREAD_PER_EXTR_OP = 1 << LOG_THREAD_PER_EXTR_OP;

const unsigned LOG_LEN_WARP = 5;
const unsigned LEN_WARP = 1 << LOG_LEN_WARP;
const unsigned ALGO_N_2_CUTOFF = 64;

int **h_encode_p_slot;
int **h_encode_y_slot;
int **h_decode_x_slot;
int **h_decode_y_slot;
int **h_decode_p_slot;

unsigned *d_encode_p_slot;
unsigned *d_encode_y_slot;
unsigned *d_decode_x_slot;
unsigned *d_decode_y_slot;
// unsigned* d_decode_p_slot;
unsigned *d_decode_t1_slot;
unsigned *d_decode_t2_slot;
unsigned *d_decode_ax_slot;
unsigned *d_decode_dax_slot;
unsigned *d_decode_vdax_slot;
unsigned *d_decode_n1_slot;
unsigned *d_decode_n2_slot;
unsigned *d_decode_n3_slot;

unsigned *d_N_pos;
unsigned *d_root_pow;
unsigned *d_root_inv_pow;
unsigned *d_inv;
unsigned *d_root_layer_pow;
unsigned *d_packet_product;

static JavaVM* g_jvm = nullptr;
jobject rs_obj;
jmethodID rs_after_encode, rs_after_decode;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
	g_jvm = vm;
	JNIEnv* env;
    vm->GetEnv((void**)&env, JNI_VERSION_24);

	jclass rsClass = env->FindClass("rs_java/RS_Native");

	rs_after_encode = env->GetMethodID(rsClass, "encodeProcessAfter", "(I[I)V");
	rs_after_decode = env->GetMethodID(rsClass, "decodeProcessAfter", "(I[I)V");

	env->DeleteLocalRef(rsClass);

	return JNI_VERSION_24;
}

struct CB_DATA
{
	jint slot_id;
};

void CUDART_CB h_end_slot_encode(void *data)
{

	CB_DATA *dat = static_cast<CB_DATA *>(data);

	JNIEnv* env = nullptr;
	g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_24);
	g_jvm->AttachCurrentThread((void**)&env, nullptr);

	jintArray res = env->NewIntArray(LEN_LARGE);
	JNI_CHECK_LAST(env);
	env->SetIntArrayRegion(res, 0, LEN_LARGE, h_encode_y_slot[dat->slot_id]);
	JNI_CHECK_LAST(env);
	
	env->CallVoidMethod(rs_obj, rs_after_encode, dat->slot_id, res);
	JNI_CHECK_LAST(env);

	env->DeleteLocalRef(res);
	JNI_CHECK_LAST(env);
	g_jvm->DetachCurrentThread();
	delete dat;
}

void CUDART_CB h_end_slot_decode(void *data)
{

	CB_DATA *dat = static_cast<CB_DATA *>(data);

	JNIEnv* env = nullptr;
	g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_24);
	g_jvm->AttachCurrentThread((void**)&env, nullptr);

	jintArray res = env->NewIntArray(LEN_SMALL);
	JNI_CHECK_LAST(env);
	env->SetIntArrayRegion(res, 0, LEN_SMALL, h_decode_p_slot[dat->slot_id]);
	JNI_CHECK_LAST(env);

	env->CallVoidMethod(rs_obj, rs_after_decode, dat->slot_id, res);
	JNI_CHECK_LAST(env);

	env->DeleteLocalRef(res);
	JNI_CHECK_LAST(env);
	g_jvm->DetachCurrentThread();
	delete dat;
}

__host__ __device__ __forceinline__ unsigned num_block(unsigned n)
{
	return min(max(n >> LOG_THREAD_PER_BLOCK, 1), MAX_NUM_BLOCK_PER_OP);
}

__host__ __device__ __forceinline__ unsigned add_small_mod(unsigned a, unsigned b)
{
	unsigned res = a + b;
    if (res >= MOD) res -= MOD;
    return res;
}

__host__ __device__ __forceinline__ unsigned sub_small_mod(unsigned a, unsigned b)
{
    unsigned res = a + MOD - b;
    if (res >= MOD) res -= MOD;
    return res;
}

__host__ __device__ __forceinline__ unsigned fast_mod(unsigned x) {

	// if (x < MOD) return x;
	// unsigned res = MOD + (x & 0xFFFFU) - (x >> 16);
	// if (res >= MOD) res -= MOD;
	// return res;
	return sub_small_mod((x & 0xFFFFU), (x >> 16));

}

__host__ __device__ __forceinline__ unsigned mul_mod(unsigned a, unsigned b)
{
	//if (a == SPECIAL && b == SPECIAL)
	//	return 1; // overflow
	//return (a * b) % MOD;
	return (a == SPECIAL && b == SPECIAL) ? 1 : fast_mod(a * b);
}

__device__ __forceinline__ unsigned div_mod(unsigned a, unsigned b, unsigned *d_inv)
{
	return mul_mod(a, d_inv[b]);
}

__host__ __device__ __forceinline__ unsigned add_mod(unsigned a, unsigned b)
{
	return fast_mod(a + b);
}

__host__ __device__ __forceinline__ unsigned sub_mod(unsigned a, unsigned b)
{
	return fast_mod(a - b + MOD);
}

__host__ __device__ __forceinline__ unsigned pow_mod(unsigned a, unsigned b)
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

__global__ void g_pre_fnt(unsigned *a, unsigned *b, unsigned st_d_N_pos, unsigned na, unsigned *d_N_pos)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < na; k += tpo)
		b[d_N_pos[st_d_N_pos + k]] = a[k];
}

__global__ void g_end_fnt(unsigned *b, unsigned nbd2, unsigned *d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	unsigned inv_nb = div_mod(1, nbd2 << 1, d_inv);

	for (unsigned k = id; k < nbd2; k += tpo)
	{
		b[k] = mul_mod(b[k], inv_nb);
		b[k + nbd2] = mul_mod(b[k + nbd2], inv_nb);
	}
}

__global__ void g_fnt_i(unsigned *b, unsigned i, bool inv, unsigned nbd2, unsigned *d_root_layer_pow)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	unsigned haft_len = 1 << i, wlen_os = LEN_ROOT_LAYER_POW * inv + haft_len - 1;
	for (unsigned k = id; k < nbd2; k += tpo)
	{
		unsigned bl_st = ((k >> i) << (i + 1)), th_id = (k & (haft_len - 1));
		unsigned pos = bl_st + th_id;
		unsigned u = b[pos];
		unsigned v = mul_mod(b[pos + haft_len], d_root_layer_pow[wlen_os + th_id]);
		b[pos] = add_small_mod(u, v);
		b[pos + haft_len] = sub_small_mod(u, v);
	}
}

__host__ __forceinline__ __device__ void fnt(unsigned *a, unsigned *b, unsigned log_na, unsigned log_nb, unsigned opt, unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv, cudaStream_t stream)
{

	/*
	opt 2 bit: x1 x2
	 - x1: w_n or 1/w_n
	 - x2: need result * 1/n
	*/

	// size_b >= size_a;
	// need memset *b before use unless size_a == size_b

	unsigned nb = 1 << log_nb, wp = (opt & 2) >> 1;
	unsigned na = 1 << log_na, nbd2 = nb >> 1;

	unsigned n_bl = num_block(na);
	g_pre_fnt CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(a, b, nb - 1, na, d_N_pos);

	n_bl = num_block(nbd2);
	for (unsigned i = 0; i < log_nb; i++)
		g_fnt_i CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(b, i, wp, nbd2, d_root_layer_pow);

	if (opt & 1)
		g_end_fnt CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(b, nbd2, d_inv);
}

__global__ void g_vector_mul_i(unsigned *a, unsigned *b, unsigned *c, unsigned n)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < n; k += tpo)
		c[k] = mul_mod(a[k], b[k]);
}

__global__ void g_fill(unsigned *a, unsigned val, unsigned na)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < na; k += tpo)
		a[k] = val;
}

// __global__ void g_cpy(unsigned *a, unsigned *b, unsigned n)
// {

// 	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
// 	unsigned tpo = gridDim.x * blockDim.x;
// 	for (unsigned k = id; k < n; k += tpo)
// 		b[k] = a[k];
// }

__forceinline__ __device__ void d_poly_mul(unsigned *a, unsigned *b, unsigned *t1, unsigned *t2, unsigned *c, unsigned log_n, unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv)
{

	// 2 ^ log_n == size_a && size_a == size_b
	// *c == *a && *a + na == *b (allow)

	unsigned na = 1 << log_n, nc = na << 1;

	if (na > ALGO_N_2_CUTOFF)
	{

		unsigned n_bl = num_block(nc);
		g_fill CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, cudaStreamPerThread)(t1, 0, nc);
		g_fill CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, cudaStreamPerThread)(t2, 0, nc);

		fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, cudaStreamPerThread);
		fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, cudaStreamPerThread);

		g_vector_mul_i CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, cudaStreamPerThread)(t1, t2, t1, nc);

		fnt(t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, cudaStreamPerThread);

	}
	else
	{

		for (unsigned i = 0; i < na; i++)
		{
			t1[i] = a[i];
			t2[i] = b[i];
		}
		unsigned size_nc = nc * sizeof(unsigned);
		memset(c, 0, size_nc);
		for (unsigned i = 0; i < na; i++)
			for (unsigned j = 0; j < na; j++)
				c[i + j] = add_mod(c[i + j], mul_mod(t1[i], t2[j]));
	}
}

inline void h_poly_mul(unsigned *a, unsigned *b, unsigned *t1, unsigned *t2, unsigned *c, unsigned log_n, unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv, cudaStream_t stream)
{

	unsigned nc = 1 << (log_n + 1), size_nc = nc * sizeof(unsigned);
	unsigned n_bl = num_block(nc);

	// g_fill CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(t1, 0, nc);
	// CUDA_CHECK_LAST();
	// g_fill CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(t2, 0, nc);
	// CUDA_CHECK_LAST();
	CUDA_CHECK(cudaMemsetAsync(t1, 0, size_nc, stream));
	CUDA_CHECK(cudaMemsetAsync(t2, 0, size_nc, stream));

	fnt(a, t1, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
	fnt(b, t2, log_n, log_n + 1, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();

	g_vector_mul_i CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(t1, t2, t1, nc);
	CUDA_CHECK_LAST();

	fnt(t1, c, log_n + 1, log_n + 1, 3, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
}

__global__ void g_poly_deriv(unsigned *ax, unsigned *dax)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < NUM_OF_NEED_SYMBOL; k += tpo)
		dax[k] = mul_mod(ax[k + 1], k + 1);
}

__global__ void g_build_product_i(unsigned *p, unsigned *t1, unsigned *t2, unsigned i, unsigned n_g, unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	unsigned len = 1 << i;

	for (unsigned k = id; k < n_g; k += tpo)
	{
		unsigned st = k << (i + 1);
		d_poly_mul(p + st, p + st + len, t1 + st, t2 + st, p + st, i, d_N_pos, d_root_layer_pow, d_inv);
	}
}

inline void h_build_product(unsigned *p, unsigned *t1, unsigned *t2, unsigned log_n1, unsigned log_n2, cudaStream_t stream)
{

	// p, t1, t2 in device

	for (unsigned i = log_n1; i < log_n2; i++)
	{
		unsigned n_g = 1 << (log_n2 - i - 1);
		g_build_product_i CUDA_KERNEL(1, THREAD_PER_EXTR_OP, 0, stream)(p, t1, t2, i, n_g, d_N_pos, d_root_layer_pow, d_inv);
		CUDA_CHECK_LAST();
	}
}

// __global__ void g_pre_build_ax(unsigned *ax, unsigned *x, unsigned *d_packet_product)
// {

// 	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
// 	unsigned st_p1 = id << (LOG_SYMBOL + 1), st_p2 = x[id << LOG_SYMBOL] << 2;
// 	memcpy(ax + st_p1, d_packet_product + st_p2, SIZE_ONE_PACKET_PRODUCT);
// }

inline void h_build_ax(int *x, unsigned *ax, unsigned *t1, unsigned *t2, cudaStream_t stream)
{

	for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		unsigned st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 2;
		CUDA_CHECK(cudaMemcpyAsync(ax + st_p1, d_packet_product + st_p2, SIZE_ONE_PACKET_PRODUCT, cudaMemcpyDeviceToDevice, stream));
		// unsigned n_bl = (LEN_ONE_PACKET_PRODUCT + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
		// g_cpy CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(d_packet_product + st_p2, ax + st_p1, LEN_ONE_PACKET_PRODUCT);
		// CUDA_CHECK_LAST();
	}
	// g_pre_build_ax CUDA_KERNEL(1, NUM_OF_NEED_PACKET, 0, stream) (ax, x, d_packet_product);
	// CUDA_CHECK_LAST();

	h_build_product(ax, t1, t2, LOG_SYMBOL + 1, MAX_LOG, stream);
}

__global__ void g_build_n1(unsigned *n1, unsigned *vdax, unsigned *x, unsigned *y, unsigned *d_inv)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < NUM_OF_NEED_SYMBOL; k += tpo)
		n1[k] = div_mod(y[k], vdax[x[k]], d_inv);
}

__global__ void g_build_n2(unsigned *n2, unsigned *n1, unsigned *x)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < NUM_OF_NEED_SYMBOL; k += tpo)
		n2[x[k]] = n1[k];
}

__global__ void g_build_n3(unsigned *n3, unsigned *p_n3)
{

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tpo = gridDim.x * blockDim.x;
	for (unsigned k = id; k < NUM_OF_NEED_SYMBOL; k += tpo)
		n3[k] = sub_mod(0, p_n3[k + 1]);
}

inline void h_build_px(unsigned *p, unsigned *ax, unsigned *n3, unsigned *t1, unsigned *t2, unsigned *d_N_pos, unsigned *d_root_layer_pow, unsigned *d_inv, cudaStream_t stream)
{

	h_poly_mul(ax, n3, t1, t2, p, MAX_LOG - 1, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
}

inline void h_encode(CB_DATA *data, int *p)
{

	unsigned slot_id = data->slot_id;

	int *sl_p = h_encode_p_slot[slot_id];
	int *sl_y = h_encode_y_slot[slot_id];

	unsigned *d_p = d_encode_p_slot + 1LL * slot_id * LEN_ENCODE_P;
	unsigned *d_y = d_encode_y_slot + 1LL * slot_id * LEN_ENCODE_Y;

	unsigned n_bl = num_block(LEN_LARGE);

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	g_fill CUDA_KERNEL(n_bl, THREAD_PER_BLOCK, 0, stream)(d_y, 0, LEN_LARGE);
	CUDA_CHECK_LAST();
	// CUDA_CHECK(cudaMemsetAsync(d_y, 0, SIZE_LARGE, stream));

	memcpy(sl_p, p, SIZE_ENCODE_P);
	CUDA_CHECK(cudaMemcpyAsync(d_p, sl_p, SIZE_ENCODE_P, cudaMemcpyHostToDevice, stream));

	fnt(d_p, d_y, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();

	CUDA_CHECK(cudaMemcpyAsync(sl_y, d_y, SIZE_ENCODE_Y, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaLaunchHostFunc(stream, h_end_slot_encode, data));

	CUDA_CHECK(cudaStreamDestroy(stream));
}

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_encode(
	JNIEnv *env, jobject j_obj, jint slot_id, jintArray j_p) 
{

	jint *p = env->GetIntArrayElements(j_p, nullptr);
	JNI_CHECK_LAST(env);

	CB_DATA *data = new CB_DATA{slot_id};
	h_encode(data, p);

	env->ReleaseIntArrayElements(j_p, p, JNI_ABORT);
	JNI_CHECK_LAST(env);

}

inline void h_decode(CB_DATA *data, int *x, int *y)
{

	unsigned slot_id = data->slot_id;

	int *sl_x = h_decode_x_slot[slot_id];
	int *sl_y = h_decode_y_slot[slot_id];
	int *sl_p = h_decode_p_slot[slot_id];

	unsigned *d_x = d_decode_x_slot + 1LL * slot_id * LEN_DECODE_X;
	unsigned *d_y = d_decode_y_slot + 1LL * slot_id * LEN_DECODE_Y;
	unsigned *d_t1 = d_decode_t1_slot + 1LL * slot_id * LEN_LARGE;
	unsigned *d_t2 = d_decode_t2_slot + 1LL * slot_id * LEN_LARGE;
	unsigned *d_ax = d_decode_ax_slot + 1LL * slot_id * LEN_LARGE;
	unsigned *d_dax = d_decode_dax_slot + 1LL * slot_id * LEN_SMALL;
	unsigned *d_vdax = d_decode_vdax_slot + 1LL * slot_id * LEN_LARGE;
	unsigned *d_n1 = d_decode_n1_slot + 1LL * slot_id * LEN_SMALL;
	unsigned *d_n2 = d_decode_n2_slot + 1LL * slot_id * LEN_LARGE;
	unsigned *d_n3 = d_decode_n3_slot + 1LL * slot_id * LEN_SMALL;

	unsigned n_bl_large = num_block(LEN_LARGE);
	unsigned n_bl_small = num_block(LEN_SMALL);

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	g_fill CUDA_KERNEL(n_bl_large, THREAD_PER_BLOCK, 0, stream)(d_vdax, 0, LEN_LARGE);
	CUDA_CHECK_LAST();
	g_fill CUDA_KERNEL(n_bl_large, THREAD_PER_BLOCK, 0, stream)(d_n2, 0, LEN_LARGE);
	CUDA_CHECK_LAST();
	// CUDA_CHECK(cudaMemsetAsync(d_n2, 0, SIZE_LARGE, stream));
	// CUDA_CHECK(cudaMemsetAsync(d_vdax, 0, SIZE_LARGE, stream));

	memcpy(sl_x, x, SIZE_DECODE_X);
	CUDA_CHECK(cudaMemcpyAsync(d_x, sl_x, SIZE_DECODE_X, cudaMemcpyHostToDevice, stream));

	memcpy(sl_y, y, SIZE_DECODE_Y);
	CUDA_CHECK(cudaMemcpyAsync(d_y, sl_y, SIZE_DECODE_Y, cudaMemcpyHostToDevice, stream));

	h_build_ax(x, d_ax, d_t1, d_t2, stream);

	g_poly_deriv CUDA_KERNEL(n_bl_small, THREAD_PER_BLOCK, 0, stream)(d_ax, d_dax);
	CUDA_CHECK_LAST();

	fnt(d_dax, d_vdax, MAX_LOG - 1, MAX_LOG, 0, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();

	g_build_n1 CUDA_KERNEL(n_bl_small, THREAD_PER_BLOCK, 0, stream)(d_n1, d_vdax, d_x, d_y, d_inv);
	CUDA_CHECK_LAST();

	g_build_n2 CUDA_KERNEL(n_bl_small, THREAD_PER_BLOCK, 0, stream)(d_n2, d_n1, d_x);
	CUDA_CHECK_LAST();

	fnt(d_n2, d_t2, MAX_LOG, MAX_LOG, 2, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK_LAST();
	g_build_n3 CUDA_KERNEL(n_bl_small, THREAD_PER_BLOCK, 0, stream)(d_n3, d_t2);
	CUDA_CHECK_LAST();

	h_build_px(d_n2, d_ax, d_n3, d_t1, d_t2, d_N_pos, d_root_layer_pow, d_inv, stream);
	CUDA_CHECK(cudaMemcpyAsync(sl_p, d_n2, SIZE_SMALL, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaLaunchHostFunc(stream, h_end_slot_decode, data));

	CUDA_CHECK(cudaStreamDestroy(stream));
}

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_decode(
	JNIEnv *env, jobject j_obj, jint slot_id, jintArray j_x, jintArray j_y)
{

	jint *x = env->GetIntArrayElements(j_x, nullptr);
	JNI_CHECK_LAST(env);
	jint *y = env->GetIntArrayElements(j_y, nullptr);
	JNI_CHECK_LAST(env);

	CB_DATA *data = new CB_DATA{slot_id};
	h_decode(data, x, y);

	env->ReleaseIntArrayElements(j_x, x, JNI_ABORT);
	JNI_CHECK_LAST(env);
	env->ReleaseIntArrayElements(j_y, y, JNI_ABORT);
	JNI_CHECK_LAST(env);

}

void init(unsigned max_active_encode, unsigned max_active_decode)
{
	// offline process

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaDeviceReset());
	CUDA_CHECK(cudaDeviceSynchronize());
	// CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, max(max_active_decode * MAX_DECODE_LAUNCH_CNT, max_active_encode * MAX_ENCODE_LAUNCH_CNT)));

	// cudaDeviceProp prop;
	// int device;
	// cudaGetDevice(&device);
	// cudaGetDeviceProperties(&prop, device);
	// std::cout << "Device name: " << prop.name << std::endl;
	// std::cout << "Allow concurrent Kernels: " << prop.concurrentKernels << std::endl;
	// std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	// std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
	// std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
	// std::cout << "Max concurrent threads on device: " << prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount << std::endl;

	size_t size_encode_p_slot = sizeof(unsigned) * LEN_ENCODE_P * max_active_encode;
	size_t size_encode_y_slot = sizeof(unsigned) * LEN_ENCODE_Y * max_active_encode;
	size_t size_decode_x_slot = sizeof(unsigned) * LEN_DECODE_X * max_active_decode;
	size_t size_decode_y_slot = sizeof(unsigned) * LEN_DECODE_Y * max_active_decode;
	size_t size_decode_t1_slot = sizeof(unsigned) * LEN_LARGE * max_active_decode;
	size_t size_decode_t2_slot = sizeof(unsigned) * LEN_LARGE * max_active_decode;
	size_t size_decode_ax_slot = sizeof(unsigned) * LEN_LARGE * max_active_decode;
	size_t size_decode_dax_slot = sizeof(unsigned) * LEN_SMALL * max_active_decode;
	size_t size_decode_vdax_slot = sizeof(unsigned) * LEN_LARGE * max_active_decode;
	size_t size_decode_n1_slot = sizeof(unsigned) * LEN_SMALL * max_active_decode;
	size_t size_decode_n2_slot = sizeof(unsigned) * LEN_LARGE * max_active_decode;
	size_t size_decode_n3_slot = sizeof(unsigned) * LEN_SMALL * max_active_decode;

	h_encode_p_slot = (int **)malloc(max_active_encode * sizeof(unsigned *));
	h_encode_y_slot = (int **)malloc(max_active_encode * sizeof(unsigned *));
	h_decode_x_slot = (int **)malloc(max_active_decode * sizeof(unsigned *));
	h_decode_y_slot = (int **)malloc(max_active_decode * sizeof(unsigned *));
	h_decode_p_slot = (int **)malloc(max_active_decode * sizeof(unsigned *));

	for (unsigned i = 0; i < max_active_encode; i++)
	{
		CUDA_CHECK(cudaMallocHost(&(h_encode_p_slot[i]), SIZE_ENCODE_P));
		CUDA_CHECK(cudaMallocHost(&(h_encode_y_slot[i]), SIZE_ENCODE_Y));
	}

	for (unsigned i = 0; i < max_active_decode; i++)
	{
		CUDA_CHECK(cudaMallocHost(&(h_decode_x_slot[i]), SIZE_DECODE_X));
		CUDA_CHECK(cudaMallocHost(&(h_decode_y_slot[i]), SIZE_DECODE_Y));
		CUDA_CHECK(cudaMallocHost(&(h_decode_p_slot[i]), SIZE_DECODE_P));
	}

	CUDA_CHECK(cudaMalloc(&d_encode_p_slot, size_encode_p_slot));
	CUDA_CHECK(cudaMalloc(&d_encode_y_slot, size_encode_y_slot));

	CUDA_CHECK(cudaMalloc(&d_decode_x_slot, size_decode_x_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_y_slot, size_decode_y_slot));
	// CUDA_CHECK(cudaMalloc(&d_decode_p_slot, SIZE_DECODE_P_SLOT));
	CUDA_CHECK(cudaMalloc(&d_decode_t1_slot, size_decode_t1_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_t2_slot, size_decode_t2_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_ax_slot, size_decode_ax_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_dax_slot, size_decode_dax_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_vdax_slot, size_decode_vdax_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_n1_slot, size_decode_n1_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_n2_slot, size_decode_n2_slot));
	CUDA_CHECK(cudaMalloc(&d_decode_n3_slot, size_decode_n3_slot));

	unsigned size_N_pos = LEN_N_POS * sizeof(unsigned);
	unsigned *N_pos = (unsigned *)malloc(size_N_pos);
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
	unsigned *root_pow = (unsigned *)malloc(size_root);
	unsigned *root_inv_pow = (unsigned *)malloc(size_root);
	unsigned *inv = (unsigned *)malloc(size_root);
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
	unsigned *root_layer_pow = (unsigned *)malloc(size_root_layer_pow);
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
	unsigned *packet_product = (unsigned *)malloc(size_packet_product);
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
	unsigned *tmp;
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

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_init(
	JNIEnv *env, jobject j_obj, jint max_active_encode, jint max_active_decode) 
{
	rs_obj = env->NewGlobalRef(j_obj);
	JNI_CHECK_LAST(env);

	init(max_active_encode, max_active_decode);
}

void fin(unsigned max_active_encode, unsigned max_active_decode)
{
	// clear cuda memory

	CUDA_CHECK(cudaDeviceSynchronize());

	for (unsigned i = 0; i < max_active_encode; i++)
	{
		CUDA_CHECK(cudaFreeHost(h_encode_p_slot[i]));
		CUDA_CHECK(cudaFreeHost(h_encode_y_slot[i]));
	}

	for (unsigned i = 0; i < max_active_decode; i++)
	{
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

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_fin(
	JNIEnv *env, jobject j_obj, jint max_active_encode, jint max_active_decode)
{
	fin(max_active_encode, max_active_decode);
	
	env->DeleteGlobalRef(rs_obj);
	JNI_CHECK_LAST(env);
}

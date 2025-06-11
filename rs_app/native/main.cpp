#include <iostream>
#include <tbb/task_group.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <chrono>

#include "rs_java_RS_Native.h"

#include "kernel.h"

#define JNI_CHECK_LAST(env) \
    if ((env)->ExceptionCheck()) { \
        std::cerr << "!! JNI EXCEPTION PENDING at " << __FILE__ << ":" << __LINE__ << std::endl; \
        (env)->ExceptionDescribe(); \
        return; \
    }

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

jobject rs_obj;
jmethodID rs_after_encode, rs_after_decode;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
	JNIEnv* env;
    vm->GetEnv((void**)&env, JNI_VERSION_24);

	jclass rsClass = env->FindClass("rs_java/RS_Native");

	rs_after_encode = env->GetMethodID(rsClass, "encodeProcessAfter", "(I[I)V");
	rs_after_decode = env->GetMethodID(rsClass, "decodeProcessAfter", "(I[I)V");

	env->DeleteLocalRef(rsClass);

	return JNI_VERSION_24;
}

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

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_init(
	JNIEnv *env, jobject j_obj, jint max_active_encode, jint max_active_decode) 
{
	rs_obj = env->NewGlobalRef(j_obj);
	JNI_CHECK_LAST(env);

	init();
}

void fin()
{
	// clear cuda memory

	free(n_pos);
	free(inv);
	free(root_layer_pow);
	free(packet_product);
	
}

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_fin(
	JNIEnv *env, jobject j_obj, jint max_active_encode, jint max_active_decode)
{
	fin();
	
	env->DeleteGlobalRef(rs_obj);
	JNI_CHECK_LAST(env);
}

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_encode(
	JNIEnv *env, jobject j_obj, jint slot_id, jintArray j_p) 
{

	jint *p = env->GetIntArrayElements(j_p, nullptr);
	JNI_CHECK_LAST(env);

	jintArray res = env->NewIntArray(LEN_LARGE);
	JNI_CHECK_LAST(env);
	jint *y = env->GetIntArrayElements(res, nullptr);
	JNI_CHECK_LAST(env);

	ispc::encode(reinterpret_cast<uint32_t*>(p), reinterpret_cast<uint32_t*>(y), n_pos, root_layer_pow, inv);
	
	env->ReleaseIntArrayElements(res, y, 0);
	JNI_CHECK_LAST(env);

	env->CallVoidMethod(rs_obj, rs_after_encode, slot_id, res);
	JNI_CHECK_LAST(env);

	env->DeleteLocalRef(res);
	JNI_CHECK_LAST(env);

	env->ReleaseIntArrayElements(j_p, p, JNI_ABORT);
	JNI_CHECK_LAST(env);

}

JNIEXPORT void JNICALL Java_rs_1java_RS_1Native_decode(
	JNIEnv *env, jobject j_obj, jint slot_id, jintArray j_x, jintArray j_y)
{

	jint *x = env->GetIntArrayElements(j_x, nullptr);
	JNI_CHECK_LAST(env);
	jint *y = env->GetIntArrayElements(j_y, nullptr);
	JNI_CHECK_LAST(env);

	jintArray res = env->NewIntArray(LEN_SMALL);
	JNI_CHECK_LAST(env);
	jint *p = env->GetIntArrayElements(res, nullptr);
	JNI_CHECK_LAST(env);

	std::vector<unsigned> t1(LEN_LARGE), t2(LEN_LARGE), ax(LEN_LARGE), dax(LEN_SMALL);
	std::vector<unsigned> vdax(LEN_LARGE), n1(LEN_SMALL), n2(LEN_LARGE), n3(LEN_SMALL);
	ispc::decode(reinterpret_cast<uint32_t*>(x), reinterpret_cast<uint32_t*>(y), reinterpret_cast<uint32_t*>(p), 
		t1.data(), t2.data(), ax.data(), dax.data(), vdax.data(), 
		n1.data(), n2.data(), n3.data(),
		packet_product, n_pos, root_layer_pow, inv);

	env->ReleaseIntArrayElements(res, p, 0);
	JNI_CHECK_LAST(env);

	env->CallVoidMethod(rs_obj, rs_after_decode, slot_id, res);
	JNI_CHECK_LAST(env);

	env->DeleteLocalRef(res);
	JNI_CHECK_LAST(env);

	env->ReleaseIntArrayElements(j_x, x, JNI_ABORT);
	JNI_CHECK_LAST(env);
	env->ReleaseIntArrayElements(j_y, y, JNI_ABORT);
	JNI_CHECK_LAST(env);

}
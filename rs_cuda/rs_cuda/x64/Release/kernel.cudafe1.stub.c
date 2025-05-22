#define __NV_MODULE_ID _dcb4e26f_9_kernel_cu_f9c6e15d
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel.fatbin.c"
extern void __device_stub__Z3fntPjS_jjjS_S_S_j(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *, unsigned);
extern void __device_stub__Z14g_vector_mul_iPjS_S_jj(unsigned *, unsigned *, unsigned *, unsigned, unsigned);
extern void __device_stub__Z6g_fillPjjjj(unsigned *, unsigned, unsigned, unsigned);
extern void __device_stub__Z12g_poly_derivPjS_j(unsigned *, unsigned *, unsigned);
extern void __device_stub__Z10g_build_n1PjS_S_S_S_j(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *, unsigned);
extern void __device_stub__Z10g_build_n2PjS_S_j(unsigned *, unsigned *, unsigned *, unsigned);
extern void __device_stub__Z10g_build_n3PjS_j(unsigned *, unsigned *, unsigned);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCT",read)
__declspec(allocate(".CRT$XCT"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
void __device_stub__Z3fntPjS_jjjS_S_S_j(
unsigned *__par0, 
unsigned *__par1, 
unsigned __par2, 
unsigned __par3, 
unsigned __par4, 
unsigned *__par5, 
unsigned *__par6, 
unsigned *__par7, 
unsigned __par8)
{
__cudaLaunchPrologue(9);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 20Ui64);
__cudaSetupArgSimple(__par4, 24Ui64);
__cudaSetupArgSimple(__par5, 32Ui64);
__cudaSetupArgSimple(__par6, 40Ui64);
__cudaSetupArgSimple(__par7, 48Ui64);
__cudaSetupArgSimple(__par8, 56Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *, unsigned))fnt)));
}
void fnt( unsigned *__cuda_0,unsigned *__cuda_1,unsigned __cuda_2,unsigned __cuda_3,unsigned __cuda_4,unsigned *__cuda_5,unsigned *__cuda_6,unsigned *__cuda_7,unsigned __cuda_8)
{__device_stub__Z3fntPjS_jjjS_S_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z14g_vector_mul_iPjS_S_jj(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2, 
unsigned __par3, 
unsigned __par4)
{
__cudaLaunchPrologue(5);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 28Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *, unsigned, unsigned))g_vector_mul_i)));
}
void g_vector_mul_i( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2,unsigned __cuda_3,unsigned __cuda_4)
{__device_stub__Z14g_vector_mul_iPjS_S_jj( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z6g_fillPjjjj(
unsigned *__par0, 
unsigned __par1, 
unsigned __par2, 
unsigned __par3)
{
__cudaLaunchPrologue(4);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 12Ui64);
__cudaSetupArgSimple(__par3, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned, unsigned, unsigned))g_fill)));
}
void g_fill( unsigned *__cuda_0,unsigned __cuda_1,unsigned __cuda_2,unsigned __cuda_3)
{__device_stub__Z6g_fillPjjjj( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z12g_poly_derivPjS_j(
unsigned *__par0, 
unsigned *__par1, 
unsigned __par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned))g_poly_deriv)));
}
void g_poly_deriv( unsigned *__cuda_0,unsigned *__cuda_1,unsigned __cuda_2)
{__device_stub__Z12g_poly_derivPjS_j( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z10g_build_n1PjS_S_S_S_j(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2, 
unsigned *__par3, 
unsigned *__par4, 
unsigned __par5)
{
__cudaLaunchPrologue(6);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaSetupArgSimple(__par5, 40Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *, unsigned))g_build_n1)));
}
void g_build_n1( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2,unsigned *__cuda_3,unsigned *__cuda_4,unsigned __cuda_5)
{__device_stub__Z10g_build_n1PjS_S_S_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z10g_build_n2PjS_S_j(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2, 
unsigned __par3)
{
__cudaLaunchPrologue(4);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *, unsigned))g_build_n2)));
}
void g_build_n2( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2,unsigned __cuda_3)
{__device_stub__Z10g_build_n2PjS_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z10g_build_n3PjS_j(
unsigned *__par0, 
unsigned *__par1, 
unsigned __par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned))g_build_n3)));
}
void g_build_n3( unsigned *__cuda_0,unsigned *__cuda_1,unsigned __cuda_2)
{__device_stub__Z10g_build_n3PjS_j( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback(
void **__T5)
{
__nv_dummy_param_ref(__T5);
__nv_save_fatbinhandle_for_managed_rt(__T5);
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned))g_build_n3), _Z10g_build_n3PjS_j, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned *, unsigned))g_build_n2), _Z10g_build_n2PjS_S_j, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *, unsigned))g_build_n1), _Z10g_build_n1PjS_S_S_S_j, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned))g_poly_deriv), _Z12g_poly_derivPjS_j, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned, unsigned, unsigned))g_fill), _Z6g_fillPjjjj, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned *, unsigned, unsigned))g_vector_mul_i), _Z14g_vector_mul_iPjS_S_jj, (-1));
__cudaRegisterEntry(__T5, ((void ( *)(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *, unsigned))fnt), _Z3fntPjS_jjjS_S_S_j, (-1));
}
static void __sti____cudaRegisterAll(void)
{
____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);
}

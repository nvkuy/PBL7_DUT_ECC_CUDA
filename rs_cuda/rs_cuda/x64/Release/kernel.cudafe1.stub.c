#define __NV_MODULE_ID _dcb4e26f_9_kernel_cu_6c617819
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel.fatbin.c"
extern void __device_stub__Z7pre_fntPjS_jS_(unsigned *, unsigned *, unsigned, unsigned *);
extern void __device_stub__Z7end_fntPjjS_(unsigned *, unsigned, unsigned *);
extern void __device_stub__Z5fnt_iPjjbS_(unsigned *, unsigned, bool, unsigned *);
extern void __device_stub__Z5g_fntPjS_jjjS_S_S_(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *);
extern void __device_stub__Z14g_vector_mul_iPjS_S_(unsigned *, unsigned *, unsigned *);
extern void __device_stub__Z10poly_derivPjS_(unsigned *, unsigned *);
extern void __device_stub__Z15build_product_iPjS_S_jS_S_S_(unsigned *, unsigned *, unsigned *, unsigned, unsigned *, unsigned *, unsigned *);
extern void __device_stub__Z8build_n1PjS_S_S_S_(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *);
extern void __device_stub__Z8build_n2PjS_S_(unsigned *, unsigned *, unsigned *);
extern void __device_stub__Z8build_n3PjS_(unsigned *, unsigned *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCT",read)
__declspec(allocate(".CRT$XCT"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
void __device_stub__Z7pre_fntPjS_jS_(
unsigned *__par0, 
unsigned *__par1, 
unsigned __par2, 
unsigned *__par3)
{
__cudaLaunchPrologue(4);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned, unsigned *))pre_fnt)));
}
void pre_fnt( unsigned *__cuda_0,unsigned *__cuda_1,unsigned __cuda_2,unsigned *__cuda_3)
{__device_stub__Z7pre_fntPjS_jS_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z7end_fntPjjS_(
unsigned *__par0, 
unsigned __par1, 
unsigned *__par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned, unsigned *))end_fnt)));
}
void end_fnt( unsigned *__cuda_0,unsigned __cuda_1,unsigned *__cuda_2)
{__device_stub__Z7end_fntPjjS_( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z5fnt_iPjjbS_(
unsigned *__par0, 
unsigned __par1, 
bool __par2, 
unsigned *__par3)
{
__cudaLaunchPrologue(4);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 12Ui64);
__cudaSetupArgSimple(__par3, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned, bool, unsigned *))fnt_i)));
}
void fnt_i( unsigned *__cuda_0,unsigned __cuda_1,bool __cuda_2,unsigned *__cuda_3)
{__device_stub__Z5fnt_iPjjbS_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z5g_fntPjS_jjjS_S_S_(
unsigned *__par0, 
unsigned *__par1, 
unsigned __par2, 
unsigned __par3, 
unsigned __par4, 
unsigned *__par5, 
unsigned *__par6, 
unsigned *__par7)
{
__cudaLaunchPrologue(8);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 20Ui64);
__cudaSetupArgSimple(__par4, 24Ui64);
__cudaSetupArgSimple(__par5, 32Ui64);
__cudaSetupArgSimple(__par6, 40Ui64);
__cudaSetupArgSimple(__par7, 48Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *))g_fnt)));
}
void g_fnt( unsigned *__cuda_0,unsigned *__cuda_1,unsigned __cuda_2,unsigned __cuda_3,unsigned __cuda_4,unsigned *__cuda_5,unsigned *__cuda_6,unsigned *__cuda_7)
{__device_stub__Z5g_fntPjS_jjjS_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z14g_vector_mul_iPjS_S_(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *))g_vector_mul_i)));
}
void g_vector_mul_i( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2)
{__device_stub__Z14g_vector_mul_iPjS_S_( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z10poly_derivPjS_(
unsigned *__par0, 
unsigned *__par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *))poly_deriv)));
}
void poly_deriv( unsigned *__cuda_0,unsigned *__cuda_1)
{__device_stub__Z10poly_derivPjS_( __cuda_0,__cuda_1);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z15build_product_iPjS_S_jS_S_S_(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2, 
unsigned __par3, 
unsigned *__par4, 
unsigned *__par5, 
unsigned *__par6)
{
__cudaLaunchPrologue(7);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaSetupArgSimple(__par5, 40Ui64);
__cudaSetupArgSimple(__par6, 48Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *, unsigned, unsigned *, unsigned *, unsigned *))build_product_i)));
}
void build_product_i( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2,unsigned __cuda_3,unsigned *__cuda_4,unsigned *__cuda_5,unsigned *__cuda_6)
{__device_stub__Z15build_product_iPjS_S_jS_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z8build_n1PjS_S_S_S_(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2, 
unsigned *__par3, 
unsigned *__par4)
{
__cudaLaunchPrologue(5);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *))build_n1)));
}
void build_n1( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2,unsigned *__cuda_3,unsigned *__cuda_4)
{__device_stub__Z8build_n1PjS_S_S_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z8build_n2PjS_S_(
unsigned *__par0, 
unsigned *__par1, 
unsigned *__par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *, unsigned *))build_n2)));
}
void build_n2( unsigned *__cuda_0,unsigned *__cuda_1,unsigned *__cuda_2)
{__device_stub__Z8build_n2PjS_S_( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
void __device_stub__Z8build_n3PjS_(
unsigned *__par0, 
unsigned *__par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned *))build_n3)));
}
void build_n3( unsigned *__cuda_0,unsigned *__cuda_1)
{__device_stub__Z8build_n3PjS_( __cuda_0,__cuda_1);
}
#line 1 "x64/Release/kernel.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback(
void **__T45)
{
__nv_dummy_param_ref(__T45);
__nv_save_fatbinhandle_for_managed_rt(__T45);
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *))build_n3), _Z8build_n3PjS_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned *))build_n2), _Z8build_n2PjS_S_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned *, unsigned *, unsigned *))build_n1), _Z8build_n1PjS_S_S_S_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned *, unsigned, unsigned *, unsigned *, unsigned *))build_product_i), _Z15build_product_iPjS_S_jS_S_S_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *))poly_deriv), _Z10poly_derivPjS_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned *))g_vector_mul_i), _Z14g_vector_mul_iPjS_S_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *))g_fnt), _Z5g_fntPjS_jjjS_S_S_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned, bool, unsigned *))fnt_i), _Z5fnt_iPjjbS_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned, unsigned *))end_fnt), _Z7end_fntPjjS_, (-1));
__cudaRegisterEntry(__T45, ((void ( *)(unsigned *, unsigned *, unsigned, unsigned *))pre_fnt), _Z7pre_fntPjS_jS_, (-1));
}
static void __sti____cudaRegisterAll(void)
{
____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);
}

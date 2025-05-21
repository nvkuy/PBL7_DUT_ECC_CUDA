#line 1 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
extern  /* COMDAT group: _ZTISt20bad_array_new_length */ const struct __si_class_type_info _ZTISt20bad_array_new_length;
extern  /* COMDAT group: _ZTISt12system_error */ const struct __si_class_type_info _ZTISt12system_error;
extern  /* COMDAT group: _ZTISt8bad_cast */ const struct __si_class_type_info _ZTISt8bad_cast;
extern  /* COMDAT group: _ZTINSt8ios_base7failureE */ const struct __si_class_type_info _ZTINSt8ios_base7failureE;
#line 129 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
unsigned **h_encode_p_slot = 0;
unsigned **h_encode_y_slot = 0;
unsigned **h_decode_x_slot = 0;
unsigned **h_decode_y_slot = 0;
unsigned **h_decode_p_slot = 0;

unsigned *d_encode_p_slot = 0;
unsigned *d_encode_y_slot = 0;
unsigned *d_decode_x_slot = 0;
unsigned *d_decode_y_slot = 0;

unsigned *d_decode_t1_slot = 0;
unsigned *d_decode_t2_slot = 0;
unsigned *d_decode_ax_slot = 0;
unsigned *d_decode_dax_slot = 0;
unsigned *d_decode_vdax_slot = 0;
unsigned *d_decode_n1_slot = 0;
unsigned *d_decode_n2_slot = 0;
unsigned *d_decode_n3_slot = 0;

unsigned *d_N_pos = 0;
unsigned *d_root_pow = 0;
unsigned *d_root_inv_pow = 0;
unsigned *d_inv = 0;
unsigned *d_root_layer_pow = 0;
unsigned *d_packet_product = 0;
#line 166 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
struct _ZSt5queueIjSt5dequeIjSaIjEEE encode_slot = {{{{{0}}}}};
#line 166 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
struct _ZSt5queueIjSt5dequeIjSaIjEEE decode_slot = {{{{{0}}}}};
struct _ZSt5mutex mt_encode_slot = {{{0}}};
#line 167 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
struct _ZSt5mutex mt_decode_slot = {{{0}}};
struct _ZSt18condition_variable cv_encode_slot = {{0}};
#line 168 "C:\\Users\\captain3060\\Projects\\PBL7_DUT_ECC_CUDA\\rs_cuda\\rs_cuda\\kernel.cu"
struct _ZSt18condition_variable cv_decode_slot = {{0}};
extern void *__dso_handle;
#line 624 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\system_error"
 /* COMDAT group: _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static */ struct _ZSt27_Constexpr_immortalize_implISt25_Iostream_error_category2E _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static = {{{{0}}}};
 /* COMDAT group: _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static */ unsigned __int64 _ZGVZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static;
extern struct __C7 *__curr_eh_stack_entry;
extern unsigned short __eh_curr_region;
extern int __catch_clause_number;
extern  /* COMDAT group: _ZTISt9exception */ const struct __class_type_info _ZTISt9exception;
extern  /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13];
extern  /* COMDAT group: _ZTISt9bad_alloc */ const struct __si_class_type_info _ZTISt9bad_alloc;
extern  /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13];
extern  /* COMDAT group: _ZTSSt20bad_array_new_length */ const char _ZTSSt20bad_array_new_length[25];
extern  /* COMDAT group: _ZTISt13runtime_error */ const struct __si_class_type_info _ZTISt13runtime_error;
extern  /* COMDAT group: _ZTSSt13runtime_error */ const char _ZTSSt13runtime_error[18];
extern  /* COMDAT group: _ZTISt13_System_error */ const struct __si_class_type_info _ZTISt13_System_error;
extern  /* COMDAT group: _ZTSSt13_System_error */ const char _ZTSSt13_System_error[18];
extern  /* COMDAT group: _ZTSSt12system_error */ const char _ZTSSt12system_error[17];
extern  /* COMDAT group: _ZTSSt8bad_cast */ const char _ZTSSt8bad_cast[12];
extern  /* COMDAT group: _ZTSNSt8ios_base7failureE */ const char _ZTSNSt8ios_base7failureE[22];
#line 115 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
extern int _ZNSt6locale2id7_Id_cntE;
#line 2680 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
extern struct _ZNSt6locale2idE _ZNSt5ctypeIcE2idE;
#line 419 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
extern  /* COMDAT group: _ZNSt9_FacetptrISt5ctypeIcEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt5ctypeIcEE6_PsaveE;
#line 419 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
extern  /* COMDAT group: _ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE;
#line 1602 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocnum"
 /* COMDAT group: _ZNSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE2idE */ struct _ZNSt6locale2idE _ZNSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE2idE = {0};
#line 255 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocnum"
 /* COMDAT group: _ZNSt8numpunctIcE2idE */ struct _ZNSt6locale2idE _ZNSt8numpunctIcE2idE = {0};
#line 1303 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xmemory"
static const struct _ZSt15_Fake_allocator __nv_static_30__dcb4e26f_9_kernel_cu_f9c6e15d__ZN39_INTERNAL_dcb4e26f_9_kernel_cu_f9c6e15dSt11_Fake_allocE;
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\iostream"
extern _ZSt7ostream _ZSt4cout;
extern _ZSt7ostream _ZSt4cerr;
 /* COMDAT group: _ZNSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE2idE */ unsigned __int64 _ZGVNSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE2idE;
 /* COMDAT group: _ZNSt8numpunctIcE2idE */ unsigned __int64 _ZGVNSt8numpunctIcE2idE;
 /* COMDAT group: _ZTISt20bad_array_new_length */ const struct __si_class_type_info _ZTISt20bad_array_new_length = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt20bad_array_new_length}},((const struct __class_type_info *)(&_ZTISt9bad_alloc.base))};
 /* COMDAT group: _ZTISt12system_error */ const struct __si_class_type_info _ZTISt12system_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt12system_error}},((const struct __class_type_info *)(&_ZTISt13_System_error.base))};
 /* COMDAT group: _ZTISt8bad_cast */ const struct __si_class_type_info _ZTISt8bad_cast = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt8bad_cast}},(&_ZTISt9exception)};
 /* COMDAT group: _ZTINSt8ios_base7failureE */ const struct __si_class_type_info _ZTINSt8ios_base7failureE = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSNSt8ios_base7failureE}},((const struct __class_type_info *)(&_ZTISt12system_error.base))};
 /* COMDAT group: _ZTISt9exception */ const struct __class_type_info _ZTISt9exception = {{(_ZTVN10__cxxabiv117__class_type_infoE + 2),_ZTSSt9exception}};
 /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13] = "St9exception";
 /* COMDAT group: _ZTISt9bad_alloc */ const struct __si_class_type_info _ZTISt9bad_alloc = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt9bad_alloc}},(&_ZTISt9exception)};
 /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13] = "St9bad_alloc";
 /* COMDAT group: _ZTSSt20bad_array_new_length */ const char _ZTSSt20bad_array_new_length[25] = "St20bad_array_new_length";
 /* COMDAT group: _ZTISt13runtime_error */ const struct __si_class_type_info _ZTISt13runtime_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt13runtime_error}},(&_ZTISt9exception)};
 /* COMDAT group: _ZTSSt13runtime_error */ const char _ZTSSt13runtime_error[18] = "St13runtime_error";
 /* COMDAT group: _ZTISt13_System_error */ const struct __si_class_type_info _ZTISt13_System_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt13_System_error}},((const struct __class_type_info *)(&_ZTISt13runtime_error.base))};
 /* COMDAT group: _ZTSSt13_System_error */ const char _ZTSSt13_System_error[18] = "St13_System_error";
 /* COMDAT group: _ZTSSt12system_error */ const char _ZTSSt12system_error[17] = "St12system_error";
 /* COMDAT group: _ZTSSt8bad_cast */ const char _ZTSSt8bad_cast[12] = "St8bad_cast";
 /* COMDAT group: _ZTSNSt8ios_base7failureE */ const char _ZTSNSt8ios_base7failureE[22] = "NSt8ios_base7failureE";
#line 419 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
 /* COMDAT group: _ZNSt9_FacetptrISt5ctypeIcEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt5ctypeIcEE6_PsaveE = ((const struct _ZNSt6locale5facetE *)0i64);
#line 419 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xlocale"
 /* COMDAT group: _ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE = ((const struct _ZNSt6locale5facetE *)0i64);
#line 1303 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\include\\xmemory"
static const struct _ZSt15_Fake_allocator __nv_static_30__dcb4e26f_9_kernel_cu_f9c6e15d__ZN39_INTERNAL_dcb4e26f_9_kernel_cu_f9c6e15dSt11_Fake_allocE = {0};

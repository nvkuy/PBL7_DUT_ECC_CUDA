uniform const uint32 MOD = 65537;
uniform const uint32 SPECIAL = MOD - 1;
uniform const uint32 ROOT = 3;
uniform const uint32 ROOT_INV = 21846;
uniform const uint32 MAX_LOG = 16;

uniform const uint32 ALGO_N_2_CUTOFF = 64;

uniform const uint32 LOG_DATA = 10;
uniform const uint32 LOG_SYMBOL = LOG_DATA - 1;
uniform const uint32 LOG_SEG = LOG_SYMBOL - 1;
uniform const uint32 SYMBOL_PER_PACKET = 1 << LOG_SYMBOL;
uniform const uint32 NUM_OF_PACKET = 1 << (MAX_LOG - LOG_SYMBOL);
uniform const uint32 NUM_OF_NEED_PACKET = NUM_OF_PACKET >> 1;
uniform const uint32 SEG_PER_PACKET = 1 << LOG_SEG;
uniform const uint32 SEG_DIFF = 1 << (MAX_LOG - 1);
uniform const uint32 NUM_OF_NEED_SYMBOL = 1 << (MAX_LOG - 1);

uniform const uint32 LEN_ROOT_LAYER_POW = (1 << MAX_LOG) - 1;
uniform const uint32 LEN_ROOT_LAYER_POW_2 = LEN_ROOT_LAYER_POW << 1;
uniform const uint32 LEN_N_POS = ((1 << (MAX_LOG + 1)) - 1);
uniform const uint32 LEN_PACKET_PRODUCT = NUM_OF_PACKET * (SYMBOL_PER_PACKET << 1);
uniform const uint32 LEN_ONE_PACKET_PRODUCT = 1 << (LOG_SYMBOL + 1);

uniform const uint32 LEN_SMALL = NUM_OF_NEED_SYMBOL;
uniform const uint32 LEN_LARGE = LEN_SMALL << 1;

static inline uniform uint32
add_small_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
    uniform uint32 res = a + b;
    if (res >= MOD) res -= MOD;
    return res;
}

static inline varying uint32
add_small_mod(
    varying uint32 a, 
    varying uint32 b
)
{
    varying uint32 res = a + b;
    if (res >= MOD) res -= MOD;
    return res;
}

static inline uniform uint32
sub_small_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
    uniform uint32 res = a + MOD - b;
    if (res >= MOD) res -= MOD;
    return res;
}

static inline varying uint32
sub_small_mod(
    varying uint32 a, 
    varying uint32 b
)
{
    varying uint32 res = a + MOD - b;
    if (res >= MOD) res -= MOD;
    return res;
}

static inline uniform uint32 
fast_mod(uniform uint32 x) {

	// if (x < MOD) return x;
	// uniform uint32 res = MOD + (x & 0xFFFFU) - (x >> 16);
	// if (res >= MOD) res -= MOD;
	// return res;

    return sub_small_mod((x & 0xFFFFU), (x >> 16));

}

static inline varying uint32 
fast_mod(varying uint32 x) {

	// if (x < MOD) return x;
	// varying uint32 res = MOD + (x & 0xFFFFU) - (x >> 16);
	// if (res >= MOD) res -= MOD;
	// return res;

    return sub_small_mod((x & 0xFFFFU), (x >> 16));

}

static inline uniform uint32 
mul_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
	// if (a == SPECIAL && b == SPECIAL)
	// 	return 1; // overflow
	// return (a * b) % MOD;
    return (a == SPECIAL && b == SPECIAL) ? 1 : fast_mod(a * b);
}

static inline varying uint32 
mul_mod(
    varying uint32 a, 
    varying uint32 b
)
{
	// if (a == SPECIAL && b == SPECIAL)
	// 	return 1; // overflow
	// return (a * b) % MOD;
    return (a == SPECIAL && b == SPECIAL) ? 1 : fast_mod(a * b);
}

static inline uniform uint32 
div_mod(
    uniform uint32 a, 
    uniform uint32 b, 
    uniform uint32 * uniform inv
)
{
	return mul_mod(a, inv[b]);
}

static inline varying uint32 
div_mod(
    varying uint32 a, 
    varying uint32 b, 
    uniform uint32 * uniform inv
)
{
	return mul_mod(a, inv[b]);
}

static inline uniform uint32  
add_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
	// return (a + b) % MOD;
    return fast_mod(a + b);
}

static inline varying uint32  
add_mod(
    varying uint32 a, 
    varying uint32 b
)
{
	// return (a + b) % MOD;
    return fast_mod(a + b);
}

static inline uniform uint32  
sub_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
	// return (a - b + MOD) % MOD;
    return fast_mod(a - b + MOD);
}

static inline varying uint32  
sub_mod(
    varying uint32 a, 
    varying uint32 b
)
{
	// return (a - b + MOD) % MOD;
    return fast_mod(a - b + MOD);
}

static inline uniform uint32  
pow_mod(
    uniform uint32 a, 
    uniform uint32 b
)
{
	uniform uint32 res = 1;
	while (b > 0)
	{
		if (b & 1)
			res = mul_mod(res, a);
		a = mul_mod(a, a);
		b >>= 1;
	}
	return res;
}

static inline unmasked void
vector_cpy(
    uniform uint32 * uniform a, 
    uniform uint32 * uniform b, 
    uniform uint32 n
) 
{

    foreach (i = 0 ... n) {
        b[i] = a[i];
    }

}

static inline unmasked void
vector_fill(
    uniform uint32 * uniform a, 
    uniform uint32 val, 
    uniform uint32 n
) 
{

    foreach (i = 0 ... n) {
        a[i] = val;
    }

}

static inline unmasked void
vector_mul(
    uniform uint32 * uniform a, 
    uniform uint32 * uniform b, 
    uniform uint32 * uniform c, 
    uniform uint32 n
) 
{

    foreach (i = 0 ... n) {
        c[i] = mul_mod(a[i], b[i]);
    }

}

// export void
static inline unmasked void
fnt(
    uniform uint32 * uniform a, 
    uniform uint32 * uniform b, 
    uniform uint32 log_na, 
    uniform uint32 log_nb, 
    uniform uint32 opt, 
    uniform uint32 * uniform n_pos,
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv)
{

    /*
	opt 2 bit: x1 x2
	 - x1: w_n or 1/w_n
	 - x2: need result * 1/n
	*/

	// size_b >= size_a;
	// need memset *b before use unless size_a == size_b

    uniform uint32 nb = 1 << log_nb, na = 1 << log_na;
    uniform uint32 st_n_pos = nb - 1, nbd2 = nb >> 1;
    uniform uint32 wp = (opt & 2) >> 1;

    foreach (i = 0 ... na) {
        b[n_pos[st_n_pos + i]] = a[i];
    }

    for (uniform uint32 i = 0; i < log_nb; i++) {

        uniform uint32 haft_len = 1 << i;
        uniform uint32 wlen_os = LEN_ROOT_LAYER_POW * wp + haft_len - 1;

        foreach (k = 0 ... nbd2) {

            varying uint32 bl_st = ((k >> i) << (i + 1)), th_id = (k & (haft_len - 1));
            varying uint32 pos = bl_st + th_id;
            varying uint32 u = b[pos];
            varying uint32 v = mul_mod(b[pos + haft_len], root_layer_pow[wlen_os + th_id]);
            // b[pos] = add_mod(u, v);
            // b[pos + haft_len] = sub_mod(u, v);
            b[pos] = add_small_mod(u, v);
            b[pos + haft_len] = sub_small_mod(u, v);

        }

    }

    if (opt & 1) {
        uniform uint32 inv_nb = div_mod(1, nb, inv);
        foreach (i = 0 ... nb) {
            b[i] = mul_mod(b[i], inv_nb);
        }
    }

    
}

static inline unmasked void
poly_mul(
    uniform uint32 * uniform a, 
    uniform uint32 * uniform b, 
    uniform uint32 * uniform t1, 
    uniform uint32 * uniform t2, 
    uniform uint32 * uniform c, 
    uniform uint32 log_n, 
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
)
{

    // 2 ^ log_n == size_a && size_a == size_b
	// *c == *a && *a + na == *b (allow)

    uniform uint32 na = 1 << log_n;
    uniform uint32 nc = na << 1;

    if (na > ALGO_N_2_CUTOFF) {

        vector_fill(t1, 0, nc);
        vector_fill(t2, 0, nc);

        fnt(a, t1, log_n, log_n + 1, 0, n_pos, root_layer_pow, inv);
        fnt(b, t2, log_n, log_n + 1, 0, n_pos, root_layer_pow, inv);

        vector_mul(t1, t2, t1, nc);

        fnt(t1, c, log_n + 1, log_n + 1, 3, n_pos, root_layer_pow, inv);

    } else {

        vector_cpy(a, t1, na);
        vector_cpy(b, t2, na);
        vector_fill(c, 0, nc);

        for (uniform uint32 i = 0; i < na; i++) {
            foreach (j = 0 ... na) {
                c[i + j] = add_mod(c[i + j], mul_mod(t1[i], t2[j]));
            }
        }

    }

}

task void
t_poly_mul(
    uniform uint32 * uniform a, 
    uniform uint32 * uniform b, 
    uniform uint32 * uniform t1, 
    uniform uint32 * uniform t2, 
    uniform uint32 * uniform c, 
    uniform uint32 log_n, 
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
)
{

    poly_mul(a, b, t1, t2, c, log_n, n_pos, root_layer_pow, inv);

}

static inline unmasked void 
poly_deriv(
    uniform uint32 * uniform ax, 
    uniform uint32 * uniform dax
) 
{

    foreach (i = 0 ... LEN_SMALL) {
        dax[i] = mul_mod(ax[i + 1], i + 1);
    } 

}

export void 
build_product(
    uniform uint32 * uniform p, 
    uniform uint32 * uniform t1, 
    uniform uint32 * uniform t2, 
    uniform uint32 log_n1, 
    uniform uint32 log_n2, 
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
)
{

    uniform uint32 final_poly_len = 1 << log_n2;

    for (uniform uint32 i = log_n1; i < log_n2; i++) {

        uniform uint32 poly_len_cur = 1 << i;
        uniform uint32 poly_len_next = poly_len_cur << 1;
        for (uniform uint32 st = 0; st < final_poly_len; st += poly_len_next) {

            // poly_mul(
            //     p + st, 
            //     p + st + poly_len_cur, 
            //     t1 + st, t2 + st, 
            //     p + st, i, n_pos, root_layer_pow, inv
            // );

            launch t_poly_mul(
                p + st, 
                p + st + poly_len_cur, 
                t1 + st, t2 + st, 
                p + st, i, n_pos, root_layer_pow, inv
            );

        }

        sync;

    }

}

static inline unmasked void 
build_ax(
    uniform uint32 * uniform x, 
    uniform uint32 * uniform ax, 
    uniform uint32 * uniform t1, 
    uniform uint32 * uniform t2,
    uniform uint32 * uniform packet_product,
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
)
{

	for (uniform uint32 i = 0; i < NUM_OF_NEED_PACKET; i++)
	{
		uniform uint32 st_p1 = i << (LOG_SYMBOL + 1), st_p2 = x[i << LOG_SYMBOL] << 2;
		vector_cpy(packet_product + st_p2, ax + st_p1, LEN_ONE_PACKET_PRODUCT);
	}

	build_product(ax, t1, t2, LOG_SYMBOL + 1, MAX_LOG, n_pos, root_layer_pow, inv);
}

static inline unmasked void 
build_n1(
    uniform uint32 * uniform n1, 
    uniform uint32 * uniform vdax, 
    uniform uint32 * uniform x, 
    uniform uint32 * uniform y, 
    uniform uint32 * uniform inv
)
{

    foreach (i = 0 ... LEN_SMALL) {
		n1[i] = div_mod(y[i], vdax[x[i]], inv);
    }

}

static inline unmasked void 
build_n2(
    uniform uint32 * uniform n2, 
    uniform uint32 * uniform n1, 
    uniform uint32 * uniform x
)
{

    foreach (i = 0 ... LEN_SMALL) {
		n2[x[i]] = n1[i];
    }

}

static inline unmasked void 
build_n3(
    uniform uint32 * uniform n3, 
    uniform uint32 * uniform p_n3
)
{

    foreach (i = 0 ... LEN_SMALL) {
		n3[i] = sub_mod(0, p_n3[i + 1]);
    }

}

static inline unmasked void 
build_px(
    uniform uint32 * uniform p, 
    uniform uint32 * uniform ax, 
    uniform uint32 * uniform n3, 
    uniform uint32 * uniform t1, 
    uniform uint32 * uniform t2, 
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
)
{

	// launch t_poly_mul(ax, n3, t1, t2, p, MAX_LOG - 1, n_pos, root_layer_pow, inv);
    // sync;
    poly_mul(ax, n3, t1, t2, p, MAX_LOG - 1, n_pos, root_layer_pow, inv);

}

// task void
// t_encode(
//     uniform uint32 * uniform p, 
//     uniform uint32 * uniform y,
//     uniform uint32 * uniform n_pos, 
//     uniform uint32 * uniform root_layer_pow, 
//     uniform uint32 * uniform inv
// )
// {

//     vector_fill(y, 0, LEN_LARGE);
//     fnt(p, y, MAX_LOG - 1, MAX_LOG, 0, n_pos, root_layer_pow, inv);

// }

// task void
// t_decode(
//     uniform uint32 * uniform x, 
//     uniform uint32 * uniform y, 
//     uniform uint32 * uniform p,
//     uniform uint32 * uniform t1,
//     uniform uint32 * uniform t2,
//     uniform uint32 * uniform ax,
//     uniform uint32 * uniform dax,
//     uniform uint32 * uniform vdax,
//     uniform uint32 * uniform n1,
//     uniform uint32 * uniform n2,
//     uniform uint32 * uniform n3,
//     uniform uint32 * uniform packet_product,
//     uniform uint32 * uniform n_pos, 
//     uniform uint32 * uniform root_layer_pow, 
//     uniform uint32 * uniform inv
// )
// {

//     build_ax(x, ax, t1, t2, packet_product, n_pos, root_layer_pow, inv);

//     poly_deriv(ax, dax);

//     vector_fill(vdax, 0, LEN_LARGE);
//     fnt(dax, vdax, MAX_LOG - 1, MAX_LOG, 0, n_pos, root_layer_pow, inv);

//     build_n1(n1, vdax, x, y, inv);

//     vector_fill(n2, 0, LEN_LARGE);
//     build_n2(n2, n1, x);

//     fnt(n2, t2, MAX_LOG, MAX_LOG, 2, n_pos, root_layer_pow, inv);

//     build_n3(n3, t2);

//     build_px(n2, ax, n3, t1, t2, n_pos, root_layer_pow, inv);

//     vector_cpy(n2, p, LEN_SMALL);

// }

export void encode(
    uniform uint32 * uniform p, 
    uniform uint32 * uniform y,
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
) 
{

    // unmasked {
    //     launch t_encode(p, y, n_pos, root_layer_pow, inv);
    //     // sync;
    // }

    vector_fill(y, 0, LEN_LARGE);
    fnt(p, y, MAX_LOG - 1, MAX_LOG, 0, n_pos, root_layer_pow, inv);

}

export void decode(
    uniform uint32 * uniform x, 
    uniform uint32 * uniform y, 
    uniform uint32 * uniform p,
    uniform uint32 * uniform t1,
    uniform uint32 * uniform t2,
    uniform uint32 * uniform ax,
    uniform uint32 * uniform dax,
    uniform uint32 * uniform vdax,
    uniform uint32 * uniform n1,
    uniform uint32 * uniform n2,
    uniform uint32 * uniform n3,
    uniform uint32 * uniform packet_product,
    uniform uint32 * uniform n_pos, 
    uniform uint32 * uniform root_layer_pow, 
    uniform uint32 * uniform inv
) 
{

    // unmasked {
    //     launch t_decode(x, y, p, t1, t2, ax, dax, vdax, n1, n2, n3, 
    //         packet_product, n_pos, root_layer_pow, inv);
    //     // sync;
    // }

    build_ax(x, ax, t1, t2, packet_product, n_pos, root_layer_pow, inv);

    poly_deriv(ax, dax);

    vector_fill(vdax, 0, LEN_LARGE);
    fnt(dax, vdax, MAX_LOG - 1, MAX_LOG, 0, n_pos, root_layer_pow, inv);

    build_n1(n1, vdax, x, y, inv);

    vector_fill(n2, 0, LEN_LARGE);
    build_n2(n2, n1, x);

    fnt(n2, t2, MAX_LOG, MAX_LOG, 2, n_pos, root_layer_pow, inv);

    build_n3(n3, t2);

    build_px(n2, ax, n3, t1, t2, n_pos, root_layer_pow, inv);

    vector_cpy(n2, p, LEN_SMALL);

}

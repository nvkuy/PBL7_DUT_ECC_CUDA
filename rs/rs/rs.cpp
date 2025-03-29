#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>

using namespace std;

const unsigned MOD = 65537;
const unsigned SPECIAL = MOD - 1;
const unsigned ROOT = 3;
const unsigned ROOT_INV = 21846;
const unsigned MAX_LOG = 16;

const unsigned WARP_SIZE = 32;
const unsigned POLY_CUTOFF = WARP_SIZE << 1;

const unsigned LOG_DATA = 10;
const unsigned LOG_SYMBOL = LOG_DATA - 1;
const unsigned LOG_SEG = LOG_SYMBOL - 1;
const unsigned DATA_PER_PACKET = 1 << LOG_DATA; // in bytes
const unsigned SYMBOL_PER_PACKET = 1 << LOG_SYMBOL;
const unsigned NUM_OF_PACKET = 1 << (MAX_LOG - LOG_SYMBOL);
const unsigned NUM_OF_NEED_PACKET = NUM_OF_PACKET >> 1;
const unsigned SEG_PER_PACKET = 1 << LOG_SEG;
const unsigned SEG_DIFF = 1 << (MAX_LOG - 1);
const unsigned NUM_OF_NEED_SYMBOL = 1 << (MAX_LOG - 1);

unsigned** N_pos;
unsigned* root_pow;
unsigned* root_inv_pow;
unsigned*** root_layer_pow;
unsigned* inv;

vector<vector<unsigned>> packet_product;

inline unsigned mul_mod(unsigned a, unsigned b) {
    // TODO: use 32-bit simd
    if (a == b && a == SPECIAL) return 1; // overflow
    return (a * b) % MOD;
}

inline unsigned div_mod(unsigned a, unsigned b) {
    return mul_mod(a, inv[b]);
}

inline unsigned add_mod(unsigned a, unsigned b) {
    return (a + b) % MOD;
}

inline unsigned sub_mod(unsigned a, unsigned b) {
    return (a - b + MOD) % MOD;
}

inline unsigned pow_mod(unsigned a, unsigned b) {
    unsigned res = 1;
    while (b > 0) {
        if (b & 1) res = mul_mod(res, a);
        a = mul_mod(a, a);
        b >>= 1;
    }
    return res;
}


vector<unsigned> fnt(const vector<unsigned>& a, unsigned log_n, unsigned opt) {
    // TODO: use spmd for j in each layer i, need sync each layer 

    /*
    opt 2 bit: x1 x2
     - x1: w_n or 1/w_n
     - x2: need result * 1/n
    */

    unsigned n = 1 << log_n, haft_n = n >> 1, wp = (opt & 2) >> 1;
    vector<unsigned> b(n, 0);
    for (unsigned i = 0; i < a.size(); i++) b[N_pos[log_n][i]] = a[i];

    for (unsigned i = 0; i < log_n; i++) {
        unsigned haft_len = 1 << i;
        for (unsigned j = 0; j < haft_n; j++) {
            unsigned bl_st = ((j >> i) << (i + 1)), th_id = (j & (haft_len - 1));
            unsigned pos = bl_st + th_id;
            unsigned u = b[pos], v = mul_mod(b[pos + haft_len], root_layer_pow[wp][i][th_id]);
            b[pos] = add_mod(u, v);
            b[pos + haft_len] = sub_mod(u, v);
        }
    }

    if (opt & 1) {
        for (unsigned i = 0; i < n; i++)
            b[i] = div_mod(b[i], n);
    }

    return b;

}

inline vector<unsigned> poly_mul(const vector<unsigned>& a, const vector<unsigned>& b, unsigned log_n) {

    unsigned n = 1 << log_n;
    vector<unsigned> c(n << 1, 0);

    if (n < POLY_CUTOFF) {

        for (unsigned i = 0; i < n; i++)
            for (unsigned j = 0; j < n; j++)
                c[i + j] = add_mod(c[i + j], mul_mod(a[i], b[j]));

    }
    else {

        vector<unsigned> fa = fnt(a, log_n + 1, 0);
        vector<unsigned> fb = fnt(b, log_n + 1, 0);

        for (unsigned i = 0; i < c.size(); i++)
            c[i] = mul_mod(fa[i], fb[i]);

        c = fnt(c, log_n + 1, 3);

    }

    return c;

}

inline vector<unsigned> poly_deriv(const vector<unsigned>& p) {

    vector<unsigned> pd(p.size(), 0);
    for (unsigned i = 1; i < p.size(); i++)
        pd[i - 1] = mul_mod(p[i], i);

    return pd;

}

inline vector<unsigned> build_product(const vector<vector<unsigned>>& p, unsigned log_n1, unsigned log_n2) {

    // TODO: parallel for each j in layer i

    unsigned n = 1 << log_n2;
    vector<unsigned> product(n);
    for (unsigned i = 0; i < n; i++) {
        unsigned bl_id = i >> log_n1, th_id = i & ((1 << log_n1) - 1);
        product[i] = p[bl_id][th_id];
    }

    for (unsigned i = log_n1; i < log_n2; i++) {
        unsigned len = 1 << i, m = n >> (i + 1);
        for (unsigned j = 0; j < m; j++) {
            unsigned st = j << (i + 1);
            vector<unsigned> p1(product.begin() + st, product.begin() + st + len);
            vector<unsigned> p2(product.begin() + st + len, product.begin() + st + (len << 1));
            vector<unsigned> p3 = poly_mul(p1, p2, i);
            for (unsigned k = 0; k < p3.size(); k++)
                product[k + st] = p3[k];
        }
    }

    return product;

}

inline vector<unsigned> build_ax(const vector<unsigned>& x) {

    vector<vector<unsigned>> p(NUM_OF_NEED_PACKET);
    for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++)
        p[i] = packet_product[x[i << LOG_SYMBOL] >> LOG_SEG];

    return build_product(p, LOG_SYMBOL + 1, MAX_LOG);

}

vector<unsigned> encode(const vector<unsigned>& chunk) {
    return fnt(chunk, MAX_LOG, 0);
}

vector<unsigned> decode(const vector<unsigned>& x, const vector<unsigned>& y) {

    vector<unsigned> ax = build_ax(x);

    vector<unsigned> dax = poly_deriv(ax);

    vector<unsigned> vdax = fnt(dax, MAX_LOG, 0);
    vector<unsigned> n1(NUM_OF_NEED_SYMBOL);
    for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
        n1[i] = div_mod(y[i], vdax[x[i]]);

    vector<unsigned> n2(1 << MAX_LOG, 0);
    for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
        n2[x[i]] = n1[i];

    vector<unsigned> vn2 = fnt(n2, MAX_LOG, 2);
    unsigned vn2_0 = vn2[0];
    for (unsigned i = 1; i < vn2.size(); i++)
        vn2[i - 1] = sub_mod(MOD, vn2[i]);
    vn2[vn2.size() - 1] = sub_mod(MOD, vn2_0);

    vector<unsigned> px = poly_mul(ax, vn2, MAX_LOG - 1);
    return vector<unsigned>(px.begin(), px.begin() + NUM_OF_NEED_SYMBOL);

}

void init() {
    // TODO: offline process

    N_pos = new unsigned* [MAX_LOG + 1];
    for (unsigned i = 1; i <= MAX_LOG; i++) {
        unsigned n = 1 << i;
        N_pos[i] = new unsigned[n];
        for (unsigned j = 0; j < n; j++) N_pos[i][j] = j;
    }

    for (unsigned i = 1; i <= MAX_LOG; i++) {
        unsigned n = 1 << i;
        for (unsigned j = 0; j < n; j++) {
            unsigned rev_num = 0;
            for (unsigned k = 0; k < i; k++) {
                if (j & (1 << k))
                    rev_num |= (1 << (i - 1 - k));
            }
            if (j < rev_num)
                swap(N_pos[i][j], N_pos[i][rev_num]);
        }
    }

    root_pow = new unsigned[MOD];
    root_inv_pow = new unsigned[MOD];
    inv = new unsigned[MOD];
    root_pow[0] = 1, root_inv_pow[0] = 1;
    for (unsigned i = 1; i < MOD; i++) {
        root_pow[i] = mul_mod(root_pow[i - 1], ROOT);
        root_inv_pow[i] = mul_mod(root_inv_pow[i - 1], ROOT_INV);
        inv[i] = pow_mod(i, MOD - 2);
    }

    root_layer_pow = new unsigned** [2];
    for (unsigned i = 0; i < 2; i++) {
        root_layer_pow[i] = new unsigned* [MAX_LOG];
        for (unsigned j = 0; j < MAX_LOG; j++) {
            unsigned haft_len = 1 << j;
            root_layer_pow[i][j] = new unsigned[haft_len];
            unsigned ang = 1 << (MAX_LOG - j - 1);
            unsigned wn = i ? root_inv_pow[ang] : root_pow[ang], w = 1;
            for (unsigned k = 0; k < haft_len; k++) {
                root_layer_pow[i][j][k] = w;
                w = mul_mod(w, wn);
            }
        }
    }

    packet_product = vector<vector<unsigned>>(NUM_OF_PACKET);
    for (unsigned i = 0; i < NUM_OF_PACKET; i++) {
        vector<vector<unsigned>> p(SYMBOL_PER_PACKET, vector<unsigned>(2, 1));
        for (unsigned j = 0; j < SEG_PER_PACKET; j++) {
            unsigned k = i * SEG_PER_PACKET + j;
            p[j][0] = sub_mod(MOD, root_pow[k]);
            p[j + SEG_PER_PACKET][0] = sub_mod(MOD, root_pow[k + SEG_DIFF]);
        }
        packet_product[i] = build_product(p, 1, LOG_SYMBOL + 1);
    }

}

void fin() {
    // TODO: call at end of program

    for (unsigned i = 1; i <= MAX_LOG; i++)
        delete[] N_pos[i];
    delete[] N_pos;

    delete[] root_pow;
    delete[] root_inv_pow;
    delete[] inv;


    for (unsigned i = 0; i < 2; i++) {
        for (unsigned j = 0; j < MAX_LOG; j++)
            delete[] root_layer_pow[i][j];
        delete[] root_layer_pow[i];
    }
    delete[] root_layer_pow;

}

void testFNT();

void testEncodeDecode();

int main() {

    init();

    testFNT();

    testEncodeDecode();

    fin();

    return 0;
}

void testEncodeDecode() {

    srand(time(0));

    vector<unsigned> a(NUM_OF_NEED_SYMBOL);
    for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
        a[i] = rand() % (MOD - 1); // 2 bytes

    vector<unsigned> b = encode(a);

    vector<unsigned> x(NUM_OF_NEED_SYMBOL), y(NUM_OF_NEED_SYMBOL);

    for (unsigned i = 0; i < NUM_OF_NEED_PACKET; i++) {
        unsigned stx = i * SYMBOL_PER_PACKET;
        for (unsigned j = 0; j < SEG_PER_PACKET; j++) {
            x[stx + j] = stx + j;
            x[stx + j + SEG_PER_PACKET] = stx + j + SEG_DIFF;
            y[stx + j] = b[stx + j];
            y[stx + j + SEG_PER_PACKET] = b[stx + j + SEG_DIFF];
        }
    }

    vector<unsigned> c = decode(x, y);
    for (unsigned i = 0; i < NUM_OF_NEED_SYMBOL; i++)
        assert(a[i] == c[i]);

}

void testFNT() {

    unsigned log_n = 16;
    vector<unsigned> c1;
    for (unsigned i = 0; i < (1 << log_n); i++)
        c1.push_back(i);
    shuffle(c1.begin(), c1.end(), default_random_engine(time(NULL)));
    vector<unsigned> v = fnt(c1, log_n, 0);
    vector<unsigned> c2 = fnt(v, log_n, 3);
    for (unsigned i = 0; i < (1 << log_n); i++)
        assert(c1[i] == c2[i]);

    vector<unsigned> u = fnt(v, log_n, 2);
    for (unsigned i = 0; i < (1 << log_n); i++)
        assert(div_mod(u[i], (1 << log_n)) == c1[i]);

}
// Q-BTC OpenCL SHA-256 PoW Miner — 1:1 port of qbtc_sha256_miner.cu
//
// Features carried over from the CUDA kernel:
//   - block 1 midstate (computed on the host)
//   - per-launch precompute of block 2 rounds 0..10 plus the nonce-independent part of round 11
//   - WVAR_MASK: W[i] values that do not depend on nlo are computed on the host
//   - KW3: constant message schedule of block 3 (uploaded by the host, __constant)
//   - EARLY_REJECT: skip round 63 when cmp_hi == 0 and o1 > cmp_lo
//
// Header: timestamp(8BE)|prev(32)|merkle(32)|commit(32)|nonce(8BE)|target(8BE) = 120 B

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct {
    uint a11, e11, b11, c11, d11, f11, g11, h11;
    uint mid[2];
    uint mid_rest[6];
    uint nhi;
    uint cmp_hi, cmp_lo;
    uint kw[64];
    uint wc[64];
} QParams;

// OpenCL rotate() rotates LEFT, so a right rotation by n is rotate(x, 32 - n).
#define ROTR(x,n)  rotate((uint)(x), (uint)(32 - (n)))
#define CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define EP0(a)     (ROTR(a,2)^ROTR(a,13)^ROTR(a,22))
#define EP1(e)     (ROTR(e,6)^ROTR(e,11)^ROTR(e,25))
#define SIG0(x)    (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define SIG1(x)    (ROTR(x,17)^ROTR(x,19)^((x)>>10))

static __constant uint K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// Bit i is set <=> block 2 word i depends on W11 (= nonce_lo).
// After #pragma unroll the wvar(i) test folds into a constant.
#define WVAR_MASK 0xffffffffff540800UL
#define wvar(i)   (((WVAR_MASK)>>(i))&1UL)

// PoW prefix (o0:o1) for one nonce_lo.
// With cmp_hi == 0, o1 is already final after round 62 of block 3, so if
// o1 > cmp_lo round 63 is not needed and o0 = 0xFFFFFFFF (guaranteed reject).
inline void pow_hash(__constant const QParams* P,
                     __constant const uint*   KW3,
                     uint nlo, uint* o0, uint* o1)
{
    uint w[64];
    uint a = P->a11 + nlo, e = P->e11 + nlo;
    uint b = P->b11, c = P->c11, d = P->d11;
    uint f = P->f11, g = P->g11, h = P->h11;
    w[11] = nlo;

    #pragma unroll
    for (int i = 12; i < 64; i++) {
        uint kw;
        if (wvar(i)) {
            uint x = P->wc[i];
            if (wvar(i-2))  x += SIG1(w[i-2]);
            if (wvar(i-7))  x += w[i-7];
            if (wvar(i-15)) x += SIG0(w[i-15]);
            if (wvar(i-16)) x += w[i-16];
            w[i] = x;
            kw = K[i] + x;
        } else {
            kw = P->kw[i];
        }
        uint t1 = h + EP1(e) + CH(e,f,g) + kw;
        uint t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    // Final state of block 2 == initial state of block 3
    uint s0 = P->mid[0] + a, s1 = P->mid[1] + b;
    uint s2 = P->mid_rest[0] + c, s3 = P->mid_rest[1] + d;
    uint s4 = P->mid_rest[2] + e, s5 = P->mid_rest[3] + f;
    uint s6 = P->mid_rest[4] + g, s7 = P->mid_rest[5] + h;
    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 63; i++) {
        uint t1 = h + EP1(e) + CH(e,f,g) + KW3[i];
        uint t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    *o1 = s1 + a;
    if (P->cmp_hi == 0u && *o1 > P->cmp_lo) { *o0 = 0xFFFFFFFFu; return; }

    uint t1 = h + EP1(e) + CH(e,f,g) + KW3[63];
    uint t2 = EP0(a) + MAJ(a,b,c);
    *o0 = s0 + t1 + t2;
}

#define STOP_CHECK_INTERVAL 1024u

__kernel void qbtc_mine_kernel(
    __constant const QParams* __restrict P,
    const uint slo,                 // nonce_lo start
    const uint len,                 // range length
    const uint tot,                 // global size
    volatile __global ulong* __restrict found,
    volatile __global int*   __restrict stopf,
    __constant const uint* __restrict KW3)
{
    uint tid = get_global_id(0);
    if (tid >= len) return;

    uint cnt = (uint)(((ulong)(len - tid) + (ulong)tot - 1UL) / (ulong)tot);
    uint nlo = slo + tid;
    for (uint it = 0; it < cnt; it++, nlo += tot) {
        if ((it & (STOP_CHECK_INTERVAL - 1u)) == 0u) {
            if (*stopf) return;
            if (*found != 0xFFFFFFFFFFFFFFFFUL) return;
        }
        uint h0, h1;
        pow_hash(P, KW3, nlo, &h0, &h1);
        if (h0 < P->cmp_hi || (h0 == P->cmp_hi && h1 <= P->cmp_lo)) {
            atom_cmpxchg((volatile __global ulong*)found,
                         0xFFFFFFFFFFFFFFFFUL,
                         ((ulong)P->nhi << 32) | (ulong)nlo);
            return;
        }
    }
}
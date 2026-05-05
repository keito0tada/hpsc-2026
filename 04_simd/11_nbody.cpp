#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_loadu_ps(x);
  __m512 yvec = _mm512_loadu_ps(y);
  __m512 mvec = _mm512_loadu_ps(m);
  __m512i jidx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);

    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512 r2  = _mm512_add_ps(rx2, ry2);

    __mmask16 mask = _mm512_cmpneq_epi32_mask(jidx, _mm512_set1_epi32(i));
    __m512 r2_safe = _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0f), r2);

    __m512 rinv  = _mm512_rsqrt14_ps(r2_safe);
    __m512 rinv2 = _mm512_mul_ps(rinv, rinv);
    __m512 rinv3 = _mm512_mul_ps(rinv, rinv2);

    __m512 fxj = _mm512_mul_ps(rx, _mm512_mul_ps(mvec, rinv3));
    __m512 fyj = _mm512_mul_ps(ry, _mm512_mul_ps(mvec, rinv3));

    fx[i] -= _mm512_reduce_add_ps(fxj);
    fy[i] -= _mm512_reduce_add_ps(fyj);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}

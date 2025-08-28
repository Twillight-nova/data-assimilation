#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_uniform(void) {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

static double randn(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    return r * cos(theta);
}

static void forward(double dt, const double a[6], double x1, double y1, int nmax, double *x, double *y) {
    for (int i = 0; i < nmax; ++i) {
        x[i] = 0.0;
        y[i] = 0.0;
    }
    x[0] = x1;
    y[0] = y1;
    for (int n = 0; n < nmax - 1; ++n) {
        double gx = a[0] + a[1] * x[n] + a[2] * y[n];
        double gy = a[3] + a[4] * y[n] + a[5] * x[n];
        x[n + 1] = x[n] + dt * x[n] * gx;
        y[n + 1] = y[n] + dt * y[n] * gy;
    }
}

static void adjoint(double dt,
                    const double a[6],
                    const double *x,
                    const double *y,
                    const double *xo,
                    const double *yo,
                    const unsigned char *is_obs,
                    int nmax,
                    double grad[2]) {
    double *ax = (double*)calloc((size_t)nmax, sizeof(double));
    double *ay = (double*)calloc((size_t)nmax, sizeof(double));
    if (!ax || !ay) {
        fprintf(stderr, "Memory allocation failed in adjoint()\n");
        free(ax); free(ay);
        grad[0] = 0.0; grad[1] = 0.0;
        return;
    }

    for (int n = nmax - 2; n >= 0; --n) {
        if (is_obs[n]) {
            ax[n] = ax[n] + (x[n] - xo[n]);
            ay[n] = ay[n] + (y[n] - yo[n]);
        }
        ax[n] = ax[n] + dt * a[5] * y[n] * ay[n + 1];
        ay[n] = ay[n] + dt * a[4] * y[n] * ay[n + 1];
        ay[n] = ay[n] + (1.0 + dt * (a[3] + a[4] * y[n] + a[5] * x[n])) * ay[n + 1];
        ay[n] = ay[n] + dt * a[2] * x[n] * ax[n + 1];
        ax[n] = ax[n] + dt * a[1] * x[n] * ax[n + 1];
        ax[n] = ax[n] + (1.0 + dt * (a[0] + a[1] * x[n] + a[2] * y[n])) * ax[n + 1];
    }

    grad[0] = ax[0];
    grad[1] = ay[0];
    free(ax);
    free(ay);
}

static double calc_cost_on_obs(const double *xf, const double *yf,
                               const double *xo, const double *yo,
                               const unsigned char *is_obs,
                               int nmax) {
    double cost = 0.0;
    for (int i = 0; i < nmax; ++i) {
        if (!is_obs[i]) continue;
        double dx = xf[i] - xo[i];
        double dy = yf[i] - yo[i];
        cost += dx * dx + dy * dy;
    }
    return cost;
}

static double stddev_population(const double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += arr[i];
    double mean = sum / (double)n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = arr[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    return sqrt(var);
}

int main(void) {
    const int nmax = 3001;
    const double dt = 0.001;
    const double at[6] = {4.0, -2.0, -4.0, -6.0, 2.0, 4.0};
    const double obs_x1 = 1.0, obs_y1 = 1.0;

    /* 再現性のため固定シード（必要に応じて time(NULL) に変更可） */
    srand(42u);

    /* 観測時刻: 1, 11, 21, ... */
    unsigned char *is_obs = (unsigned char*)calloc((size_t)nmax, 1);
    if (!is_obs) {
        fprintf(stderr, "Memory allocation failed for is_obs\n");
        return 1;
    }
    for (int i = 1; i < nmax; i += 10) is_obs[i] = 1u;

    /* 観測データ生成 */
    double *xt = (double*)malloc((size_t)nmax * sizeof(double));
    double *yt = (double*)malloc((size_t)nmax * sizeof(double));
    if (!xt || !yt) {
        fprintf(stderr, "Memory allocation failed for xt/yt\n");
        free(is_obs); free(xt); free(yt);
        return 1;
    }
    forward(dt, at, obs_x1, obs_y1, nmax, xt, yt);
    for (int i = 0; i < nmax; ++i) {
        yt[i] += 0.05 * randn();
    }
    double y_std = stddev_population(yt, nmax);

    /* 観測値を xo, yo として利用 */
    const double *xo = xt;
    const double *yo = yt;

    /* 最適化対象のパラメータ: a は固定, x0, y0 を推定 */
    double a[6];
    for (int i = 0; i < 6; ++i) a[i] = at[i];
    double x0 = 1.5, y0 = 1.5;

    const int maxiter = 2000;
    const double alpha = 1.0e-3;
    const double gtol = 1.0e-10;

    double *x = (double*)malloc((size_t)nmax * sizeof(double));
    double *y = (double*)malloc((size_t)nmax * sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed for x/y\n");
        free(is_obs); free(xt); free(yt); free(x); free(y);
        return 1;
    }

    int niter = 0;
    double last_cost = NAN;
    double last_gnorm = NAN;

    while (niter < maxiter) {
        forward(dt, a, x0, y0, nmax, x, y);
        double cost = calc_cost_on_obs(x, y, xo, yo, is_obs, nmax);

        double grad[2] = {0.0, 0.0};
        adjoint(dt, a, x, y, xo, yo, is_obs, nmax, grad);
        double gnorm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

        last_cost = cost;
        last_gnorm = gnorm;

        if (gnorm < gtol) {
            break;
        }

        /* 勾配降下: x0, y0 のみ更新 */
        x0 -= alpha * grad[0];
        y0 -= alpha * grad[1];

        ++niter;
    }

    printf("FINAL:%d,%.15g,%.15g,%.15g,%.15g,%.15g\n",
           niter, last_cost, last_gnorm, x0, y0, y_std);

    free(is_obs);
    free(xt);
    free(yt);
    free(x);
    free(y);
    return 0;
}



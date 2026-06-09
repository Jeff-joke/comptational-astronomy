#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "fmm.h"
#include "quadtree.h"
#include "operators.h"

#pragma omp declare target
/* 由 node_id 反查它所在的 level：最大的 lv 使得 base_arr[lv] <= node_id。 */
static int find_level(int node_id, const int *base_arr, int max_level)
{
    for (int lv = max_level; lv >= 0; lv--)
        if (node_id >= base_arr[lv])
            return lv;
    return 0;
}
#pragma omp end declare target

static void fmm_solve(Particle *particles, int n, int MAX_LEVEL, int MAX_PER_LEAF, bool use_gpu)
{
    double binom[2 * MAX_P][2 * MAX_P];
    precompute_binomial(binom);

    HilbertTree *tree = hilbert_tree_build(particles, n, MAX_PER_LEAF, MAX_LEVEL);
    /* P2M & M2M from deepest level upward */
    double complex *multipole = calloc(tree->total_nodes * MAX_P, sizeof(double complex));

    bool *has_multi = calloc(tree->total_nodes, sizeof(bool));

    /* 預計算每一層的 base 和 count */
    int base_arr[MAX_LEVEL + 1], count_arr[MAX_LEVEL + 1];
    base_arr[0] = 0;  count_arr[0] = 1;
    for (int lv = 1; lv <= tree->max_level; lv++) {
        base_arr[lv] = base_arr[lv-1] + count_arr[lv-1];
        count_arr[lv] = count_arr[lv-1] * 4;
    }

    double complex *local_exp = calloc(tree->total_nodes * MAX_P, sizeof(double complex));

    int total_nodes = tree->total_nodes;
    int n_particles = tree->n_particles;
    int num_leaves = tree->num_leaves;
    /* 將所有計算所需資料映射至GPU記憶體中*/
    #pragma omp target data if(use_gpu) \
                            map(to: binom, base_arr, count_arr) \
                            map(to: tree[0:1], \
                                    tree->sort_indices[0:n_particles], \
                                    tree->node_start[0:total_nodes], \
                                    tree->node_end[0:total_nodes], \
                                    tree->leaf_indices[0:num_leaves], \
                                    tree->node_cx[0:total_nodes], \
                                    tree->node_cy[0:total_nodes], \
                                    tree->node_size[0:total_nodes], \
                                    tree->sort_x[0:total_nodes], \
                                    tree->sort_y[0:n_particles], \
                                    tree->sort_mass[0:n_particles]) \
                            map(alloc:  tree->sort_potential[0:n_particles], \
                                        tree->sort_fx[0:n_particles], \
                                        tree->sort_fy[0:n_particles]) \
                            map(tofrom: multipole[0:total_nodes * MAX_P]) \
                            map(tofrom: local_exp[0:total_nodes * MAX_P]) \
                            map(tofrom: has_multi[0:total_nodes])
    {
        //開始多線程計算P2M
        #pragma omp target teams distribute parallel for if(use_gpu)
        for (int i = 0; i < tree->num_leaves; i++) {
            int leaf_id = tree->leaf_indices[i];
            P2M(tree, leaf_id, multipole);
            has_multi[leaf_id] = true;
        }


        // /*for debug*/
        // #ifdef DEBUG
        // FILE *fp_1 = fopen("multipole_log_1.txt", "a");
        // for (int i = 0; i < tree->total_nodes; i++) {
        //     if(has_multi[i]) {
        //         for (int j = 0; j < MAX_P; j++) {
        //             fprintf(fp_1, "%f ", creal(multipole[i * MAX_P + j]));
        //         }
        //         fprintf(fp_1, "\n");
        //     }
        // }
        // fclose(fp_1);
        // #endif

        /*從最深層開始，先判斷自己的M2M_result[i]是否為0，若為零則檢查子節點的M2M_result是否為0，若不為零則計算M2M，並將結果存入M2M_result[i]，如過M2M_result[i]不為零則跳過計算*/
        for (int lv = tree->max_level - 1; lv >= 0; lv--) {
            #pragma omp target teams distribute parallel for if(use_gpu)
            for (int idx = 0; idx < count_arr[lv]; idx++) {
                int parent_id = base_arr[lv] + idx;
                if (tree->node_start[parent_id] > tree->node_end[parent_id])
                    continue;  // 空節點跳過

                /* 如果自己已經算過（例如它是葉節點），直接跳過 */
                if (has_multi[parent_id])
                    continue;

                for (int c = 0; c < 4; c++) {
                    int child_id = base_arr[lv+1] + 4 * idx + c;
                    if (tree->node_start[child_id] > tree->node_end[child_id])
                        continue;  // 空子節點跳過

                    /* 檢查子節點是否已經有結果，若有才進行 M2M */
                    if (!has_multi[child_id])
                        continue;
                    /* 將子節點的結果標記為已計算 */
                    has_multi[parent_id] = true;
                    M2M(tree, parent_id, child_id, multipole, binom);
                }
            }
        }

        // /*for debug*/
        // #ifdef DEBUG
        // FILE *fp_2 = fopen("multipole_log_2.txt", "a");
        // for (int i = 0; i < tree->total_nodes; i++) {
        //     if(has_multi[i]) {
        //         for (int j = 0; j < MAX_P; j++) {
        //             fprintf(fp_2, "%f ", creal(multipole[i * MAX_P + j]));
        //         }
        //         fprintf(fp_2, "\n");
        //     }
        // }
        // fclose(fp_2);
        // #endif

        /* ---- Downward pass: M2L + L2L，由頂到底 ---- */

        //FILE *fp_L2L = fopen("L2L_log.txt", "a");
        //FILE *fp_M2L = fopen("M2L_log.txt", "a");
        for (int lv = 1; lv <= tree->max_level; lv++) {
            int grid_lv      = 1 << lv;        // 2^lv
            int grid_parent  = 1 << (lv - 1);  // 2^(lv-1)
            #pragma omp target teams distribute parallel for if(use_gpu)
            for (int idx = 0; idx < count_arr[lv]; idx++) {
                int node_id = base_arr[lv] + idx;
                if (tree->node_start[node_id] > tree->node_end[node_id])
                    continue;  // 空節點跳過

                /* --- L2L: 父 → 自己（父已在上一層處理完畢）--- */
                int parent_idx = idx / 4;
                int parent_id  = base_arr[lv-1] + parent_idx;
                L2L(tree, parent_id, node_id, local_exp, binom);
                /* --- M2L: 找 interaction list --- */
                int nx, ny;
                hilbert_d2xy(grid_lv, (uint64_t)idx, &nx, &ny);

                int px, py;
                hilbert_d2xy(grid_parent, (uint64_t)parent_idx, &px, &py);

                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int npx = px + dx, npy = py + dy;
                        if (npx < 0 || npx >= grid_parent ||
                            npy < 0 || npy >= grid_parent)
                            continue;  // 超出邊界

                        uint64_t nb_parent_hid = hilbert_xy2d(grid_parent, npx, npy);
                        int nb_parent_idx = (int)nb_parent_hid;

                        for (int c = 0; c < 4; c++) {
                            int src_idx = 4 * nb_parent_idx + c;
                            int src_id  = base_arr[lv] + src_idx;

                            if (src_id == node_id) continue;  // 跳過自己
                            if (tree->node_start[src_id] > tree->node_end[src_id])
                                continue;  // 空節點跳過

                            /* 檢查是否為鄰居 */
                            int cx, cy;
                            hilbert_d2xy(grid_lv, (uint64_t)src_idx, &cx, &cy);

                            if (abs(cx - nx) <= 1 && abs(cy - ny) <= 1)
                                continue;  // 是鄰居 → 不屬於 interaction list

                            /* 是 interaction list 成員 → 做 M2L */
                            M2L(tree, node_id,src_id, multipole, local_exp, binom);
                        }
                    }
                }
            }
        }
        //fclose(fp_L2L);
        //fclose(fp_M2L);

        // /*for debug*/
        // #ifdef DEBUG
        // FILE *fp_3 = fopen("local_exp.txt", "a");
        // for (int i = 0; i < tree->total_nodes; i++) {
        //     for (int j = 0; j < MAX_P; j++) {
        //         fprintf(fp_3, "%f ", creal(local_exp[i * MAX_P + j]));
        //         }
        //         fprintf(fp_3, "\n");
        //     }
        // fclose(fp_3);
        // #endif

        /* ---- Final evaluation: leaves only ---- */
        #pragma omp target teams distribute parallel for if(use_gpu)
        for (int i = 0; i < tree->num_leaves; i++)
            L2P(tree, tree->leaf_indices[i], local_exp);

        /*P2P evaluation for non-leaves*/
        /*
        * 平行化策略：每個 thread 只「寫」自己的 leaf_id 粒子，不再使用對稱寫法。
        * 因此呼叫 P2P 時 is_symmetric = false，並且不再以 nb_id > leaf_id 過濾，
        * 改成每個 leaf 都掃完全部 8 個鄰居（每對近場交互被算兩次，但不需 atomic）。
        * P2P_S 只寫自己 leaf 內的粒子，本身就 thread-safe。
        */
        if(use_gpu){
            /* GPU平行（不用schedule(dynamic)）*/
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < tree->num_leaves; i++) {
                int leaf_id = tree->leaf_indices[i];
                int lv  = find_level(leaf_id, base_arr, tree->max_level);
                int idx = leaf_id - base_arr[lv];
                int grid_lv = 1 << lv;
                /* 自交互：leaf 內部粒子彼此作用 */
                P2P_S(tree, leaf_id);
                /* 找同層鄰居 */
                int nx, ny;
                hilbert_d2xy(grid_lv, (uint64_t)idx, &nx, &ny);
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        int nbx = nx + dx, nby = ny + dy;
                        if (nbx < 0 || nbx >= grid_lv || nby < 0 || nby >= grid_lv)
                            continue;
                        int nb_idx = (int)hilbert_xy2d(grid_lv, nbx, nby);
                        int nb_id  = base_arr[lv] + nb_idx;
                        if (tree->node_start[nb_id] > tree->node_end[nb_id])
                            continue;
                        P2P(tree, leaf_id, nb_id, false);
                    }
                }
            }
            #pragma omp target update from(tree->sort_potential[0:n_particles], tree->sort_fx[0:n_particles], tree->sort_fy[0:n_particles])
        }else{
            /* CPU平行*/
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < tree->num_leaves; i++) {
                int leaf_id = tree->leaf_indices[i];
                int lv  = find_level(leaf_id, base_arr, tree->max_level);
                int idx = leaf_id - base_arr[lv];
                int grid_lv = 1 << lv;
                /* 自交互：leaf 內部粒子彼此作用 */
                P2P_S(tree, leaf_id);
                /* 找同層鄰居 */
                int nx, ny;
                hilbert_d2xy(grid_lv, (uint64_t)idx, &nx, &ny);
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        int nbx = nx + dx, nby = ny + dy;
                        if (nbx < 0 || nbx >= grid_lv || nby < 0 || nby >= grid_lv)
                            continue;
                        int nb_idx = (int)hilbert_xy2d(grid_lv, nbx, nby);
                        int nb_id  = base_arr[lv] + nb_idx;
                        if (tree->node_start[nb_id] > tree->node_end[nb_id])
                            continue;
                        P2P(tree, leaf_id, nb_id, false);
                    }
                }
            }
        }
    }
    
    /* 將 tree 內部的計算結果（依 Hilbert 順序排列）寫回原始 particles 陣列 */
    for (int i = 0; i < n; i++) {
        int orig = tree->sort_indices[i];
        particles[orig].potential = tree->sort_potential[i];
        particles[orig].fx        = tree->sort_fx[i];
        particles[orig].fy        = tree->sort_fy[i];
    }

    /* cleanup */
    free(multipole);
    free(local_exp);
    hilbert_tree_free(tree);
}


/* brute-force O(N^2) direct sum for verification */
static void direct_sum(const Particle *particles, int n,
                       double *pot, double *fx, double *fy)
{
    #pragma omp target teams distribute parallel for \
        map(to: particles[0:n]) \
        map(from: pot[0:n], fx[0:n], fy[0:n])
    for (int i = 0; i < n; i++) {
        pot[i] = 0.0;
        fx[i]  = 0.0;
        fy[i]  = 0.0;
        for (int j = i + 1; j < n; j++) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r2 = dx * dx + dy * dy;
            if (r2 < 1e-15) r2 = 1e-15;

            double ln_r2  = 0.5 * log(r2);
            double inv_r2 = 1.0 / r2;

            pot[i] += G_CONST * particles[j].mass * ln_r2;

            double fxij = dx * inv_r2;
            double fyij = dy * inv_r2;
            fx[i] -= G_CONST * particles[j].mass * fxij;
            fy[i] -= G_CONST * particles[j].mass * fyij;
        }
        for (int j = i -1; j >= 0; j--) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r2 = dx * dx + dy * dy;
            if (r2 < 1e-15) r2 = 1e-15;

            double ln_r2  = 0.5 * log(r2);
            double inv_r2 = 1.0 / r2;

            pot[i] += G_CONST * particles[j].mass * ln_r2;

            double fxij = dx * inv_r2;
            double fyij = dy * inv_r2;
            fx[i] -= G_CONST * particles[j].mass * fxij;
            fy[i] -= G_CONST * particles[j].mass * fyij;
        }
    }
}

/* 依 OpenMP 核心數將粒子結果分區寫入 result/起始~結束.txt */
static void write_results_by_cores(const Particle *particles, int n)
{
#ifdef _WIN32
    _mkdir("result");
#else
    mkdir("result", 0755);
#endif

    int num_threads = omp_get_max_threads();
    int chunk_size  = (n + num_threads - 1) / num_threads;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk_size;
        if (start >= n)
            continue;
        int end = start + chunk_size - 1;
        if (end >= n)
            end = n - 1;

        char filename[64];
        snprintf(filename, sizeof(filename), "result/%d~%d.txt", start, end);

        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "無法建立檔案：%s\n", filename);
            continue;
        }

        fprintf(fp, "粒子編號 x y 位能 fx fy\n");
        for (int i = start; i <= end; i++) {
            fprintf(fp, "%d %.6e %.6e %.6e %.6e %.6e\n",
                    i,
                    particles[i].x,
                    particles[i].y,
                    particles[i].potential,
                    particles[i].fx,
                    particles[i].fy);
        }
        fclose(fp);
    }
}

int main(int argc, char *argv[])
{
    int N = 1000;
    int MAX_LEVEL = 10;
    int MAX_PER_LEAF = 10;
    int NUM_SAMPLES = 5;                       /* 對照輸出的粒子數 */
    int NUM_THREADS = omp_get_max_threads();   /* 使用的核心數 */
    unsigned int SEED = 42;                    /* 隨機資料種子 */
    int USE_DIRECT = 1;                        /* 是否跑 direct sum 對照（y/n） */
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) MAX_LEVEL = atoi(argv[2]);
    if (argc > 3) MAX_PER_LEAF = atoi(argv[3]);
    if (argc > 4) NUM_SAMPLES = atoi(argv[4]);
    if (argc > 5) NUM_THREADS = atoi(argv[5]);
    if (argc > 6) SEED = (unsigned int)strtoul(argv[6], NULL, 10);
    if (argc > 7) USE_DIRECT = (argv[7][0] == 'y' || argv[7][0] == 'Y' || argv[7][0] == '1');

    if (NUM_THREADS < 1) NUM_THREADS = 1;
    omp_set_num_threads(NUM_THREADS);
    if (NUM_SAMPLES < 0) NUM_SAMPLES = 0;

    srand(SEED);
    Particle *particles = malloc(N * sizeof(Particle));
    for (int i = 0; i < N; i++) {
        particles[i].x         = (double)rand() / RAND_MAX * 100.0;
        particles[i].y         = (double)rand() / RAND_MAX * 100.0;
        particles[i].mass      = (double)rand() / RAND_MAX + 0.5;
        particles[i].potential = 0.0;
        particles[i].fx        = 0.0;
        particles[i].fy        = 0.0;
    }

    printf("=== FMM benchmark ===\n");
    printf("N            : %d\n", N);
    printf("FMM_ORDER    : %d\n", FMM_ORDER);
    printf("MAX_LEVEL    : %d\n", MAX_LEVEL);
    printf("MAX_PER_LEAF : %d\n", MAX_PER_LEAF);
    printf("NUM_THREADS  : %d\n", NUM_THREADS);
    printf("NUM_SAMPLES  : %d\n", NUM_SAMPLES);
    printf("SEED         : %u\n", SEED);
    printf("USE_DIRECT   : %s\n", USE_DIRECT ? "y" : "n");

    double t0 = omp_get_wtime();
    fmm_solve(particles, N, MAX_LEVEL, MAX_PER_LEAF, use_gpu);
    double t1 = omp_get_wtime();
    double fmm_time = t1 - t0;
    printf("FMM         : %.4f sec\n", fmm_time);

    /* 保存 FMM 結果（potential / fx / fy）以便後續對照 */
    double *fmm_pot = malloc(N * sizeof(double));
    double *fmm_fx  = malloc(N * sizeof(double));
    double *fmm_fy  = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        fmm_pot[i] = particles[i].potential;
        fmm_fx[i]  = particles[i].fx;
        fmm_fy[i]  = particles[i].fy;
    }

    // write_results_by_cores(particles, N);
    // printf("結果已寫入 result/ 資料夾（共 %d 個核心分區）\n",
    //        omp_get_max_threads());

    /* 只在 N 不大時跑 O(N^2) direct sum 作為 reference */
    if (USE_DIRECT && N <= 100000) {
        double *ref_pot = malloc(N * sizeof(double));
        double *ref_fx  = malloc(N * sizeof(double));
        double *ref_fy  = malloc(N * sizeof(double));

        double d0 = omp_get_wtime();
        direct_sum(particles, N, ref_pot, ref_fx, ref_fy);
        double d1 = omp_get_wtime();
        double dir_time = d1 - d0;
        printf("Direct sum  : %.4f sec  (speedup x%.2f)\n",
               dir_time, fmm_time > 0 ? dir_time / fmm_time : 0.0);

        /* 統計 L_inf / L2 RMS / 相對誤差 */
        double pot_max_abs = 0.0, pot_max_ref = 0.0, pot_sse = 0.0, pot_norm2 = 0.0;
        double f_max_abs   = 0.0, f_max_ref   = 0.0, f_sse   = 0.0, f_norm2   = 0.0;

        for (int i = 0; i < N; i++) {
            double dp  = fmm_pot[i] - ref_pot[i];
            double rp  = fabs(ref_pot[i]);
            pot_sse   += dp * dp;
            pot_norm2 += ref_pot[i] * ref_pot[i];
            if (fabs(dp) > pot_max_abs) pot_max_abs = fabs(dp);
            if (rp      > pot_max_ref) pot_max_ref = rp;

            double dfx = fmm_fx[i] - ref_fx[i];
            double dfy = fmm_fy[i] - ref_fy[i];
            double df2 = dfx * dfx + dfy * dfy;
            double rf2 = ref_fx[i] * ref_fx[i] + ref_fy[i] * ref_fy[i];
            f_sse   += df2;
            f_norm2 += rf2;
            if (sqrt(df2) > f_max_abs) f_max_abs = sqrt(df2);
            if (sqrt(rf2) > f_max_ref) f_max_ref = sqrt(rf2);
        }

        double pot_rmse = sqrt(pot_sse  / N);
        double pot_rel  = pot_norm2 > 0 ? sqrt(pot_sse / pot_norm2) : pot_rmse;
        double f_rmse   = sqrt(f_sse / N);
        double f_rel    = f_norm2   > 0 ? sqrt(f_sse / f_norm2)     : f_rmse;

        printf("--- Potential error ---\n");
        printf("  L_inf abs : %.3e   (max |ref| = %.3e)\n", pot_max_abs, pot_max_ref);
        printf("  L_inf rel : %.3e\n", pot_max_ref > 0 ? pot_max_abs / pot_max_ref : pot_max_abs);
        printf("  RMSE      : %.3e\n", pot_rmse);
        printf("  L2 rel    : %.3e\n", pot_rel);

        printf("--- Force error ---\n");
        printf("  L_inf abs : %.3e   (max |ref| = %.3e)\n", f_max_abs, f_max_ref);
        printf("  L_inf rel : %.3e\n", f_max_ref > 0 ? f_max_abs / f_max_ref : f_max_abs);
        printf("  RMSE      : %.3e\n", f_rmse);
        printf("  L2 rel    : %.3e\n", f_rel);

        /* 印出前幾顆粒子對照 */
        int sample = N < NUM_SAMPLES ? N : NUM_SAMPLES;
        printf("--- Sample (first %d particles) ---\n", sample);
        printf("%4s | %12s %12s | %12s %12s\n",
               "i", "FMM pot", "REF pot", "FMM |F|", "REF |F|");
        for (int i = 0; i < sample; i++) {
            double fmm_f = sqrt(fmm_fx[i] * fmm_fx[i] + fmm_fy[i] * fmm_fy[i]);
            double ref_f = sqrt(ref_fx[i] * ref_fx[i] + ref_fy[i] * ref_fy[i]);
            printf("%4d | %12.4e %12.4e | %12.4e %12.4e\n",
                   i, fmm_pot[i], ref_pot[i], fmm_f, ref_f);
        }

        free(ref_pot);
        free(ref_fx);
        free(ref_fy);
    } else if (!USE_DIRECT) {
        printf("(skip direct sum: USE_DIRECT=n)\n");
    } else {
        printf("(skip direct sum: N=%d > 100000)\n", N);
    }

    free(fmm_pot);
    free(fmm_fx);
    free(fmm_fy);
    free(particles);
    return 0;
}

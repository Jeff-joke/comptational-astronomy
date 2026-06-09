#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// =====================================================================
// 1. 參數與結構定義
// =====================================================================

#define P_TERMS 10
#define P_TOTAL ((P_TERMS + 1) * (P_TERMS + 2) / 2)  
#define MAX_PARTICLES 40   // finest box 粒子數上限

typedef struct {
    double real;
    double imag;
} Complex;

typedef struct {
    double x, y, z;
    double charge;
    double potential;
} Particle;

typedef struct Box {
    double cx, cy, cz;      
    double size;            
    int level;              
    int is_leaf;            
    
    int* particle_indices;  
    int num_particles;
    int capacity;
    
    struct Box* parent;     
    struct Box* children[8];
    struct Box* neighbors[27];
    
    Complex multipole[P_TERMS+1][2*P_TERMS+1]; 
    Complex local[P_TERMS+1][2*P_TERMS+1];     
} Box;

// --- 函數原型宣告 ---
Box* create_box(double cx, double cy, double cz, double size, int level, Box* parent);
void subdivide(Box* box);
void build_uniform_tree(Box* box,int MAX_LEVEL);
void insert_particle(Box* box, int p_idx, Particle* particles);
Box* find_box(Box* current, double x, double y, double z, int target_level);
int is_neighbor(Box* a, Box* b);
Box* get_neighbor(Box* box, int neighbor_index, Box* root);

// Do not touch
// --- 複數運算 ---
Complex make_complex(double r, double i) {
    Complex c = {r, i};
    return c;
}

Complex c_add(Complex a, Complex b) {
    return make_complex(a.real + b.real, a.imag + b.imag);
}

Complex c_mul_real(Complex a, double r) {
    return make_complex(a.real * r, a.imag * r);
}

Complex c_mul_c(Complex a, Complex b){
    return(make_complex(a.real * b.real - a.imag * b.imag, a.imag * b.real + a.real * b.imag));
}

Complex c_conj(Complex a){
    return(make_complex(a.real, -a.imag));
}

// =====================================================================
// 2. 空間搜尋與均勻樹管理
// =====================================================================

Box* create_box(double cx, double cy, double cz, double size, int level, Box* parent) {
    Box* box = (Box*)malloc(sizeof(Box));
    box->cx = cx; box->cy = cy; box->cz = cz;
    box->size = size;
    box->level = level;
    box->parent = parent;
    
    box->num_particles = 0;
    box->capacity = 10; // 初始容量
    box->particle_indices = (int*)malloc(sizeof(int) * box->capacity);
    box->is_leaf = 1;
    
    for (int i = 0; i < 8; i++) box->children[i] = NULL;
    for (int i = 0; i < 27; i++) box->neighbors[i] = NULL;
    memset(box->multipole, 0, sizeof(box->multipole));
    memset(box->local, 0, sizeof(box->local));
    return box;
}

void subdivide(Box* box) {
    box->is_leaf = 0;
    // 內部節點不儲存粒子，釋放記憶體節省空間
    free(box->particle_indices);
    box->particle_indices = NULL;
    
    double new_size = box->size / 2.0;
    for (int i = 0; i < 8; i++) {
        double ox = ((i & 1) ? 0.25 : -0.25) * box->size;
        double oy = (((i >> 1) & 1) ? 0.25 : -0.25) * box->size;
        double oz = (((i >> 2) & 1) ? 0.25 : -0.25) * box->size;
        box->children[i] = create_box(box->cx + ox, box->cy + oy, box->cz + oz, new_size, box->level + 1, box);
    }
}

// 一開始就強迫切分到 MAX_LEVEL 的均勻樹構建函數
void build_uniform_tree(Box* box,int MAX_LEVEL) {
    if (box->level >= MAX_LEVEL) return;
    subdivide(box);
    for (int i = 0; i < 8; i++) {
        build_uniform_tree(box->children[i],MAX_LEVEL);
    }
}

void insert_particle(Box* box, int p_idx, Particle* particles) {
    if (box->is_leaf) {
        if (box->num_particles >= box->capacity) {
            box->capacity *= 2;
            box->particle_indices = (int*)realloc(box->particle_indices, sizeof(int) * box->capacity);
        }
        box->particle_indices[box->num_particles++] = p_idx;
    } else {
        int idx = 0;
        if (particles[p_idx].x > box->cx) idx |= 1;
        if (particles[p_idx].y > box->cy) idx |= 2;
        if (particles[p_idx].z > box->cz) idx |= 4;
        insert_particle(box->children[idx], p_idx, particles);
    }
}

Box* find_box(Box* current, double x, double y, double z, int target_level) {
    if (!current || current->level > target_level) return NULL;

    double half = current->size / 2.0;
    if (x < current->cx - half - 1e-9 || x > current->cx + half + 1e-9 ||
        y < current->cy - half - 1e-9 || y > current->cy + half + 1e-9 ||
        z < current->cz - half - 1e-9 || z > current->cz + half + 1e-9) {
        return NULL;
    }

    if (current->level == target_level || current->is_leaf) return current;
    if (current->is_leaf) return NULL;

    int idx = 0;
    if (x > current->cx) idx |= 1;
    if (y > current->cy) idx |= 2;
    if (z > current->cz) idx |= 4;
    return find_box(current->children[idx], x, y, z, target_level);
}

int is_neighbor(Box* a, Box* b) {
    if (!a || !b) return 0;
    double dx = fabs(a->cx - b->cx);
    double dy = fabs(a->cy - b->cy);
    double dz = fabs(a->cz - b->cz);
    // 均勻網格中同一層的盒子 size 必定相同
    double threshold = a->size + 1e-9;
    return (dx < threshold && dy < threshold && dz < threshold);
}

Box* get_neighbor(Box* box, int neighbor_index, Box* root) {
    int dx = (neighbor_index % 3) - 1;
    int dy = ((neighbor_index / 3) % 3) - 1;
    int dz = (neighbor_index / 9) - 1;
    double nx = box->cx + dx * box->size;
    double ny = box->cy + dy * box->size;
    double nz = box->cz + dz * box->size;
    
    Box* neighbor = find_box(root, nx, ny, nz, box->level);
    if (neighbor != NULL && is_neighbor(box, neighbor)) {
        return neighbor;
    }
    return NULL;
}

// convenient functions
void init_const_array(double pAlm[][2*P_TERMS+1], double pNlm[][2*P_TERMS+1]){
    double facto[4*P_TERMS + 1];
    facto[0] = 1;
    for(int i=1;i<4*P_TERMS+1;i++) facto[i] = facto[i-1] * (double)i;

    for(int l=0;l<2*P_TERMS+1;l++){
        for(int m=0;m<l+1;m++){
            pNlm[l][m] = sqrt(facto[l-m] / facto[l+m]);
            pAlm[l][m] = ((l%2) ? -1.0:1.0) / sqrt(facto[l-m] * facto[l+m]); 
        }
    }
    return;
}


void get_sph_num(double x1, double x2, double y1, double y2, double z1, double z2, double pds[3] ,double* pr, double* psintheta, double* pcostheta, Complex* pe_iphi, double L0){
    pds[0] = x1 - x2;
    pds[1] = y1 - y2;
    pds[2] = z1 - z2;
    double dxy = pow(pds[0], 2) + pow(pds[1], 2);
    double r_abs = sqrt(dxy + pds[2] * pds[2]);
    if (r_abs < 1e-12) r_abs = 1e-12;
    *pr = r_abs;

    *pcostheta = pds[2] / r_abs;
    double sqrt_dxy = sqrt(dxy);
    *psintheta = sqrt_dxy / r_abs;
    if (sqrt_dxy < 1e-12){
        pe_iphi->real = 1.0;
        pe_iphi->imag = 0.0;
    }else{
        pe_iphi->real = pds[0] / sqrt_dxy;
        pe_iphi->imag = pds[1] / sqrt_dxy;
    }
    return;
}

void get_P(int Pscale, double sintheta, double costheta, double pPlm[][2*P_TERMS+1]){
    // for L2M    Pscale = 2*m + 1
    // for others Pscale = m + 1
    (pPlm)[0][0] = 1.0;
    for(int m=0;m<Pscale-1;m++){
        (pPlm)[m+1][m+1] = -(2*m + 1) * sintheta * (pPlm)[m][m];
        (pPlm)[m+1][m]   =  (2*m + 1) * costheta * (pPlm)[m][m];
    }
    for(int l=2;l<=Pscale-1;l++){
        for(int m=0;m<l-1;m++){
            (pPlm)[l][m] = (double)(2*l - 1) / (l - m) * costheta * (pPlm)[l-1][m] 
                          - (double)(l + m - 1) / (l - m) * (pPlm)[l-2][m];
        }
    }
    return;
}

void get_Y(int Yscale, Complex e_iphi, double pPlm[][2*P_TERMS+1], Complex pYlm[][2*P_TERMS+1], double pNlm[][2*P_TERMS+1]){
    for(int l=0;l<Yscale;l++){
        Complex exp_now = make_complex(1, 0);
        for(int m=0;m<=l;m++){
            (pYlm)[l][m] = c_mul_real(exp_now, (pNlm)[l][m] * (pPlm)[l][m]);
            exp_now = c_mul_c(exp_now, e_iphi); // exp(i*m*phi)
        }
    }
    return;
}

void collect_level_nodes(Box* box, int target_level, Box** node_array, int* count) {
    if (!box) return;
    if (box->level == target_level) {
        node_array[*count] = box;
        (*count)++;
        return;
    }
    if (!box->is_leaf) {
        for (int i = 0; i < 8; i++) {
            collect_level_nodes(box->children[i], target_level, node_array, count);
        }
    }
}

void precompute_all_neighbors(Box* box, Box* root) {
    if (!box) return;
    for (int n = 0; n < 27; n++) {
        box->neighbors[n] = get_neighbor(box, n, root);
    }
    if (!box->is_leaf) {
        for (int i = 0; i < 8; i++) precompute_all_neighbors(box->children[i], root);
    }
}

// Do not touch
// =====================================================================
// 3. FMM 算子 (Operators)
// =====================================================================

void p2m(Box* box, Particle* particles, double (*pNlm)[2*P_TERMS+1]) {
    if (!box->is_leaf) return;

    for (int i = 0; i < box->num_particles; i++){
        Particle* p = &particles[box->particle_indices[i]];
        double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
        Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];

        get_sph_num(p->x, box->cx, p->y, box->cy, p->z, box->cz, 
                      ds, &r, &sintheta, &costheta, &e_iphi, 1.0);
        get_P(P_TERMS + 1, sintheta, costheta, Plm);
        get_Y(P_TERMS + 1, e_iphi, Plm, Ylm, pNlm);
        double r_now = 1;
        // rnow => r^l
        for(int l=0;l<P_TERMS+1;l++){
            double coeff_indep_m = p->charge * r_now;
            for(int m=-l;m<=l;m++){
                Complex flag;
                if(m>=0){
                    flag = c_conj(Ylm[l][m]);
                }else{
                    flag = Ylm[l][-m];

                }
                box->multipole[l][P_TERMS+m] =c_add(box->multipole[l][P_TERMS+m],c_mul_real(flag, coeff_indep_m));
            }
            r_now *= r;
        }
    }
}

void m2m(Box* parent, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    memset(parent->multipole, 0, sizeof(parent->multipole));
    Complex i_power[4] = {make_complex(1, 0), make_complex(0, 1), make_complex(-1, 0), make_complex(0, -1)};
    for(int i=0;i<8;i++){
        double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
        Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];
        Box *child = parent->children[i];
        if (!child) continue;
        get_sph_num(child->cx, parent->cx, child->cy, parent->cy, child->cz, parent->cz, ds, &r, &sintheta, &costheta, &e_iphi, 1.0);
        get_P(P_TERMS+1, sintheta, costheta, Plm);
        get_Y(P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

        Complex (*Olm)[2*P_TERMS+1] = child->multipole;
        for(int j=0;j<=P_TERMS;j++){
            for(int k=-j;k<=j;k++){
                Complex add_num = make_complex(0, 0);
                double r_now = 1;
                for(int n=0;n<=j;n++){
                    for(int m=-n;m<=n;m++){
                        if(abs(k-m) > j-n) continue;
                        Complex flag;
                        if(m>=0) flag = c_conj(Ylm[n][m]);
                        else flag = Ylm[n][-m];
                        

                        add_num = c_add(add_num,
                                    c_mul_real(
                                        c_mul_c(
                                            c_mul_c(Olm[j-n][P_TERMS+k-m], i_power[(((abs(k)-abs(m)-abs(k-m))%4)+4)%4]),
                                            flag),
                                        pAlm[n][abs(m)] * pAlm[j-n][abs(k-m)] * r_now));
                    }
                    r_now *= r;
                }
                parent->multipole[j][P_TERMS+k] = c_add(parent->multipole[j][P_TERMS+k],
                                                        c_mul_real(add_num, 1.0 / pAlm[j][abs(k)]));
            }
        }
    }
    return;
}

void m2l(Box* target, Box* source, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    double scale = target->size;
    double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
    Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];
    Complex i_power[4] = {make_complex(1, 0), make_complex(0, 1), make_complex(-1, 0), make_complex(0, -1)};
    get_sph_num(source->cx, target->cx, source->cy, target->cy, source->cz, target->cz, ds, &r, &sintheta, &costheta, &e_iphi, 1.0);
    get_P(2*P_TERMS+1, sintheta, costheta, Plm);
    get_Y(2*P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

    Complex (*Olm)[2*P_TERMS+1] = source->multipole;
    for(int j=0;j<=P_TERMS;j++){
        for(int k=-j;k<=j;k++){
            double m1_now = 1, rnow = pow(r, j+1);
            Complex add_num = make_complex(0, 0);
            for(int n=0;n<=P_TERMS;n++){
                for(int m=-n;m<=n;m++){
                    Complex flag;
                    if(m-k<0){
                        flag = make_complex(Ylm[j+n][-m+k].real, -Ylm[j+n][-m+k].imag);
                    }else{
                        flag = Ylm[j+n][m-k];
                    }

                    add_num = c_add(add_num, 
                                c_mul_real(
                                    c_mul_c(
                                        c_mul_c(Olm[n][P_TERMS+m], i_power[((abs(k-m)-abs(k)-abs(m))%4+4)%4]), 
                                                flag), 
                                    pAlm[n][abs(m)] * pAlm[j][abs(k)]/ (m1_now * rnow * pAlm[j+n][abs(m-k)])));
                }
                m1_now *= -1;
                rnow *= r;
            }
            target->local[j][P_TERMS+k] = c_add(target->local[j][P_TERMS+k], add_num);
        }
    }
    return;
}

void l2l(Box* parent, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    if (parent->is_leaf) return;
    Complex i_power[4] = {make_complex(1, 0), make_complex(0, 1), make_complex(-1, 0), make_complex(0, -1)};
    for (int i = 0; i < 8; i++) {
        Box *child = parent->children[i];
        if(!child) continue;
        double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
        Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];

        memset(Plm, 0, sizeof(Plm));
        memset(Ylm, 0, sizeof(Ylm));

        get_sph_num(child->cx, parent->cx, child->cy, parent->cy, child->cz, parent->cz, ds, &r, &sintheta, &costheta, &e_iphi, 1.0);
        get_P(2*P_TERMS+1, sintheta, costheta, Plm);
        get_Y(2*P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

        Complex (*Olm)[2*P_TERMS+1] = parent->local;
        for(int j=0;j<=P_TERMS;j++){
            for(int k=-j;k<=j;k++){
                Complex add_num = make_complex(0, 0);
                for(int n=j;n<=P_TERMS;n++){
                    double rnow = pow(r, n-j);
                    for(int m=-n;m<=n;m++){
                        if(abs(m-k) > n-j) continue;
                        Complex flag;
                        if(m-k>0) flag = Ylm[n-j][m-k];
                        else flag = c_conj(Ylm[n-j][k-m]);
                        add_num = c_add(add_num,
                            c_mul_real(
                                c_mul_c(
                                    c_mul_c(Olm[n][P_TERMS+m], i_power[((abs(m)-abs(m-k)-abs(k))%4+4)%4]),
                                    flag),
                                pAlm[n-j][abs(m-k)] * pAlm[j][abs(k)] * rnow / pAlm[n][abs(m)]));                    
                    }

                }
                child->local[j][P_TERMS+k] = c_add(child->local[j][P_TERMS+k], add_num);
            }
        }
    }
}

void l2p(Box* box, Particle* particles, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    if (!box->is_leaf) return;
    for (int i = 0; i < box->num_particles; i++) {
        Particle* p = &particles[box->particle_indices[i]];
        double ds[3], r, sintheta, costheta, Plm[2*P_TERMS+1][2*P_TERMS+1];
        Complex e_iphi, Ylm[2*P_TERMS+1][2*P_TERMS+1];
        get_sph_num(p->x, box->cx, p->y, box->cy, p->z, box->cz, ds, &r, &sintheta, &costheta, &e_iphi, 1.0);
        get_P(P_TERMS+1, sintheta, costheta, Plm);
        get_Y(P_TERMS+1, e_iphi, Plm, Ylm, pNlm);

        Complex (*Ljk)[2*P_TERMS+1] = box->local, pot = make_complex(0, 0);
        double rnow =1;
        for(int j=0;j<=P_TERMS;j++){
            for(int  k=-j;k<=j;k++){
                Complex flag;
                if(k<0) flag = c_conj(Ylm[j][-k]);
                else flag = Ylm[j][k];
                pot = c_add(pot, 
                            c_mul_c(Ljk[j][P_TERMS+k], 
                                c_mul_real(flag, rnow)));
            }
            rnow *= r;
        }
        p->potential += pot.real;
    }
}

// =====================================================================
// 運算執行
// =====================================================================

void upward_pass(Box* root, Particle* particles, int MAX_LEVEL, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    Box** leaf_nodes = (Box**)malloc(sizeof(Box*) * 524288);
    int leaf_count = 0;
    collect_level_nodes(root, MAX_LEVEL, leaf_nodes, &leaf_count);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < leaf_count; i++) {
        p2m(leaf_nodes[i], particles, pNlm);
    }
    free(leaf_nodes);

    for (int lv = MAX_LEVEL - 1; lv >= 0; lv--) {
        Box** level_nodes = (Box**)malloc(sizeof(Box*) * 524288);
        int node_count = 0;
        collect_level_nodes(root, lv, level_nodes, &node_count);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < node_count; i++) {
            m2m(level_nodes[i], pAlm, pNlm);
        }
        free(level_nodes);
    }
}

void compute_m2l(Box* target, Box* root,
                          double (*pAlm)[2*P_TERMS+1],
                          double (*pNlm)[2*P_TERMS+1]) {
    Box* parent = target->parent;
    if (!parent) return;
    int count = 0;
    for (int i = 0; i < 27; i++) {
        Box* p_neighbor = parent->neighbors[i];
        if (!p_neighbor) continue;
        if (!p_neighbor->is_leaf) {
            for (int j = 0; j < 8; j++) {
                Box* candidate = p_neighbor->children[j];
                if (candidate && !is_neighbor(target, candidate)) {
                    m2l(target, candidate, pAlm, pNlm);
                    count++;
                }
            }
        } else {
            if (!is_neighbor(target, p_neighbor)) {
                m2l(target, p_neighbor, pAlm, pNlm);
                count++;
            }
        }
    }
}


void downward_pass(Box* root, int MAX_LEVEL, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    for (int lv = 0; lv <= MAX_LEVEL; lv++) {
        Box** level_nodes = (Box**)malloc(sizeof(Box*) * 524288);
        int node_count = 0;
        collect_level_nodes(root, lv, level_nodes, &node_count);

        // 所有非根節點都做 M2L（不限葉子）
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < node_count; i++) {
            compute_m2l(level_nodes[i], root, pAlm, pNlm);
        }

        // L2L：把自身的 local 往子節點傳
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < node_count; i++) {
            l2l(level_nodes[i], pAlm, pNlm);
        }

        free(level_nodes);
    }
}

// 均勻網格不再需要遞迴撈取葉子，鄰居一定是跟你一樣大的葉子
void compute_near_field_uniform(Box* target_leaf, Box* neighbor_leaf, Particle* particles) {
    if (!neighbor_leaf || !neighbor_leaf->is_leaf) return;

    for (int i = 0; i < target_leaf->num_particles; i++) {
        int idx1 = target_leaf->particle_indices[i];
        double pot_accum = 0.0;

        for (int j = 0; j < neighbor_leaf->num_particles; j++) {
            int idx2 = neighbor_leaf->particle_indices[j];

            double dx = particles[idx1].x - particles[idx2].x;
            double dy = particles[idx1].y - particles[idx2].y;
            double dz = particles[idx1].z - particles[idx2].z;
            double r = sqrt(dx*dx + dy*dy + dz*dz);

            if (r > 1e-10) {
                pot_accum += particles[idx2].charge / r;
            }
        }
        particles[idx1].potential += pot_accum;
    }
}

// 預處理節點以進行 parallel
void collect_leaves(Box* box, Box** leaf_array, int* count){
    if (!box) return;
    if (box->is_leaf){
        leaf_array[*count] = box;
        (*count)++;
    }else{
        for(int i = 0; i < 8; i++){
            collect_leaves(box->children[i], leaf_array, count);
        }
    }
}

int count_leaves(Box* box){
    if(!box) return 0;

    if(box->is_leaf)
        return 1;

    int total = 0;
    for(int i=0;i<8;i++)
        total += count_leaves(box->children[i]);

    return total;
}

void evaluate(Box* root, Particle* particles, double (*pAlm)[2*P_TERMS+1], double (*pNlm)[2*P_TERMS+1]) {
    Box* leaf_array[262144];
    int leaf_count = 0;
    collect_leaves(root, leaf_array, &leaf_count);

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < leaf_count; k++){
        Box* box = leaf_array[k];

        // (a) 遠場l2p
        l2p(box, particles, pAlm, pNlm);
    
        // (b) 近場（自身與鄰居）
        for (int i = 0; i < box->num_particles; i++) {
            int idx1 = box->particle_indices[i];
            double pot_accum = 0.0;
            // 自身＋自身
            for (int j = 0; j < box->num_particles; j++) {
                if (i == j) continue;
                int idx2 = box->particle_indices[j];
                double dx = particles[idx1].x - particles[idx2].x;
                double dy = particles[idx1].y - particles[idx2].y;
                double dz = particles[idx1].z - particles[idx2].z;
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                if (r > 1e-10) {
                    pot_accum += particles[idx2].charge / r;
                }
            }

            // 與鄰居盒子的粒子對：前面改成單向，所以要把所有方向都算進去
            for (int n = 0; n < 27; n++) {
                if (n == 13) continue;
                Box* neighbor = box->neighbors[n];
                if (!neighbor || !neighbor->is_leaf) continue;

                for (int j = 0; j < neighbor->num_particles; j++){
                    int idx2 = neighbor->particle_indices[j];
                    double dx = particles[idx1].x - particles[idx2].x;
                    double dy = particles[idx1].y - particles[idx2].y;
                    double dz = particles[idx1].z - particles[idx2].z;
                    double r = sqrt(dx*dx + dy*dy + dz*dz);
                    if (r > 1e-10) {
                        pot_accum += particles[idx2].charge / r;
                    }
                }
            }
            particles[idx1].potential += pot_accum;
        }
    }
}

void free_tree(Box* box) {
    if (!box) return;
    if (!box->is_leaf) {
        for (int i = 0; i < 8; i++) free_tree(box->children[i]);
    }
    if (box->particle_indices) free(box->particle_indices);
    free(box);
}

int main(int argc, char* argv[]) {
    int N = 5000;
    int target_cpus = 8;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) target_cpus = atoi(argv[2]);
    omp_set_num_threads(target_cpus);

    int MAX_LEVEL=(int)(log((double)N / MAX_PARTICLES) / log(8.0));
    //if (MAX_LEVEL > 3) MAX_LEVEL = 3;
    //MAX_LEVEL = 3;
    if (MAX_LEVEL < 0) MAX_LEVEL = 0;
    double Nlm[2*P_TERMS+1][2*P_TERMS+1], Alm[2*P_TERMS+1][2*P_TERMS+1]; // some const array needed for P2M and etc...
    Particle* particles = (Particle*)malloc(sizeof(Particle) * N);

    srand(42);
    for (int i = 0; i < N; i++) {
        particles[i].x = (double)rand() / RAND_MAX;
        particles[i].y = (double)rand() / RAND_MAX;
        particles[i].z = (double)rand() / RAND_MAX;
        particles[i].charge = 1.0;
        particles[i].potential = 0.0;
    }
    // build Nlm,Alm
    init_const_array(Alm, Nlm);

    // FMM
    double fmm_start = omp_get_wtime();

    // 建立根節點
    Box* root = create_box(0.5, 0.5, 0.5, 1.0, 0, NULL);
    build_uniform_tree(root,MAX_LEVEL);
    for (int i = 0; i < N; i++) insert_particle(root, i, particles);
    precompute_all_neighbors(root, root);

    upward_pass(root, particles, MAX_LEVEL, Alm, Nlm);
    downward_pass(root, MAX_LEVEL, Alm, Nlm);
    evaluate(root, particles, Alm, Nlm);

    double fmm_end = omp_get_wtime();
    double fmm_time = fmm_end - fmm_start;
    printf("FMM Time        : %f seconds\n", fmm_time);

    double err2 = 0.0;
    double ref2 = 0.0;
    double error = 0.0;

    // Direct sum
    double dir_start = omp_get_wtime();
    int dir_limit = 10000000;
    if (N <= dir_limit){
        #pragma omp parallel for reduction(+:err2, ref2) schedule(guided)
        for (int i = 0; i < N; i++) {
            double direct_pot = 0;
            for (int j = 0; j < N; j++) {
                if (i != j)
                {
                double r = sqrt(pow(particles[i].x-particles[j].x,2)+pow(particles[i].y-particles[j].y,2)+pow(particles[i].z-particles[j].z,2));
                if (r > 1e-10) direct_pot += particles[j].charge / r;
                }
            }
            double diff = particles[i].potential - direct_pot;
            err2 += diff*diff;
            ref2 += direct_pot*direct_pot;
        }
    }
    double dir_end = omp_get_wtime();
    double dir_time = dir_end - dir_start;

    if (N <= dir_limit){
        printf("Direct Sum Time : %f seconds\n", dir_time);
        printf("Speedup         : %.2fx\n", dir_time / fmm_time);
        error = sqrt(err2/ref2);

        printf("L2 Error        : %e\n", error);
    }


    free_tree(root);
    free(particles);
    return 0;
}
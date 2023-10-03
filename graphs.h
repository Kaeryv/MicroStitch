struct rag {
    std::size_t num_components;
    std::size_t *adj_matrix;
    std::size_t *mapping;
    float       *mean_colors;
    float       *distmat;
};

struct IntPair {
    std::size_t key, value;
};

std::size_t maximum_label(std::size_t *labels, std::size_t length) {
    std::size_t maximum = 0;
    for(int i = 0; i < length; i++) maximum = maximum < labels[i] ? labels[i]: maximum;
    return maximum;
}

float * uint8_to_float(uint8_t * img, std::size_t num_pixels, uint8_t channels_in, uint8_t channels_out) {
    float *buffer = (float*) malloc(num_pixels*channels_out*sizeof(float));
    for (int i = 0; i < num_pixels; i ++){
        for (int c = 0; c < channels_out; c ++)
            buffer[i * channels_out + c] = img[i * channels_in + c] / 255.f;
    }
    return buffer;
}

void
relabel_sequential(std::size_t* labels, std::size_t length, std::size_t offset = 0) {
    IntPair * mapping = nullptr;
    std::size_t current_index = 0;
    for (int i = 0; i < length; i++) {
        std::size_t cur = labels[i];
        int loc = hmgeti(mapping, cur);
        if (loc >= 0) {
            labels[i] = offset + hmget(mapping, cur);
        } else {
            hmput(mapping, cur, current_index++);
            labels[i] = offset + hmget(mapping, cur);
        }
    }
}
void
relabel_sequential_global(std::size_t** labels, std::size_t length, std::size_t offset=0) {
    IntPair * mapping = nullptr;
    std::size_t current_index = 0;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < length; i++) {
            std::size_t cur = labels[j][i];
            int loc = hmgeti(mapping, cur);
            if (loc >= 0) {
                labels[j][i] = offset + hmget(mapping, cur);
            } else {
                hmput(mapping, cur, current_index++);
                labels[j][i] = offset + hmget(mapping, cur);
            }
        }
    }
}

rag
rag_create(std::size_t n) {
    rag r;
    r.num_components = n;
    r.adj_matrix  = (std::size_t*) calloc(n * n, sizeof(std::size_t));
    r.mapping     = (std::size_t*) calloc(n    , sizeof(std::size_t));
    for (int i = 0; i < n; i++) r.mapping[i] = i;
    r.mean_colors = (float*)       calloc(n * 4, sizeof(float));
    r.distmat     = (float*)       calloc(n * n, sizeof(float));

    assert(r.distmat);
    assert(r.mean_colors);
    assert(r.mapping);
    assert(r.adj_matrix);
    return r;
}

void
rag_free(rag r) {
    free(r.adj_matrix);
    free(r.mean_colors);
    free(r.distmat);
    free(r.mapping);
}

void 
rag_adjacency_matrix(rag r, std::size_t *labels_mat, std::size_t nx, std::size_t ny) {
    #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
    std::size_t src, dst;
    std::size_t N = r.num_components;
    #define adjmat(x, y) (r.adj_matrix[(x) * N + (y)])
    #define labels(x, y) (labels_mat[(y) * nx + (x)])
    for_interval(i, 1, nx-1) {
        for (int j=1; j<ny-1; j++) {
            src = labels(i, j);
            dst = labels(i-1, j);
            adjmat(src, dst) += 1;
            adjmat(dst, src) += 1;
            dst = labels(i+1, j);
            adjmat(src, dst) += 1;
            adjmat(dst, src) += 1;
            dst = labels(i, j+1);
            adjmat(src, dst) += 1;
            adjmat(dst, src) += 1;
            dst = labels(i, j-1);
            adjmat(src, dst) += 1;
            adjmat(dst, src) += 1;
        }
    }
    #undef adjmat
    #undef labels
}

void
rag_color_distance_matrix(rag r, float *picture, std::size_t *labels, std::size_t nx, std::size_t ny)
{
    std::size_t N = r.num_components;
    printf("Compute segments' mean color\n");
    float * mean_colors = r.mean_colors;
    float * distmat = r.distmat;
    #pragma omp parallel for shared(mean_colors,labels,picture) firstprivate(N)
    for (int i=0; i< N; i++) {
        for(int k = 0; k < nx*ny; k++) {
            if (labels[k] == i) {
                mean_colors[i*4+0] += picture[k*3+0]; // r
                mean_colors[i*4+1] += picture[k*3+1]; // g
                mean_colors[i*4+2] += picture[k*3+2]; // b
                mean_colors[i*4+3] += 1;              // area
            }
        }
        float area = mean_colors[i*4+3];
        if (area > 0) {
            mean_colors[i*4+0] /= area; // r
            mean_colors[i*4+1] /= area; // g
            mean_colors[i*4+2] /= area; // b
        }
    }
    printf("Create distances matrix.\n");
    #pragma omp parallel for shared(mean_colors, distmat) firstprivate(N)
    for (int i=0; i< N; i++) {
        for (int j=0; j< N; j++){
            if (r.adj_matrix[i*N+j] > 0) {
                distmat[i*N+j] = sqrtf(
                    + powf(r.mean_colors[i*4+0]-r.mean_colors[j*4+0],2)
                    + powf(r.mean_colors[i*4+1]-r.mean_colors[j*4+1],2)
                    + powf(r.mean_colors[i*4+2]-r.mean_colors[j*4+2],2));
            }
        }
    }
}
#define for_range(X, MAX) for (int X=0; X < (MAX); X++)
void
rag_merge(rag r, float distance_threshold) {
    #define area(x) r.mean_colors[(x)*4+3]
    #define adj(x,y) r.adj_matrix[(x)*N+(y)]
    std::size_t N = r.num_components;
    for_range(i, N) {
        for_range(j, N) {
            if (
                adj(i,j) > 0 
             && i != j
             && r.distmat[i*N+j] < distance_threshold
             ) {
                // Then j eats i
                r.mapping[i] = j;
                // All mappings from k to i are now to j
                for_range(k, N) {
                    if (r.mapping[k] == i) r.mapping[k] = j;
                }
                for (int k = 0; k < N; k++){
                    adj(j, k) = adj(j, k) + adj(i, k);
                    adj(k, j) = adj(k, j) + adj(k, i);
                    adj(i, k) = 0;
                    adj(k, i) = 0;
                }
                for (int k = 0; k < 3; k++){
                    r.mean_colors[j*4+k] = (area(i) * r.mean_colors[i*4+k] +  area(j) * r.mean_colors[j*4+k]);
                    r.mean_colors[j*4+k] /= (area(i) + area(j));
                }
                area(j) += area(i);
                for (int k = 0; k < N; k++){
                    if (adj(j,k) > 0) {
                        r.distmat[j*N+k] = sqrtf(
                            + powf(r.mean_colors[k*4+0]-r.mean_colors[j*4+0],2)
                            + powf(r.mean_colors[k*4+1]-r.mean_colors[j*4+1],2)
                            + powf(r.mean_colors[k*4+2]-r.mean_colors[j*4+2],2));
                        r.distmat[k*N+j] = r.distmat[j*N+k];
                    }
                }
            }
        }
    }
    #undef area
    #undef adj
}

void
rag_relabel(rag r, std::size_t *labels, std::size_t num_pixels) {
    for(int i = 0; i < num_pixels; i++) {
        labels[i] = r.mapping[labels[i]];
    }
}

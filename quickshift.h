#include <raylib.h>

void
quickshift(Image image, int kernel_size, int max_dist, std::size_t *parent, float ratio, int random_seed=42) {
/*
    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image.
    kernel_size : float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool
        Whether to return the full segmentation hierarchy tree and distances.
    random_seed : {None, int, `numpy.random.Generator`}, optional
        If `random_seed` is None the `numpy.random.Generator` singleton
        is used.
        If `random_seed` is an int, a new ``Generator`` instance is used,
        seeded with `random_seed`.
        If `random_seed` is already a ``Generator`` instance then that instance
        is used.

        Random seed used for breaking ties.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
*/
    double start = GetTime();
    float inv_kernel_size_sqr = -0.5 / (float) (kernel_size * kernel_size);
    int kernel_width = (int) ceil(3 * kernel_size);

    std::size_t width    = image.width;
    std::size_t height   = image.height;
    std::size_t channels = 3;

    float *densities = (float*)malloc(width*height*sizeof(float));
    for(int i = 0; i < width*height; i++) {
        float u1 = ((float) rand() / (float) RAND_MAX);
        float u2 = ((float) rand() / (float) RAND_MAX);
        densities[i] = 0.00001f * sqrtf(-2.f  * logf(u1)) * cosf(2.f*M_PI*u2);
    }

    float current_density, closest;
    std::size_t r_, c_;

    float * buffer = (float*)calloc(image.height*image.width*3, sizeof(float));
    float * dist_parent = (float*)malloc(width*height*sizeof(float));
    int counter = 0;
    for (int i = 0; i < height; i ++){
        for (int j = 0; j < width; j ++) {
            parent[i * width  + j] = counter++;
            dist_parent[i * width + j] = 0.f;
            buffer[i * width*channels + j*channels + 0] = ((u_char*)image.data)[i * width * 4 + j*4 + 0] / 255.f;
            buffer[i * width*channels + j*channels + 1] = ((u_char*)image.data)[i * width * 4 + j*4 + 1] / 255.f;
            buffer[i * width*channels + j*channels + 2] = ((u_char*)image.data)[i * width * 4 + j*4 + 2] / 255.f;
        }
    }

    for (int i = 0; i < height * width; i++) {
        float r = buffer[i * channels + 0];
        float g = buffer[i * channels + 1];
        float b = buffer[i * channels + 2];

        // RGB 2 XYZ
        r = r > 0.04045 ? powf((r + 0.055f) / 1.055f, 2.4f): r / 12.92f;
        g = g > 0.04045 ? powf((g + 0.055f) / 1.055f, 2.4f): g / 12.92f;
        b = b > 0.04045 ? powf((b + 0.055f) / 1.055f, 2.4f): b / 12.92f;
        float x = 100.f * (r * 0.4124f + g * 0.3576f + b *  0.1805f);
        float y = 100.f * (r * 0.2126f + g * 0.7152f + b *  0.0722f);
        float z = 100.f * (r * 0.0193f + g * 0.1192f + b *  0.9505f);
        // XYZ 2 L-ab
        x /= 95.047f; 
        y /= 100.000f; 
        z /= 108.883f;
        x = x > 0.008856 ? cbrtf(x): (7.787f * x ) + ( 16.f / 116.f );
        y = y > 0.008856 ? cbrtf(y): (7.787f * y ) + ( 16.f / 116.f );
        z = z > 0.008856 ? cbrtf(z): (7.787f * z ) + ( 16.f / 116.f );

        float L = (116.f * y ) - 16.f;
        float a = 500.f * ( x - y );
        float bb = 200 * ( y - z );
        buffer[i * channels  + 0] = ratio * L;
        buffer[i * channels  + 1] = ratio * a;
        buffer[i * channels  + 2] = ratio * bb;
    }

    printf("Initial distances\n");
    #pragma omp parallel for private(c_,r_) shared(densities,buffer)
    for(int r = 0; r < height; r++) {
        std:: size_t r_min = fmax(r - kernel_width, 0);
        std:: size_t r_max = fmin(r + kernel_width + 1, height);
        for(int c = 0; c < width; c++) {
            std:: size_t c_min = fmax(c - kernel_width, 0);
            std:: size_t c_max = fmin(c + kernel_width + 1, width);
            float *current_pixel_ptr = buffer + width * r * channels + c * channels;
            for(int r_ = r_min; r_ < r_max; r_++) {
                for(int c_ = c_min; c_ < c_max; c_++) {
                    float dist = 0.f;
                    float t    = 0.f;
                    for(int channel = 0; channel < 3; channel++) {
                        t = (current_pixel_ptr[channel] - buffer[r_ * width*3 + c_*3 + channel]);
                        dist += t*t;
                    }
                    t = r-r_;
                    dist += t*t;
                    t = c-c_;
                    dist += t*t;
                    densities[r * width + c] += expf(dist * inv_kernel_size_sqr);
                }
            }
        }
    }

    printf("Medoid shift\n");
    #pragma omp parallel for private(c_,r_,current_density,closest) shared(densities,buffer,parent,dist_parent)
    for(int r = 0; r < height; r++) {
        std:: size_t r_min = fmax(r - kernel_width, 0);
        std:: size_t r_max = fmin(r + kernel_width + 1, height);
        for(int c = 0; c < width; c++) {
            current_density = densities[r*width+c];
            closest = 1e10;
            std:: size_t c_min = fmax(c - kernel_width, 0);
            std:: size_t c_max = fmin(c + kernel_width + 1, width);
            float *current_pixel_ptr = buffer + width * r * channels + c * channels;
            for(int r_ = r_min; r_ < r_max; r_++) {
                for(int c_ = c_min; c_ < c_max; c_++) {
                    if (densities[r_ * width + c_] > current_density) {
                        float dist = 0;
                        float t    = 0.f;
                        for(int channel = 0; channel < 3; channel++) {
                            t = (current_pixel_ptr[channel] - buffer[r_ * width*3 + c_*3 + channel]);
                            dist += t*t;
                        }
                        t = r-r_;
                        dist += t*t;
                        t = c-c_;
                        dist += t*t;
                        if (dist < closest) {
                            closest = dist;
                            parent[r * width + c] = r_ * width + c_;
                        }
                    }
                }
            }
            dist_parent[r*width+c] = sqrtf(closest);
        }
    }

    printf("Max Dist filter\n");
    // remove parents with distance > max_dist
    for(int i = 0; i < width*height; i++) {
        if (dist_parent[i] > (float)max_dist) {
            parent[i] = i;
        }
    }

    printf("Flatten Tree\n");
    bool changed = true;
    while (changed) {
        changed = false;
        for (int j = 0; j < width*height; j++) {
            std::size_t old = parent[j];
            parent[j] = parent[old];
            changed |= (parent[j] != old);
        }
    }
    double stop = GetTime();
    printf("Quishift took %lf\n s.", stop-start);

}
#include "raylib.h"
void
DrawImageOnImage(Image& dst, Image src, Rectangle r) {
    if (dst.width < r.width) return;
    if (dst.height < r.height) return;
    if (r.x < 0) return;
    if (r.y < 0) return;

    #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
    for_interval(i, r.x, r.x+r.width) {
        for_interval(j, r.y, r.y+r.height){
            int i_ = i - r.x;
            int j_ = j - r.y;
            ((Color*)(dst.data))[j*dst.width+i] = ((Color*)src.data)[j_*(int)round(r.width)+i_];
        }
    }
    #undef for_interval
}

typedef struct SegmentProperties {
    std::size_t id;
    Vector2 centroid;
    Rectangle bbox;
    float area;
    bool selected = false;
    Texture2D blob_tex;
    Image blob;
    bool operator <(const SegmentProperties& pt) const
    {
        return id < pt.id;
    }
};
uint8_t *
ImageMaskFromImageL(std::size_t* src, Rectangle r, std::size_t nx, std::size_t ny, std::size_t label) {
    if (nx < r.width) return nullptr;
    if (ny < r.height) return nullptr;
    if (r.x < 0) return nullptr;
    if (r.y < 0) return nullptr;
    uint8_t *dst = (uint8_t*) malloc(r.width*r.height*sizeof(uint8_t));

    #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
    for_interval(i, r.x, r.x+r.width) {
        for_interval(j, r.y, r.y+r.height){
            int i_ = i - r.x;
            int j_ = j - r.y;
            dst[j_*(int)round(r.width)+i_] = 255 * (int) (src[j*nx+i] == label);
        }
    }
    #undef for_interval
    return dst;
}
std::size_t *
ImageFromImageL(std::size_t* src, Rectangle r, std::size_t nx, std::size_t ny) {
    if (nx < r.width) return nullptr;
    if (ny < r.height) return nullptr;
    if (r.x < 0) return nullptr;
    if (r.y < 0) return nullptr;
    std::size_t *dst = (std::size_t*) malloc(r.width*r.height*sizeof(std::size_t));

    #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
    for_interval(i, r.x, r.x+r.width) {
        for_interval(j, r.y, r.y+r.height){
            int i_ = i - r.x;
            int j_ = j - r.y;
            dst[j_*(int)round(r.width)+i_] = src[j*nx+i];
        }
    }
    #undef for_interval
    return dst;
}
SegmentProperties
ComputeSegmentProperties(std::size_t *labels, std::size_t nx, std::size_t ny, std::size_t label) {
    std::size_t first_seen_x = 10000000;
    std::size_t first_seen_y = 10000000;
    std::size_t last_seen_x  = 0;
    std::size_t last_seen_y  = 0;
    double centroid_x  = 0;
    double centroid_y  = 0;
    float area;

    for_range(i, nx) {
        for_range(j, ny) {
            if (labels[j*nx+i] == label) {
                centroid_x += i;
                centroid_y += j;
                area += 1;
                first_seen_x = first_seen_x > i ? i : first_seen_x;
                last_seen_x = last_seen_x >  i ? last_seen_x : i;
                first_seen_y = first_seen_y > j ? j : first_seen_y;
                last_seen_y = last_seen_y > j ? last_seen_y : j;
            }
        }
    }
    centroid_x /= (double) area;
    centroid_y /= (double) area;
    SegmentProperties seg;
    seg.id = label;
    seg.bbox = (Rectangle) {(float) first_seen_x, (float)first_seen_y, (float)(last_seen_x-first_seen_x), (float)(last_seen_y-first_seen_y)};
    seg.centroid.x = (float) centroid_x;
    seg.centroid.y = (float) centroid_y;
    seg.area = area;
    seg.blob = (Image) {
        .data = ImageMaskFromImageL(labels, seg.bbox, nx, ny, label),
        .format  = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
        .width   = (int) seg.bbox.width,
        .height  = (int) seg.bbox.height,
        .mipmaps = 1,
    };
    ImageFormat(&seg.blob, PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA);
    ImageColorReplace(&seg.blob, (Color) {0,0,0,255}, (Color) {0,0,0,0});
    ImageColorReplace(&seg.blob, (Color) {255,255,255,255}, (Color) {255,255,255,128});
    seg.blob_tex = LoadTextureFromImage(seg.blob);

    return seg;
}

void
DrawImageOnImageL(std::size_t* dst, std::size_t* src, Rectangle r, std::size_t nx, std::size_t ny, std::size_t offset) {
    if (nx < r.width) return;
    if (ny < r.height) return;
    if (r.x < 0) return;
    if (r.y < 0) return;

    #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
    for_interval(i, r.x, r.x+r.width) {
        for_interval(j, r.y, r.y+r.height){
            int i_ = i - r.x;
            int j_ = j - r.y;
            dst[j*nx+i] = offset + src[j_*(int)round(r.width)+i_];
        }
    }
    #undef for_interval
}


void
opencv_nlmeans_denoising(Image input, Image& output, int strenght, int kernel_size, int search_window, int channels=4) {
    //assert(IsImageReady(input ));
    //assert(IsImageReady(output));
    cv::Mat cv_input = cv::Mat(input.height, input.width, CV_8UC4, (unsigned*) input.data);
    cv::Mat cv_output = cv_input.clone();
    cv::fastNlMeansDenoisingColored(cv_input, cv_output, strenght, 3, kernel_size, search_window);
    memcpy(output.data, cv_output.data, input.width*input.height*channels*sizeof(unsigned char));
}

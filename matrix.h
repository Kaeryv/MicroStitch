struct matrix_header {
    std::size_t width;
    std::size_t height;
    void * data;
};

void
matrix_zeros()
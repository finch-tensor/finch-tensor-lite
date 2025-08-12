#include <stddef.h>
typedef void* (*fptr)( void**, size_t );
struct CNumpyBuffer {
    void* arr;
    void* data;
    size_t length;
    fptr resize;
};
#include <unistd.h>
double dot_product(struct CNumpyBuffer* a, struct CNumpyBuffer* b) {
    double c = (double)0.0;
    struct CNumpyBuffer* a_ = a;
    double* a__data = (double*)a_->data;
    size_t a__length = a_->length;
    struct CNumpyBuffer* b_ = b;
    double* b__data = (double*)b_->data;
    size_t b__length = b_->length;
    for (ssize_t i = (ssize_t)0; i < a__length; i++) {
        c = c + (a__data)[i] * (b__data)[i];
    }
    a_->data = (void*)a__data;
    a_->length = a__length;
    b_->data = (void*)b__data;
    b_->length = b__length;
    return c;
}
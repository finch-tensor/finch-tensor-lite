#include <stdint.h>
typedef void* (*fptr)( void*, uint64_t );
struct CMallocBuffer {
    void* data;
    uint64_t length;
    uint64_t datasize;
    fptr resize;
};
#include <stddef.h>
double dot_product(struct CMallocBuffer* a, struct CMallocBuffer* b) {
    double c = (double)0.0;
    struct CMallocBuffer* a_ = a;
    double* a__data = (double*)a_->data;
    size_t a__length = a_->length;
    struct CMallocBuffer* b_ = b;
    double* b__data = (double*)b_->data;
    size_t b__length = b_->length;
    for (int64_t i = (int64_t)0; i < a__length; i++) {
        c = c + (a__data)[i] * (b__data)[i];
    }
    a_->data = (void*)a__data;
    a_->length = a__length;
    b_->data = (void*)b__data;
    b_->length = b__length;
    return c;
}
#include <stdint.h>
struct CMallocBufferStruct {
    void* data;
    uint64_t length;
};
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static inline double*
mallocbuffer_resize(double* data, size_t len_old, size_t len_new) {
    data = realloc(data, sizeof(double) * len_new);
    if (data == 0) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(1);
    }
    if (len_new > len_old) {
        memset(&data[len_old], 0, (len_new - len_old) * sizeof(double));
    }
    return data;
}
// methods below are not used by the kernel.
static inline void
mallocbuffer_free(struct CMallocBufferStruct *m) {
    free(m->data);
    m->data = 0;
    m->length = 0;
}
static inline void
mallocbuffer_init(
    struct CMallocBufferStruct *m,
    size_t datasize,
    size_t length
) {
    m->length = length;
    m->data = malloc(length * datasize);
    memset(m->data, 0, length * datasize);
}

double dot_product(struct CMallocBufferStruct* a, struct CMallocBufferStruct* b) {
    double c = (double)0.0;
    struct CMallocBufferStruct* a_ = a;
    double* a__data = (double*)a_->data;
    size_t a__length = a_->length;
    struct CMallocBufferStruct* b_ = b;
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
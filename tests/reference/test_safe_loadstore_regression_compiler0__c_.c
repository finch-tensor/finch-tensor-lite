#include <stddef.h>
typedef void* (*fptr)( void**, size_t );
struct CNumpyBuffer {
    void* arr;
    void* data;
    size_t length;
    fptr resize;
};
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
ssize_t finch_access(struct CNumpyBuffer* a, size_t idx) {
    struct CNumpyBuffer* a_ = a;
    ssize_t* a__data = (ssize_t*)a_->data;
    size_t a__length = a_->length;
    size_t computed = (idx);
    if (computed < 0 || computed >= (a__length)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    ssize_t val = (a__data)[computed];
    size_t computed_2 = (idx);
    if (computed_2 < 0 || computed_2 >= (a__length)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    ssize_t val2 = (a__data)[computed_2];
    return val;
}
ssize_t finch_change(struct CNumpyBuffer* a, size_t idx, ssize_t val) {
    struct CNumpyBuffer* a_ = a;
    ssize_t* a__data_2 = (ssize_t*)a_->data;
    size_t a__length_2 = a_->length;
    size_t computed_3 = (idx);
    if (computed_3 < 0 || computed_3 >= (a__length_2)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    (a__data_2)[computed_3] = val;
    return (ssize_t)0;
}
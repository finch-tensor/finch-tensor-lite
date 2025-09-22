#include <stdint.h>
typedef void* (*fptr)( void**, uint64_t );
struct CNumpyBuffer {
    void* arr;
    void* data;
    uint64_t length;
    fptr resize;
};
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
int64_t finch_access(struct CNumpyBuffer* a, uint64_t idx) {
    struct CNumpyBuffer* a_ = a;
    int64_t* a__data = (int64_t*)a_->data;
    size_t a__length = a_->length;
    size_t computed = (idx);
    if (computed < 0 || computed >= (a__length)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    int64_t val = (a__data)[computed];
    size_t computed_2 = (idx);
    if (computed_2 < 0 || computed_2 >= (a__length)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    int64_t val2 = (a__data)[computed_2];
    return val;
}
int64_t finch_change(struct CNumpyBuffer* a, uint64_t idx, int64_t val) {
    struct CNumpyBuffer* a_ = a;
    int64_t* a__data_2 = (int64_t*)a_->data;
    size_t a__length_2 = a_->length;
    size_t computed_3 = (idx);
    if (computed_3 < 0 || computed_3 >= (a__length_2)) {
        fprintf(stderr, "Index out of bounds error!");
        exit(1);
    }
    (a__data_2)[computed_3] = val;
    return (int64_t)0;
}
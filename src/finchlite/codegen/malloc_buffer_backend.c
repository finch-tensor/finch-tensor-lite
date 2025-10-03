#include <stdlib.h>
#include <string.h>

struct MallocBuffer;

struct MallocBuffer {
  void *data;
  size_t length;
  size_t datasize;
  void (*resize)(struct MallocBuffer *, size_t sz);
};

void mallocbuffer_resize(struct MallocBuffer *m, size_t length) {
  m->data = realloc(m->data, m->datasize * length);
  if (length > m->length) {
    memset(m->data + (m->length * m->datasize), 0,
           (length - m->length) * m->datasize);
  }
  m->length = length;
}

void mallocbuffer_free(struct MallocBuffer *m) {
  free(m->data);
  m->data = 0;
  m->length = 0;
}

void mallocbuffer_init(struct MallocBuffer *m, size_t datasize, size_t length) {
  m->length = length;
  m->datasize = datasize;
  m->data = malloc(length * datasize);
  memset(m->data, 0, length * datasize);
  m->resize = mallocbuffer_resize;
}

#include <stdlib.h>
#include <string.h>

void engine_forward(void *ctx, float *input, float *output, int size) {
    // Stub implementation - just copy input to output
    if (input && output && size > 0) {
        memcpy(output, input, size * sizeof(float));
    }
}

//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/memory.h"
namespace lox {
namespace vm {
void *reallocate(void *buffer, int old_size, int new_size) {
  if (new_size == 0) {
    free(buffer);
    return nullptr;
  }

  auto *result = realloc(buffer, new_size);
  if (!result) exit(1);
  return result;
}
}  // namespace vm
}  // namespace lox

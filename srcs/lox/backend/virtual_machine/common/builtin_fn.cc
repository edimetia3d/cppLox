//
// LICENSE: MIT
//

#include "builtin_fn.h"

#include <time.h>

namespace lox {
namespace vm {

Object clockNative(int argCount, Object *args) {
  assert(argCount == 0);
  return Object((double)clock() / CLOCKS_PER_SEC);
}

}  // namespace vm
}  // namespace lox

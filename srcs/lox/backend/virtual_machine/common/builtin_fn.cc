//
// LICENSE: MIT
//

#include "builtin_fn.h"

#include <time.h>

namespace lox {
namespace vm {

Value clockNative(int argCount, Value *args) {
  assert(argCount == 0);
  return Value((double)clock() / CLOCKS_PER_SEC); }

}  // namespace vm
}  // namespace lox

//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/builtins/builtin_fn.h"

#include <ctime>

namespace lox::vm {
namespace {
Value Clock(int arg_count, Value *args) {
  assert(arg_count == 0);
  return Value((double)clock() / CLOCKS_PER_SEC);
}
} // namespace

const std::map<std::string, NativeFn> &AllNativeFn() {
  static std::map<std::string, NativeFn> all_native_fn = {
      {"clock", Clock},
  };
  return all_native_fn;
}

} // namespace lox::vm

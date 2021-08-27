//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/clox_value.h"
namespace lox {
namespace vm {

Value printValue(const Value &value) { printf("%f", value); }
}  // namespace vm
}  // namespace lox

//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_
#include <stdio.h>
namespace lox {
namespace vm {

using Value = double;

Value printValue(const Value &value);
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_

//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/frontend/scanner.h"
namespace lox {

namespace vm {
class Compiler {
 public:
  void Compile(const Scanner &scanner, Chunk *target);
};
}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

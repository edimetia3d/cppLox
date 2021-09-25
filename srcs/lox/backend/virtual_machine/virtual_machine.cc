
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"

namespace lox {
namespace vm {

LoxError VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  CompileUnit cu;
  ErrCode err_code = ErrCode::NO_ERROR;
  if (!compiler.Compile(&scanner, &cu)) {
    return LoxError("Compiler Error: " + std::to_string(static_cast<int>(err_code)));
  }
  auto vm = VM::Instance();
  if ((err_code = vm->Interpret(cu.entry_point->chunk)) != ErrCode::NO_ERROR) {
    return LoxError("Runtime Error: " + std::to_string(static_cast<int>(err_code)));
  }
  return LoxError();
}
}  // namespace vm
}  // namespace lox

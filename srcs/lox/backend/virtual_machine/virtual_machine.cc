
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"

namespace lox {
namespace vm {

LoxError VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  Chunk chunk;
  LexicalScope scope;
  ErrCode err_code = ErrCode::NO_ERROR;
  if ((err_code = compiler.Compile(&scanner, &chunk, &scope)) != ErrCode::NO_ERROR) {
    return LoxError("Compiler Error: " + std::to_string(static_cast<int>(err_code)));
  }
  auto vm = VM::Instance();
  if ((err_code = vm->Interpret(&chunk)) != ErrCode::NO_ERROR) {
    return LoxError("Runtime Error: " + std::to_string(static_cast<int>(err_code)));
  }
  return LoxError();
}
}  // namespace vm
}  // namespace lox

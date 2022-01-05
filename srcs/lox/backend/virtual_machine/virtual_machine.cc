
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/core/chunk.h"
#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/err_code.h"

namespace lox::vm {

LoxError VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  ErrCode err_code = ErrCode::NO_ERROR;
  ObjFunction* script = compiler.Compile(&scanner);
  if (!script) {
    return LoxError("Compiler Error: " + std::to_string(static_cast<int>(err_code)));
  }
  if ((err_code = VM::Instance()->Interpret(script)) != ErrCode::NO_ERROR) {
    return LoxError("Runtime Error: " + std::to_string(static_cast<int>(err_code)));
  }
  return LoxError();
}
}  // namespace lox

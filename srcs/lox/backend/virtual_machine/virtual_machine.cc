
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/backend/virtual_machine/errors.h"

namespace lox::vm {

void VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  std::string err_msg;
  ObjFunction* script = compiler.Compile(&scanner, &err_msg);
  if (!script) {
    throw CompilationError(err_msg, EX_DATAERR);
  }
  VM::Instance()->Interpret(script);
}
}  // namespace lox

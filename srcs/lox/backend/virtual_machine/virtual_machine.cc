
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"

namespace lox {
namespace vm {
LoxError VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  Chunk chunk;
  compiler.Compile(scanner, &chunk);

  auto offset = chunk.addConstant(0.9);
  chunk.WriteOpCode(OpCode::OP_CONSTANT, 0);
  chunk.WriteUInt8(static_cast<uint8_t>(offset), 0);

  offset = chunk.addConstant(3.4);
  chunk.WriteOpCode(OpCode::OP_CONSTANT, 0);
  chunk.WriteUInt8(offset, 0);
  chunk.WriteOpCode(OpCode::OP_ADD, 0);

  offset = chunk.addConstant(5.6);
  chunk.WriteOpCode(OpCode::OP_CONSTANT, 0);
  chunk.WriteUInt8(offset, 0);
  chunk.WriteOpCode(OpCode::OP_DIVIDE, 0);

  chunk.WriteOpCode(OpCode::OP_NEGATE, 0);
  chunk.WriteOpCode(OpCode::OP_RETURN, 0);

  auto vm = VM::Instance();
  vm->Interpret(&chunk);
  return LoxError();
}
}  // namespace vm
}  // namespace lox

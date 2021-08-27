
#include "lox/backend/virtual_machine/virtual_machine.h"

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/core/compiler.h"
#include "lox/backend/virtual_machine/core/vm.h"

namespace lox {
namespace vm {
void FillDbgChunk(Chunk& chunk) {
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
}

LoxError VirtualMachine::Run(Scanner& scanner) {
  Compiler compiler;
  Chunk chunk;
  ErrCode err_code = ErrCode::NO_ERROR;
  if ((err_code = compiler.Compile(scanner, &chunk)) != ErrCode::NO_ERROR) {
    return LoxError("Compiler Error:: " + std::to_string(static_cast<int>(err_code)));
  }
  chunk.WriteOpCode(OpCode::OP_RETURN, 0);  // avoid endless run
  auto vm = VM::Instance();
  vm->Interpret(&chunk);
  return LoxError();
}
}  // namespace vm
}  // namespace lox

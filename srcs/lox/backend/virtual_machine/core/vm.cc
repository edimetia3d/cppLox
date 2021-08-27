//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"

#include "lox/backend/virtual_machine/core/debug.h"

namespace lox {
namespace vm {
VM *VM::Instance() {
  static VM object;
  return &object;
}
InterpretResult VM::Run() {
#define READ_BYTE() (*ip++)
#define READ_CONSTANT() (chunk_->constants[READ_BYTE()])
#define BINARY_OP(op) \
  do {                \
    double b = Pop(); \
    double a = Pop(); \
    Push(a op b);     \
  } while (false)

  auto ip = ip_;
  for (;;) {
#ifdef DEBUG_TRACE_EXECUTION
    printf("          ");
    for (Value *slot = stack_; slot != sp_; ++slot) {
      printf("[ ");
      printValue(*slot);
      printf(" ]");
    }
    printf("\n");
    disassembleInstruction(chunk_, (int)(ip - chunk_->code.data()));
#endif
    OpCode instruction;
    switch (instruction = static_cast<OpCode>(READ_BYTE())) {
      case OpCode::OP_CONSTANT: {
        Value constant = READ_CONSTANT();
        Push(constant);
        break;
      }
      case OpCode::OP_ADD:
        BINARY_OP(+);
        break;
      case OpCode::OP_SUBTRACT:
        BINARY_OP(-);
        break;
      case OpCode::OP_MULTIPLY:
        BINARY_OP(*);
        break;
      case OpCode::OP_DIVIDE:
        BINARY_OP(/);
        break;
      case OpCode::OP_NEGATE: {
        Push(-Pop());
        break;
      }
      case OpCode::OP_RETURN: {
        printValue(Pop());
        printf("\n");
        goto EXIT;
      }
    }
  }
EXIT:
  ip_ = ip;
  return InterpretResult::INTERPRET_OK;
#undef READ_BYTE
#undef READ_CONSTANT
#undef BINARY_OP
}
void VM::ResetStack() { sp_ = stack_; }
void VM::Push(Value value) { *sp_++ = value; }
Value VM::Pop() { return *(--sp_); }
InterpretResult VM::Interpret(Chunk *chunk) {
  chunk_ = chunk;
  ip_ = chunk_->code.data();
  return Run();
}
}  // namespace vm
}  // namespace lox

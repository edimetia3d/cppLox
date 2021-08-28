//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"


namespace lox {
namespace vm {
VM *VM::Instance() {
  static VM object;
  return &object;
}
ErrCode VM::Run() {
#define READ_BYTE() (*ip++)
#define READ_CONSTANT() (chunk_->constants[READ_BYTE()])
#define BINARY_OP(op) \
  do {                \
    double b = Pop(); \
    double a = Pop(); \
    Push(a op b);     \
  } while (false)

  auto ip = ip_;
#ifdef DEBUG_TRACE_EXECUTION
  int dbg_op_id = 0;
  chunk_->DumpCode();
  chunk_->DumpConstant();
#endif
  for (;;) {
#ifdef DEBUG_TRACE_EXECUTION
    printf("---- CMD %d ----\n", dbg_op_id);
    chunk_->DumpCode((int)(ip - chunk_->code.data()));
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
        DPRINTF("Return with: ");
        printValue(Pop());
        printf("\n");
        goto EXIT;
      }
    }
#ifdef DEBUG_TRACE_EXECUTION
    DumpStack();
    ++dbg_op_id;
#endif
  }
EXIT:
  ip_ = ip;
  return ErrCode::NO_ERROR;
#undef READ_BYTE
#undef READ_CONSTANT
#undef BINARY_OP
}
void VM::DumpStack() const {
  printf("Stack:");
  for (const Value *slot = stack_; slot != sp_; ++slot) {
    printf("[ ");
    printValue(*slot);
    printf(" ]");
  }
  printf("\n");
}
void VM::ResetStack() { sp_ = stack_; }
void VM::Push(Value value) { *sp_++ = value; }
Value VM::Pop() { return *(--sp_); }
ErrCode VM::Interpret(Chunk *chunk) {
  chunk_ = chunk;
  ip_ = chunk_->code.data();
  return Run();
}
}  // namespace vm
}  // namespace lox

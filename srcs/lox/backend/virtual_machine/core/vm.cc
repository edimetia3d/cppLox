//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"

#include <stdarg.h>

namespace lox {
namespace vm {
VM *VM::Instance() {
  static VM object;
  return &object;
}
ErrCode VM::Run() {
#define READ_BYTE() (*ip_++)
#define CHEK_STACK_TOP_TYPE(TYPE)                   \
  do {                                              \
    if (!Peek().Is##TYPE()) {                       \
      runtimeError("Operand must be a " #TYPE "."); \
      return ErrCode::INTERPRET_RUNTIME_ERROR;      \
    }                                               \
  } while (0)
#define READ_CONSTANT() (chunk_->constants[READ_BYTE()])
#define BINARY_OP(OutputT, op)                                       \
  do {                                                               \
    CHEK_STACK_TOP_TYPE(Number);                                     \
    Value b = Pop();                                                 \
    CHEK_STACK_TOP_TYPE(Number);                                     \
    Value a = Pop();                                                 \
    Push(Value(static_cast<OutputT>(a.AsNumber() op b.AsNumber()))); \
  } while (false)

#ifdef DEBUG_TRACE_EXECUTION
  int dbg_op_id = 0;
  chunk_->DumpCode();
  chunk_->DumpConstant();
#endif
  for (;;) {
#ifdef DEBUG_TRACE_EXECUTION
    printf("---- CMD %d ----\n", dbg_op_id);
    chunk_->DumpCode((int)(ip_ - chunk_->code.data()));
#endif
    OpCode instruction;
    switch (instruction = static_cast<OpCode>(READ_BYTE())) {
      case OpCode::OP_CONSTANT: {
        Value constant = READ_CONSTANT();
        Push(constant);
        break;
      }
      case OpCode::OP_NIL:
        Push(Value());
        break;
      case OpCode::OP_TRUE:
        Push(Value(true));
        break;
      case OpCode::OP_FALSE:
        Push(Value(false));
        break;
      case OpCode::OP_EQUAL: {
        Value b = Pop();
        Value a = Pop();
        Push(Value(a.Equal(b)));
        break;
      }
      case OpCode::OP_GREATER:
        BINARY_OP(bool, >);
        break;
      case OpCode::OP_LESS:
        BINARY_OP(bool, <);
        break;
      case OpCode::OP_ADD:
        if (Peek().IsNumber() && Peek(1).IsNumber()) {
          Value b = Pop();
          Value a = Pop();
          Push(Value(static_cast<double>(a.AsNumber() + b.AsNumber())));
        } else if (Peek().IsObj() && Peek(1).IsObj()) {
          Value b = Pop();
          Value a = Pop();
          Push(Value(ObjString::Concat(a.AsObj()->As<ObjString>(), b.AsObj()->As<ObjString>())));
        } else {
          runtimeError("Add only support string and number.");
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        break;
      case OpCode::OP_SUBTRACT:
        BINARY_OP(double, -);
        break;
      case OpCode::OP_MULTIPLY:
        BINARY_OP(double, *);
        break;
      case OpCode::OP_DIVIDE:
        BINARY_OP(double, /);
        break;
      case OpCode::OP_NOT:
        Push(Value(!Pop().IsTrue()));
        break;
      case OpCode::OP_NEGATE: {
        CHEK_STACK_TOP_TYPE(Number);
        Push(Value(-Pop().AsNumber()));
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
void VM::runtimeError(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputs("\n", stderr);

  size_t instruction = ip_ - chunk_->code.data() - 1;
  int line = chunk_->lines[instruction];
  fprintf(stderr, "[line %d] in script\n", line);
  ResetStack();
}
Value VM::Peek(int distance) {
  assert(distance >= 0);
  return *(sp_ - distance - 1);
}
}  // namespace vm
}  // namespace lox

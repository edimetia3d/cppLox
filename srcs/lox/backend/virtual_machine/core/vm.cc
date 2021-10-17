//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"

#include <lox/backend/virtual_machine/common/builtin_fn.h>

#include <stdarg.h>

namespace lox {
namespace vm {
VM *VM::Instance() {
  static VM object;
  return &object;
}
ErrCode VM::Run() {
  CallFrame *frame = currentFrame();  // we use a local variable as cache, to avoid calling currentFrame.
#define READ_BYTE() (*frame->ip++)
#define READ_SHORT() (frame->ip += 2, (uint16_t)((frame->ip[-2] << 8) | frame->ip[-1]))
#define CHEK_STACK_TOP_TYPE(TYPE)                   \
  do {                                              \
    if (!Peek().Is##TYPE()) {                       \
      runtimeError("Operand must be a " #TYPE "."); \
      return ErrCode::INTERPRET_RUNTIME_ERROR;      \
    }                                               \
  } while (0)
#define READ_CONSTANT() (frame->closure->function->chunk->constants[READ_BYTE()])
#define READ_STRING() ((READ_CONSTANT()).AsObj()->As<ObjInternedString>())
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
#endif
  for (;;) {
#ifdef DEBUG_TRACE_EXECUTION
    printf("---- CMD %d ----\n", dbg_op_id);
    frame->closure->function->chunk->DumpByOffset((int)(frame->ip - frame->closure->function->chunk->code.data()));
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
      case OpCode::OP_POP:
        Pop();
        break;
      case OpCode::OP_GET_LOCAL: {
        uint8_t slot = READ_BYTE();
        Push(frame->slots[slot]);
        break;
      }
      case OpCode::OP_SET_LOCAL: {
        uint8_t slot = READ_BYTE();
        frame->slots[slot] = Peek(0);
        break;
      }
      case OpCode::OP_GET_GLOBAL: {
        ObjInternedString *name = READ_STRING();
        auto entry = globals_.Get(name);
        if (!entry) {
          runtimeError("Undefined variable '%s'.", name->c_str());
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        Push(entry->value);
        break;
      }
      case OpCode::OP_DEFINE_GLOBAL: {
        ObjInternedString *name = READ_STRING();
        globals_.Set(name, Peek(0));
        Pop();
        break;
      }
      case OpCode::OP_SET_GLOBAL: {
        ObjInternedString *name = READ_STRING();
        ;
        if (globals_.Set(name, Peek(0))) {
          globals_.Del(name);
          runtimeError("Undefined variable '%s'.", name->c_str());
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        break;
      }
      case OpCode::OP_GET_UPVALUE: {
        uint8_t slot = READ_BYTE();
        Push(*frame->closure->upvalues[slot]->location);
        break;
      }
      case OpCode::OP_SET_UPVALUE: {
        uint8_t slot = READ_BYTE();
        *frame->closure->upvalues[slot]->location = Peek(0);
        break;
      }
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
          Push(
              Value(ObjInternedString::Concat(a.AsObj()->As<ObjInternedString>(), b.AsObj()->As<ObjInternedString>())));
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
      case OpCode::OP_PRINT: {
#ifdef DEBUG_TRACE_EXECUTION
        printf("------------------------------------------------------------ ");
#endif
        printValue(Pop());
        printf("\n");
        break;
      }
      case OpCode::OP_JUMP: {
        uint16_t offset = READ_SHORT();
        frame->ip += offset;
        break;
      }
      case OpCode::OP_JUMP_IF_FALSE: {
        uint16_t offset = READ_SHORT();
        if (!Peek(0).IsTrue()) frame->ip += offset;
        break;
      }
      case OpCode::OP_JUMP_BACK: {
        uint16_t offset = READ_SHORT();
        frame->ip -= offset;
        break;
      }
      case OpCode::OP_CALL: {
        int argCount = READ_BYTE();
        if (!callValue(Peek(argCount), argCount)) {
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        frame = currentFrame();  // active frame has changed, update cache
        break;
      }
      case OpCode::OP_CLOSURE: {
        auto closure = new ObjRuntimeFunction(READ_CONSTANT().AsObj()->As<ObjFunction>());
        Push(Value(closure));
        for (int i = 0; i < closure->upvalueCount; i++) {
          uint8_t isLocal = READ_BYTE();
          uint8_t index = READ_BYTE();
          if (isLocal) {
            closure->upvalues[i] = captureUpvalue(frame->slots + index);
          } else {
            closure->upvalues[i] = frame->closure->upvalues[index];
          }
        }
        break;
      }
      case OpCode::OP_RETURN: {
        Value result = Pop();
        frameCount--;
        if (frameCount == 0) {
          Pop();
          goto EXIT;
        }
        sp_ = frame->slots;
        Push(result);
        frame = currentFrame();  // active frame has changed, update cache
        break;
      }
    }
#ifdef DEBUG_TRACE_EXECUTION
    DumpStack();
    DumpGlobals();
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
void VM::DumpGlobals() {
  printf("Globals:");
  auto iter = globals_.GetAllItem();
  while (auto entry = iter.next()) {
    printf("{ ");
    printf("%s", entry->key->c_str());
    printf(" : ");
    printValue(entry->value);
    printf(" }");
  }
  printf("\n");
}
void VM::ResetStack() { sp_ = stack_; }
void VM::Push(Value value) { *sp_++ = value; }
Value VM::Pop() { return *(--sp_); }
ErrCode VM::Interpret(ObjFunction *function) {
  Push(Value(function));
  auto rt_fn = new ObjRuntimeFunction(function);
  Pop();
  Push(Value(rt_fn));
  call(rt_fn, 0);
  return Run();
}
void VM::runtimeError(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputs("\n", stderr);
  for (int i = frameCount - 1; i >= 0; i--) {
    CallFrame *frame = &frames[i];
    ObjFunction *function = frame->closure->function;
    size_t instruction = frame->ip - function->chunk->code.data() - 1;
    fprintf(stderr, "[line %d] in ", function->chunk->lines[instruction]);
    fprintf(stderr, "%s()\n", function->name.c_str());
  }
  ResetStack();
}
Value VM::Peek(int distance) {
  assert(distance >= 0);
  return *(sp_ - distance - 1);
}
VM::~VM() {
  auto p = Obj::AllCreatedObj().Head();
  int count = 0;
  while (p) {
    ++count;
    Obj::Destroy(p->val);
    p = p->next;
  }
  if (count) {
    printf("VM destroyed %d CLoxObject at exit.\n", count);
  }
}
bool VM::callValue(Value callee, int count) {
  if (callee.IsObj()) {
    switch (callee.AsObj()->type) {
      case ObjType::OBJ_RUNTIME_FUNCTION:
        return call(callee.AsObj()->As<ObjRuntimeFunction>(), count);
      case ObjType::OBJ_NATIVE_FUNCTION: {
        auto native = callee.AsObj()->As<ObjNativeFunction>()->function;
        Value result = native(count, sp_ - count);
        sp_ -= (count + 1);
        Push(result);
        return true;
      }
      default:
        break;  // Non-callable object type.
    }
  }
  runtimeError("Can only call functions and classes.");
  return false;
}
bool VM::call(ObjRuntimeFunction *closure, int arg_count) {
  if (arg_count != closure->function->arity) {
    runtimeError("Expected %d arguments but got %d.", closure->function->arity, arg_count);
    return false;
  }
  if (frameCount == VM_FRAMES_MAX) {
    runtimeError("Stack overflow.");
    return false;
  }

  CallFrame *frame = &frames[frameCount++];
  frame->closure = closure;
  frame->ip = closure->function->chunk->code.data();
  frame->slots = sp_ - arg_count - 1;
  return true;
}
VM::VM() {
  ResetStack();
  defineBultins();
}
void VM::defineBultins() {
  defineNativeFunction("clock",&clockNative);
}
void VM::defineNativeFunction(const std::string &name, ObjNativeFunction::NativeFn function) {
  Push(Value(ObjInternedString::Make(name.c_str(), name.size())));
  Push(Value(new ObjNativeFunction(function)));
  assert(globals_.Get(Peek(1).AsObj()->As<ObjInternedString>()) == nullptr);
  globals_.Set(Peek(1).AsObj()->As<ObjInternedString>(), Peek(0));
  Pop();
  Pop();
}
ObjUpvalue *VM::captureUpvalue(Value *pValue) { return new ObjUpvalue(pValue); }

}  // namespace vm
}  // namespace lox

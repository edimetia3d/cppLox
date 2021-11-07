//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"

#include <lox/backend/virtual_machine/common/builtin_fn.h>
#include <stdarg.h>

#define DEBUG_STRESS_GC
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
        Object constant = READ_CONSTANT();
        Push(constant);
        break;
      }
      case OpCode::OP_NIL:
        Push(Object());
        break;
      case OpCode::OP_TRUE:
        Push(Object(true));
        break;
      case OpCode::OP_FALSE:
        Push(Object(false));
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
        Object b = Pop();
        Object a = Pop();
        Push(Object(a.Equal(b)));
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
          Object b = Pop();
          Object a = Pop();
          Push(Object(static_cast<double>(a.AsNumber() + b.AsNumber())));
        } else if (Peek().IsHandle() && Peek(1).IsHandle()) {
          Object b = Pop();
          Object a = Pop();
          Push(Object(
              ObjInternedString::Concat(a.AsHandle()->As<ObjInternedString>(), b.AsHandle()->As<ObjInternedString>())));
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
        Push(Object(!Pop().IsTrue()));
        break;
      case OpCode::OP_NEGATE: {
        CHEK_STACK_TOP_TYPE(Number);
        Push(Object(-Pop().AsNumber()));
        break;
      }
      case OpCode::OP_CLASS:
        Push(Object(new ObjClass(READ_STRING()->c_str())));
        break;
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
        auto closure = new ObjRuntimeFunction(READ_CONSTANT().AsHandle()->As<ObjFunction>());
        Push(Object(closure));
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
      case OpCode::OP_CLOSE_UPVALUE:
        closeUpvalues(sp_ - 1);
        Pop();
        break;
      case OpCode::OP_RETURN: {
        Object result = Pop();        // retrive return first
        closeUpvalues(frame->slots);  // discard parameter ,if some parameter is closed ,close them
        frameCount--;
        if (frameCount == 0) {
          Pop();
          tryGC();
          goto EXIT;
        } else {
          sp_ = frame->slots;
          Push(result);
          frame = currentFrame();  // active frame has changed, update cache
          tryGC();
          break;
        }
      }
      case OpCode::OP_GET_ATTR: {
        auto top_v = Peek(0);
        if (top_v.IsHandle() && top_v.AsHandle()->IsType<ObjInstance>()) {
          ObjInstance *instance = top_v.AsHandle()->As<ObjInstance>();
          ObjInternedString *name = READ_STRING();
          if (instance->dict.contains(name)) {
            Pop();  // Instance.
            Push(Object(instance->dict[name]));
            break;
          }
          if (!tryGetBoundMethod(instance->klass, name)) {
            runtimeError("Undefined attr '%s'.", name->c_str());
            return ErrCode::INTERPRET_RUNTIME_ERROR;
          }
          break;
        }
        if (top_v.IsHandle() && top_v.AsHandle()->IsType<ObjClass>()) {
          ObjClass *klass = top_v.AsHandle()->As<ObjClass>();
          auto possible_instance = *currentFrame()->slots;
          if (!possible_instance.IsHandle() || !possible_instance.AsHandle()->IsType<ObjInstance>() ||
              !possible_instance.AsHandle()->As<ObjInstance>()->IsInstance(klass)) {
            runtimeError("class method cannot access", klass);
            return ErrCode::INTERPRET_RUNTIME_ERROR;
          }
          sp_[-1] = possible_instance;  // a hack that replace class with instance
          ObjInternedString *name = READ_STRING();
          if (!tryGetBoundMethod(klass, name)) {
            runtimeError("Undefined method '%s'.", name->c_str());
            return ErrCode::INTERPRET_RUNTIME_ERROR;
          }
          break;
        }
        runtimeError("Only instances have attrs. Only class have methods.");
        return ErrCode::INTERPRET_RUNTIME_ERROR;
      }
      case OpCode::OP_SET_ATTR: {
        auto top_v_1 = Peek(1);
        if (!top_v_1.IsHandle() || !top_v_1.AsHandle()->IsType<ObjInstance>()) {
          runtimeError("Only instances have attr.");
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        ObjInstance *instance = top_v_1.AsHandle()->As<ObjInstance>();
        ObjInternedString *name = READ_STRING();
        instance->dict[name] = Peek();
        // stack need to be [... instance, attr_new_value] -> [...,attr_new_value]
        Object value = Pop();  // expression value temporay discarded
        Pop();                 // pop instance
        Push(value);           // push expression value back
        break;
      }
      case OpCode::OP_INVOKE: {
        auto *method = READ_STRING();
        int argCount = READ_BYTE();
        if (!invoke(method, argCount)) {
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        frame = &frames[frameCount - 1];
        break;
      }
      case OpCode::OP_INHERIT: {
        if (!Peek(1).AsHandle()->IsType<ObjClass>()) {
          runtimeError("Superclass must be a class.");
          return ErrCode::INTERPRET_RUNTIME_ERROR;
        }
        auto superclass = Peek(1).AsHandle()->As<ObjClass>();
        auto subclass = Peek(0).AsHandle()->As<ObjClass>();
        subclass->superclass = superclass;
        subclass->methods.insert(superclass->methods.begin(), superclass->methods.end());
        Pop();  // pop subclass.
        Pop();  // pop superclass
        break;
      }
      case OpCode::OP_METHOD:
        defineMethod(READ_STRING());
        break;
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
void VM::tryGC() const {
#ifdef DEBUG_STRESS_GC
  GC::Instance().collectGarbage();
#else
  if (ObjHandle::ObjCount() > GC::Instance().gc_threashold) {
    GC::Instance().collectGarbage();
    GC::Instance().gc_threashold = ObjHandle::ObjCount() * 1.2;
  }
#endif
}
void VM::DumpStack() const {
  printf("Stack:");
  for (const Object *slot = stack_; slot != sp_; ++slot) {
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
void VM::Push(Object value) { *sp_++ = value; }
Object VM::Pop() { return *(--sp_); }
ErrCode VM::Interpret(ObjFunction *function) {
  Push(Object(function));
  auto rt_fn = new ObjRuntimeFunction(function);
  Pop();
  Push(Object(rt_fn));
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
Object VM::Peek(int distance) {
  assert(distance >= 0);
  return *(sp_ - distance - 1);
}
VM::~VM() {
  int count = 0;
  while (auto p = ObjHandle::AllCreatedObj().Head()) {
    ++count;
    ObjHandle::Destroy(p->val);
  }
  if (count) {
    printf("VM destroyed %d CLoxObject at exit.\n", count);
  }
}
bool VM::callValue(Object callee, int count) {
  if (callee.IsHandle()) {
    switch (callee.AsHandle()->type) {
      case ObjType::OBJ_CLASS: {
        ObjClass *klass = callee.AsHandle()->As<ObjClass>();
        auto new_instance = new ObjInstance(klass);
        Object instance_value(new_instance);
        sp_[-count - 1] = instance_value;  // a hack replace the class object with self
        if (klass->methods.contains(SYMBOL_THIS)) {
          return call(klass->methods[SYMBOL_THIS]->As<ObjRuntimeFunction>(), count);
        } else if (count != 0) {
          runtimeError("Expected 0 arguments but got %d.", count);
          return false;
        }
        // if codes goes here, no init is called , we just leave the instance on stack
        return true;
      }
      case ObjType::OBJ_BOUND_METHOD: {
        ObjBoundMethod *bound = callee.AsHandle()->As<ObjBoundMethod>();
        sp_[-count - 1] = bound->receiver;  // a hack that replace bounded method with self
        return call(bound->method, count);
      }
      case ObjType::OBJ_RUNTIME_FUNCTION:
        return call(callee.AsHandle()->As<ObjRuntimeFunction>(), count);
      case ObjType::OBJ_NATIVE_FUNCTION: {
        auto native = callee.AsHandle()->As<ObjNativeFunction>()->function;
        Object result = native(count, sp_ - count);
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
VM::VM() : marker_register_guard(&markRoots, this) {
  ResetStack();
  defineBultins();
}
void VM::defineBultins() { defineNativeFunction("clock", &clockNative); }
void VM::defineNativeFunction(const std::string &name, ObjNativeFunction::NativeFn function) {
  Push(Object(ObjInternedString::Make(name.c_str(), name.size())));
  Push(Object(new ObjNativeFunction(function)));
  assert(globals_.Get(Peek(1).AsHandle()->As<ObjInternedString>()) == nullptr);
  globals_.Set(Peek(1).AsHandle()->As<ObjInternedString>(), Peek(0));
  Pop();
  Pop();
}
ObjUpvalue *VM::captureUpvalue(Object *pValue) {
  // we insert pValue into a sorted link-list
  // so `p->location` > `p->next->location` is always true
  ObjUpvalue *prevUpvalue = nullptr;
  ObjUpvalue *upvalue = openUpvalues;
  while (upvalue != nullptr && pValue < upvalue->location) {
    prevUpvalue = upvalue;
    upvalue = upvalue->next;
  }

  if (upvalue != nullptr && upvalue->location == pValue) {
    return upvalue;
  }
  auto createdUpvalue = new ObjUpvalue(pValue);
  createdUpvalue->next = upvalue;

  if (prevUpvalue == nullptr) {
    openUpvalues = createdUpvalue;
  } else {
    prevUpvalue->next = createdUpvalue;
  }
  return createdUpvalue;
}
void VM::closeUpvalues(Object *last) {
  // for `p->location` > `p->next->location` is always true
  // we could just delete all directly one by one
  while (openUpvalues != nullptr && openUpvalues->location >= last) {
    ObjUpvalue *upvalue = openUpvalues;
    upvalue->closed = *upvalue->location;
    upvalue->location = &upvalue->closed;
    openUpvalues = upvalue->next;
  }
}
void VM::markRoots(void *vm_p) {
  VM *vm = static_cast<VM *>(vm_p);

  auto &gc = GC::Instance();

  // mark data
  gc.mark(vm->SYMBOL_THIS);

  // mark stacks
  for (Object *slot = vm->stack_; slot < vm->sp_; slot++) {
    gc.mark(*slot);
  }
  // mark globals
  gc.mark(&vm->globals_);

  // mark closures
  for (int i = 0; i < vm->frameCount; i++) {
    gc.mark(vm->frames[i].closure);
  }

  // mark openUpvalue
  for (ObjUpvalue *upvalue = vm->openUpvalues; upvalue != NULL; upvalue = upvalue->next) {
    gc.mark((ObjHandle *)upvalue);
  }
}
void VM::defineMethod(ObjInternedString *name) {
  Object method = Peek(0);
  ObjClass *klass = Peek(1).AsHandle()->As<ObjClass>();
  klass->methods[name] = method.AsHandle()->As<ObjRuntimeFunction>();
  Pop();
}
bool VM::tryGetBoundMethod(ObjClass *klass, ObjInternedString *name) {
  Object method;
  if (!klass->methods.contains(name)) {
    runtimeError("Undefined property '%s'.", name->c_str());
    return false;
  }

  ObjBoundMethod *bound = new ObjBoundMethod(Peek(0), klass->methods[name]->As<ObjRuntimeFunction>());
  Pop();                // Pop instance
  Push(Object(bound));  // replace with new attr value
  return true;
}
bool VM::invoke(ObjInternedString *method_name, int arg_count) {
  Object receiver = Peek(arg_count);
  if (receiver.AsHandle()->IsType<ObjInstance>()) {
    ObjInstance *instance = receiver.AsHandle()->As<ObjInstance>();
    if (instance->dict.contains(method_name)) {
      sp_[-arg_count - 1] = instance->dict[method_name];
      return callValue(instance->dict[method_name], arg_count);
    }
    return invokeFromClass(instance->klass, method_name, arg_count);
  }
  if (receiver.AsHandle()->IsType<ObjClass>()) {
    ObjClass *klass = receiver.AsHandle()->As<ObjClass>();
    auto possible_instance = *currentFrame()->slots;
    if (!possible_instance.IsHandle() || !possible_instance.AsHandle()->IsType<ObjInstance>() ||
        !possible_instance.AsHandle()->As<ObjInstance>()->IsInstance(klass)) {
      runtimeError("class method cannot access", klass);
      return false;
    }
    // a hack that change class to the instance
    sp_[-arg_count - 1] = possible_instance;
    return invokeFromClass(klass, method_name, arg_count);
  }
  runtimeError("Only instances / class have methods.");
  return false;
}

bool VM::invokeFromClass(ObjClass *klass, ObjInternedString *name, int argCount) {
  Object method;
  if (!klass->methods.contains(name)) {
    runtimeError("Undefined property '%s'.", name->c_str());
    return false;
  }
  return call(klass->methods[name], argCount);
}

}  // namespace vm
}  // namespace lox

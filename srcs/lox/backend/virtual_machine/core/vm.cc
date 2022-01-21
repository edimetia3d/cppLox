//
// LICENSE: MIT
//

#include <spdlog/spdlog.h>

#include "lox/backend/virtual_machine/errors.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/backend/virtual_machine/builtins/builtin_fn.h"
#include "lox/backend/virtual_machine/object/object.h"

#define CHUNK_READ_BYTE() (*ip_++)
#define CHUNK_READ_SHORT() (ip_ += 2, (uint16_t)((ip_[-2] << 8) | ip_[-1]))
#define CHEK_STACK_TOP_NUMBER(MSG)             \
  do {                                         \
    if (!Peek().IsNumber()) {                  \
      Error(MSG);                              \
    }                                          \
  } while (0)
#define CHUNK_READ_CONSTANT() (active_frame_->closure->function->chunk->constants[CHUNK_READ_BYTE()])
#define CHUNK_READ_STRING() ((CHUNK_READ_CONSTANT()).AsObject()->DynAs<Symbol>())
#define BINARY_OP(OutputT, op)                                       \
  do {                                                               \
    CHEK_STACK_TOP_NUMBER("Operands must be numbers.");              \
    Value b = Pop();                                                 \
    CHEK_STACK_TOP_NUMBER("Operands must be numbers.");              \
    Value a = Pop();                                                 \
    Push(Value(static_cast<OutputT>(a.AsNumber() op b.AsNumber()))); \
  } while (false)

namespace lox::vm {
VM *VM::Instance() {
  static VM object;
  return &object;
}
void VM::Run() {
  for (;;) {
    switch (static_cast<OpCode>(CHUNK_READ_BYTE())) {
      case OpCode::OP_CONSTANT: {
        Value constant = CHUNK_READ_CONSTANT();
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
        uint8_t slot = CHUNK_READ_BYTE();
        Push(active_frame_->slots[slot]);
        break;
      }
      case OpCode::OP_SET_LOCAL: {
        uint8_t slot = CHUNK_READ_BYTE();
        active_frame_->slots[slot] = Peek(0);
        break;
      }
      case OpCode::OP_GET_GLOBAL: {
        Symbol *name = CHUNK_READ_STRING();
        if (!globals_.contains(name)) {
          Error("Undefined global variable '%s'.", name->c_str());
        }
        Push(globals_[name]);
        break;
      }
      case OpCode::OP_DEFINE_GLOBAL: {
        Symbol *name = CHUNK_READ_STRING();
        globals_[name] = Peek(0);
        Pop();  // OP_DEFINE_GLOBAL is always trigger by a var define statement, so we will pop the value.
        break;
      }
      case OpCode::OP_SET_GLOBAL: {
        Symbol *name = CHUNK_READ_STRING();
        if (!globals_.contains(name)) {
          Error("Undefined variable '%s'.", name->c_str());
        }
        globals_[name] = Peek(0);
        break;
      }
      case OpCode::OP_GET_UPVALUE: {
        uint8_t offset_in_upvalue = CHUNK_READ_BYTE();
        Push(*active_frame_->closure->upvalues[offset_in_upvalue]->location);
        break;
      }
      case OpCode::OP_SET_UPVALUE: {
        uint8_t offset_in_upvalue = CHUNK_READ_BYTE();
        *active_frame_->closure->upvalues[offset_in_upvalue]->location = Peek(0);
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
        } else if (Peek().IsObject() && Peek(1).IsObject()) {
          Value b = Pop();
          Value a = Pop();
          Push(Value(Symbol::Intern(a.AsObject()->DynAs<Symbol>()->Str() + b.AsObject()->DynAs<Symbol>()->Str())));
        } else {
          Error("Operands must be two numbers or two strings.");
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
        CHEK_STACK_TOP_NUMBER("Operand must be a number.");
        Push(Value(-Pop().AsNumber()));
        break;
      }
      case OpCode::OP_CLASS:
        Push(Value(new ObjClass(CHUNK_READ_STRING()->c_str())));
        break;
      case OpCode::OP_PRINT: {
        printf("%s\n", Pop().Str().c_str());
        break;
      }
      case OpCode::OP_JUMP: {
        uint16_t offset = CHUNK_READ_SHORT();
        ip_ += offset;
        break;
      }
      case OpCode::OP_JUMP_IF_FALSE: {
        uint16_t offset = CHUNK_READ_SHORT();
        if (!Peek(0).IsTrue()) ip_ += offset;
        break;
      }
      case OpCode::OP_JUMP_BACK: {
        uint16_t offset = CHUNK_READ_SHORT();
        ip_ -= offset;
        break;
      }
      case OpCode::OP_CALL: {
        int argCount = CHUNK_READ_BYTE();
        CallValue(Peek(argCount), argCount);
        break;
      }
      case OpCode::OP_CLOSURE: {
        auto closure = new ObjClosure(CHUNK_READ_CONSTANT().AsObject()->DynAs<ObjFunction>());
        Push(Value(closure));
        uint8_t upvalue_count = CHUNK_READ_BYTE();
        closure->upvalues.resize(upvalue_count, nullptr);
        for (int i = 0; i < upvalue_count; i++) {
          uint8_t is_on_stack_at_begin = CHUNK_READ_BYTE();
          uint8_t position_at_begin = CHUNK_READ_BYTE();
          if (is_on_stack_at_begin) {
            closure->upvalues[i] = MarkValueNeedToClose(active_frame_->slots + position_at_begin);
          } else {
            closure->upvalues[i] = active_frame_->closure->upvalues[position_at_begin];
          }
        }
        break;
      }
      case OpCode::OP_CLOSE_UPVALUE:
        CloseValuesFromStackPosition(sp_ - 1);
        Pop();
        break;
      case OpCode::OP_RETURN: {
        Value result = Pop();  // retrive return first
        PopFrame();
        if (active_frame_ < frames_) {
          Pop();
          TryGC();
          goto EXIT;
        } else {
          Push(result);
          TryGC();
          break;
        }
      }
      case OpCode::OP_GET_ATTR: {
        auto top_v = Peek(0);
        if (top_v.IsObject() && top_v.AsObject()->DynAs<ObjInstance>()) {
          ObjInstance *instance = top_v.AsObject()->DynAs<ObjInstance>();
          Symbol *name = CHUNK_READ_STRING();
          if (instance->dict.contains(name)) {
            Pop();  // Instance.
            Push(Value(instance->dict[name]));
            break;
          }
          if (!TryGetBoundMethod(instance->klass, name)) {
            Error("Undefined attr '%s'.", name->c_str());
          }
          break;
        }
        if (top_v.IsObject() && top_v.AsObject()->DynAs<ObjClass>()) {
          ObjClass *klass = top_v.AsObject()->DynAs<ObjClass>();
          auto possible_instance = *active_frame_->slots;
          if (!possible_instance.IsObject() || !possible_instance.AsObject()->DynAs<ObjInstance>() ||
              !possible_instance.AsObject()->DynAs<ObjInstance>()->IsInstance(klass)) {
            Error("class method cannot access", klass);
          }
          sp_[-1] = possible_instance;  // a hack that replace class with instance
          Symbol *name = CHUNK_READ_STRING();
          if (!TryGetBoundMethod(klass, name)) {
            Error("Undefined method '%s'.", name->c_str());
          }
          break;
        }
        Error("Only instances have attrs. Only class have methods.");
      }
      case OpCode::OP_SET_ATTR: {
        auto top_v_1 = Peek(1);
        if (!top_v_1.IsObject() || !top_v_1.AsObject()->DynAs<ObjInstance>()) {
          Error("Only instances have attr.");
        }
        ObjInstance *instance = top_v_1.AsObject()->DynAs<ObjInstance>();
        Symbol *name = CHUNK_READ_STRING();
        instance->dict[name] = Peek();
        // stack need to be [... instance, attr_new_value] -> [...,attr_new_value]
        Value value = Pop();  // expression value temporay discarded
        Pop();                // pop instance
        Push(value);          // push expression value back
        break;
      }
      case OpCode::OP_INVOKE: {
        auto *method = CHUNK_READ_STRING();
        int argCount = CHUNK_READ_BYTE();
        DispatchInvoke(method, argCount);
        break;
      }
      case OpCode::OP_INHERIT: {
        if (!Peek(1).AsObject()->DynAs<ObjClass>()) {
          Error("Superclass must be a class.");
        }
        auto superclass = Peek(1).AsObject()->DynAs<ObjClass>();
        auto subclass = Peek(0).AsObject()->DynAs<ObjClass>();
        subclass->superclass = superclass;
        subclass->methods.insert(superclass->methods.begin(), superclass->methods.end());
        Pop();  // pop subclass.
        Pop();  // pop superclass
        break;
      }
      case OpCode::OP_METHOD: {
        Symbol *name = CHUNK_READ_STRING();
        Value method = Peek(0);
        ObjClass *Klass = Peek(1).AsObject()->DynAs<ObjClass>();
        Klass->methods[name] = method.AsObject()->DynAs<ObjClosure>();
        Pop();
        break;
      }
    }
  }
EXIT:
  return;
}
void VM::TryGC() const {
  if (Object::AllCreatedObj().size() > GC::Instance().gc_threashold) {
    GC::Instance().collectGarbage();
    GC::Instance().gc_threashold = Object::AllCreatedObj().size() * 1.2;
  }
}

void VM::ResetStack() { sp_ = stack_; }
void VM::Push(Value value) { *sp_++ = value; }
Value VM::Pop() { return *(--sp_); }
void VM::Interpret(ObjFunction *function) {
  Push(Value(function));
  auto rt_fn = new ObjClosure(function);
  Pop();
  Push(Value(rt_fn));
  CallClosure(rt_fn, 0);
  return Run();
}

void VM::Error(const char *format, ...) {
  std::vector<char> buf(256);
  va_list args;
  va_start(args, format);
  vsnprintf(buf.data(), 256, format, args);
  va_end(args);
#ifndef NDEBUG
  for (auto fp = frames_; fp <= active_frame_; ++fp) {
    ObjFunction *function = fp->closure->function;
    size_t instruction = ip_ - function->chunk->code.data() - 1;
    SPDLOG_DEBUG("[line {}] in {}() \n", function->chunk->lines[instruction], function->name);
  }
#endif
  ResetStack();
  throw RuntimeError(std::string(buf.data()));
}
Value VM::Peek(int distance) {
  assert(distance >= 0);
  return *(sp_ - distance - 1);
}
VM::~VM() {
  int count = 0;
  while (!Object::AllCreatedObj().empty()) {
    ++count;
    delete *Object::AllCreatedObj().begin();
  }
  if (count) {
    SPDLOG_DEBUG("VM destroyed {} CLoxObject at exit.", count);
  }
}
void VM::CallValue(Value callee, int arg_count) {
  if (callee.IsObject()) {
    if (ObjClass *klass = callee.AsObject()->DynAs<ObjClass>()) {
      auto new_instance = new ObjInstance(klass);
      Value instance_value(new_instance);
      sp_[-arg_count - 1] = instance_value;  // a hack replace the class object with self
      if (klass->methods.contains(SYMBOL_INIT)) {
        return CallClosure(klass->methods[SYMBOL_INIT]->DynAs<ObjClosure>(), arg_count);
      } else if (arg_count != 0) {
        Error("Expected 0 arguments but got %d.", arg_count);
      }
      // if codes goes here, no init is called , we just leave the instance on stack
      return;
    }
    if (ObjBoundMethod *bound = callee.AsObject()->DynAs<ObjBoundMethod>()) {
      sp_[-arg_count - 1] = bound->receiver;  // a hack that replace bounded method with self
      return CallClosure(bound->method, arg_count);
    }
    if (ObjClosure *closure = callee.AsObject()->DynAs<ObjClosure>()) {
      return CallClosure(closure, arg_count);
    }
    if (auto native = callee.AsObject()->DynAs<ObjNativeFunction>()) {
      Value result = native->function(arg_count, sp_ - arg_count);
      sp_ -= (arg_count + 1);
      Push(result);
      return;
    }
  }
  Error("Can only call functions and classes.");
}
void VM::CallClosure(ObjClosure *callee, int arg_count) {
  if (arg_count != callee->function->arity) {
    Error("Expected %d arguments but got %d.", callee->function->arity, arg_count);
  }
  if ((active_frame_ - frames_ + 1) == VM_FRAMES_MAX) {
    Error("Too many stack frames");
  }
  PushFrame(callee);
  return;
}
VM::VM() : marker_register_guard(&MarkGCRoots, this) {
  active_frame_ = &frames_[0] - 1;  // to make the active_frame_ comparable, init with a dummy value
  ResetStack();
  DefineBuiltins();
}
void VM::DefineBuiltins() {
  for (const auto &pair : AllNativeFn()) {
    const std::string &Name = pair.first.c_str();
    Push(Value(Symbol::Intern(Name)));
    Push(Value(new ObjNativeFunction(pair.second)));
    auto Key = Peek(1).AsObject()->DynAs<Symbol>();
    assert(!globals_.contains(Key));
    globals_[Key] = Peek(0);
    Pop();
    Pop();
  }
}

ObjUpvalue *VM::MarkValueNeedToClose(Value *local_value_stack_pos) {
  // we insert local_value_stack_pos into a sorted link-list
  // so `p->location` > `p->next->location` is always true
  ObjUpvalue *prevUpvalue = nullptr;
  ObjUpvalue *upvalue = open_upvalues;
  while (upvalue != nullptr && local_value_stack_pos < upvalue->location) {
    prevUpvalue = upvalue;
    upvalue = upvalue->next;
  }

  if (upvalue != nullptr && upvalue->location == local_value_stack_pos) {
    return upvalue;
  }
  auto createdUpvalue = new ObjUpvalue(local_value_stack_pos);
  createdUpvalue->next = upvalue;

  if (prevUpvalue == nullptr) {
    open_upvalues = createdUpvalue;
  } else {
    prevUpvalue->next = createdUpvalue;
  }
  return createdUpvalue;
}
void VM::CloseValuesFromStackPosition(Value *stack_position) {
  // for `p->location` > `p->next->location` is always true
  // we could just delete open_upvalues at right side of last
  while (open_upvalues != nullptr && open_upvalues->location >= stack_position) {
    ObjUpvalue *upvalue = open_upvalues;
    upvalue->closed = *upvalue->location;
    upvalue->location = &upvalue->closed;
    open_upvalues = upvalue->next;
  }
}
void VM::MarkGCRoots(void *vm_p) {
  VM *vm = static_cast<VM *>(vm_p);

  // RecursiveMark data
  GC::Instance().RecursiveMark(vm->SYMBOL_INIT);

  // RecursiveMark stacks
  for (Value *slot = vm->stack_; slot < vm->sp_; slot++) {
    if (slot->IsObject()) {
      GC::Instance().RecursiveMark(slot->AsObject());
    }
  }
  // RecursiveMark globals
  for (auto pair : vm->globals_) {
    GC::Instance().RecursiveMark(pair.first);
    if (pair.second.IsObject()) {
      GC::Instance().RecursiveMark(pair.second.AsObject());
    }
  }

  // RecursiveMark closures
  for (auto fp = vm->frames_; fp <= vm->active_frame_; ++fp) {
    GC::Instance().RecursiveMark(fp->closure);
  }

  // RecursiveMark openUpvalue
  for (ObjUpvalue *upvalue = vm->open_upvalues; upvalue != nullptr; upvalue = upvalue->next) {
    GC::Instance().RecursiveMark((Object *)upvalue);
  }
}
bool VM::TryGetBoundMethod(ObjClass *klass, Symbol *name) {
  Value method;
  if (!klass->methods.contains(name)) {
    Error("Undefined property '%s'.", name->c_str());
    return false;
  }

  ObjBoundMethod *bound = new ObjBoundMethod(Peek(0), klass->methods[name]->DynAs<ObjClosure>());
  Pop();               // Pop instance
  Push(Value(bound));  // replace with new attr value
  return true;
}
void VM::DispatchInvoke(Symbol *method_name, int arg_count) {
  Value receiver = Peek(arg_count);
  if (receiver.AsObject()->DynAs<ObjInstance>()) {
    ObjInstance *instance = receiver.AsObject()->DynAs<ObjInstance>();
    if (instance->dict.contains(method_name)) {
      sp_[-arg_count - 1] = instance->dict[method_name];
      return CallValue(instance->dict[method_name], arg_count);
    }
    return InvokeMethod(instance->klass, method_name, arg_count);
  }
  if (receiver.AsObject()->DynAs<ObjClass>()) {
    ObjClass *klass = receiver.AsObject()->DynAs<ObjClass>();
    auto possible_instance = *active_frame_->slots;
    if (!possible_instance.IsObject() || !possible_instance.AsObject()->DynAs<ObjInstance>() ||
        !possible_instance.AsObject()->DynAs<ObjInstance>()->IsInstance(klass)) {
      Error("class method cannot access", klass);
    }
    // a hack that change class to the instance
    sp_[-arg_count - 1] = possible_instance;
    return InvokeMethod(klass, method_name, arg_count);
  }
  Error("Only instances / class have methods.");
}

void VM::InvokeMethod(ObjClass *klass, Symbol *method_name, int arg_count) {
  if (!klass->methods.contains(method_name)) {
    Error("Undefined property '%s'.", method_name->c_str());
  }
  return CallClosure(klass->methods[method_name], arg_count);
}

void VM::PushFrame(ObjClosure *closure) {
  ++active_frame_;
  active_frame_->closure = closure;
  active_frame_->return_address = ip_;
  active_frame_->slots = sp_ - closure->function->arity - 1;
  ip_ = closure->function->chunk->code.data();
  // no need to update sp_, it is already on correct location
}

void VM::PopFrame() {
  // close stack values if necessary.
  CloseValuesFromStackPosition(active_frame_->slots);
  ip_ = active_frame_->return_address;
  sp_ = active_frame_->slots;
  --active_frame_;
}

}  // namespace lox::vm

//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/vm.h"

#include <spdlog/spdlog.h>

#include "lox/backend/virtual_machine/builtins/builtin_fn.h"
#include "lox/backend/virtual_machine/debug/debug.h"
#include "lox/backend/virtual_machine/errors.h"
#include "lox/backend/virtual_machine/object/object.h"
#include "lox/global_setting.h"

#define CHUNK_READ_BYTE() (*ip_++)
#define CHUNK_READ_SHORT() (ip_ += 2, (uint16_t)((ip_[-2] << 8) | ip_[-1]))
#define CHEK_STACK_TOP_NUMBER(MSG) \
  do {                             \
    if (!Peek().IsNumber()) {      \
      Error(MSG);                  \
    }                              \
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
#ifndef NDEBUG
  SPDLOG_DEBUG("===============================================");
  SPDLOG_DEBUG("===============Start Execution=================");
  SPDLOG_DEBUG("===============================================");
  if (lox::GlobalSetting().single_step_mode) {
    printf("Press Enter to start/continue execution.\n");
    getchar();
  }
#endif
  for (;;) {
#ifndef NDEBUG
    if (lox::GlobalSetting().runtime_dump_frequency != RuntimeDumpFrequency::NONE) {
      int offset = ip_ - active_frame_->closure->function->chunk->code.data();

      bool should_dump_runtime_info = false;
      switch (lox::GlobalSetting().runtime_dump_frequency) {
        case RuntimeDumpFrequency::EVERY_INSTRUCTION:
          should_dump_runtime_info = true;
          break;  // no need to fall through
        case RuntimeDumpFrequency::EVERY_LINE: {
          bool line_should_dump = offset > 0 && active_frame_->closure->function->chunk->lines[offset] !=
                                                    active_frame_->closure->function->chunk->lines[offset - 1];
          should_dump_runtime_info |= line_should_dump;
          [[fallthrough]];
        }
        case RuntimeDumpFrequency::EVERY_FUNCTION:
          should_dump_runtime_info |= (offset == 0);
          break;
        case RuntimeDumpFrequency::NONE:
          break;
      }

      if (should_dump_runtime_info) {
        // use gdb to break in this branch, to single step through the lox source code.
        DumpStack(this);
        DumpGlobal(this);
        if (lox::GlobalSetting().single_step_mode) {
          getchar();
        }
        SPDLOG_DEBUG("===============================================");
      }
      DumpInstruction(active_frame_->closure->function->chunk.get(), offset);
    }
#endif
    auto instruction = static_cast<OpCode>(CHUNK_READ_BYTE());
    switch (instruction) {
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
          Error("Undefined variable '%s'.", name->c_str());
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
        Push(Value(Object::Make<ObjClass>(CHUNK_READ_STRING()->c_str())));
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
        auto closure = Object::Make<ObjClosure>(CHUNK_READ_CONSTANT().AsObject()->DynAs<ObjFunction>());
        uint8_t upvalue_count = CHUNK_READ_BYTE();
        uint8_t extra_closed_value = 0;
        closure->upvalues.resize(upvalue_count, nullptr);
        for (int i = 0; i < upvalue_count; i++) {
          uint8_t src = CHUNK_READ_BYTE();
          uint8_t position_at_begin = CHUNK_READ_BYTE();
          switch (src) {
            case 0:  // on known slot
              closure->upvalues[i] = MarkValueNeedToClose(active_frame_->slots + position_at_begin);
              break;
            case 1:  // on enclosing function, that is , current active function
              closure->upvalues[i] = active_frame_->closure->upvalues[position_at_begin];
              break;
            case 2: {  // on stack top
              closure->upvalues[i] = Object::Make<ObjUpvalue>(nullptr);
              closure->upvalues[i]->closed = Peek(position_at_begin);
              closure->upvalues[i]->location = &closure->upvalues[i]->closed;
              ++extra_closed_value;
              break;
            }
            default:
              Error("Invalid upvalue.");
          }
        }
        sp_ -= extra_closed_value;
        Push(Value(closure));  // the slot had been pre-occupied by the compiler
        break;
      }
      case OpCode::OP_CLOSE_UPVALUE:
        CloseValuesFromStackPosition(sp_ - 1);
        Pop();
        break;
      case OpCode::OP_RETURN: {
        Value result = Pop();  // retrive return first
        PopFrame();
        GC::Instance().TryCollet();
        if (active_frame_ < frames_) {
          Pop();
          goto EXIT;
        } else {
          Push(result);
          break;
        }
      }
      case OpCode::OP_GET_ATTR: {
        auto instnce_value = Peek(0);
        if (!instnce_value.IsObject() || !instnce_value.AsObject()->DynAs<ObjInstance>()) {
          Error("Only instances have properties.");
        }

        auto *instance = instnce_value.AsObject()->DynAs<ObjInstance>();
        Symbol *name = CHUNK_READ_STRING();
        Value final_attr;
        if (instance->dict().contains(name)) {
          final_attr = Value(instance->dict()[name]);
        } else if (instance->klass->methods.contains(name)) {
          auto *bound = Object::Make<ObjBoundMethod>(instance, instance->klass->methods[name]->As<ObjClosure>());
          final_attr = Value(bound);
        } else {
          Error("Undefined property '%s'.", name->c_str());
        }
        Pop();             // Pop instance
        Push(final_attr);  // replace with new attr value
        break;
      }
      case OpCode::OP_SET_ATTR: {
        auto instance_value = Peek(1);
        if (!instance_value.IsObject() || !instance_value.AsObject()->DynAs<ObjInstance>()) {
          Error("Only instances have fields.");
        }

        auto *instance = instance_value.AsObject()->DynAs<ObjInstance>();
        Symbol *name = CHUNK_READ_STRING();
        instance->dict()[name] = Peek();
        Value value = Pop();  // pop rvalue
        Pop();                // pop lvalue
        Push(value);          // push rvalue back
        break;
      }
      case OpCode::OP_INVOKE: {
        auto *method_name = CHUNK_READ_STRING();
        int arg_count = CHUNK_READ_BYTE();
        Value instance_value = Peek(arg_count);
        if (!instance_value.IsObject() || !instance_value.AsObject()->DynAs<ObjInstance>()) {
          Error("Only instances have methods.");
        }

        ObjInstance *instance = instance_value.AsObject()->DynAs<ObjInstance>();
        if (instance->dict().contains(method_name)) {
          CallValue(instance->dict()[method_name], arg_count);
        } else {
          if (!instance->klass->methods.contains(method_name)) {
            Error("Undefined property '%s'.", method_name->c_str());
          }
          CallClosure(instance, instance->klass->methods[method_name], arg_count);
        }
        break;
      }
      case OpCode::OP_INHERIT: {
        if (!Peek(1).IsObject() || !Peek(1).AsObject()->DynAs<ObjClass>()) {
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
      case OpCode::OP_INSTANCE_TYPE_CAST: {
        // todo: safety checks
        auto target_class = Peek(1).AsObject()->DynAs<ObjClass>();
        auto instance = Peek(0).AsObject()->DynAs<ObjInstance>();
        Pop();
        *(sp_ - 1) = Value(instance->Cast(target_class));  // replace stack top with new value
        break;
      }
    }
  }
EXIT:
  return;
}

void VM::ResetStack() { sp_ = stack_; }
void VM::Push(Value value) { *sp_++ = value; }
Value VM::Pop() { return *(--sp_); }
void VM::Interpret(ObjFunction *function) {
  auto rt_fn = Object::Make<ObjClosure>(function);
  Push(Value(rt_fn));
  CallClosure(nullptr, rt_fn, 0);
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
VM::~VM() { GC::Instance().markers.erase(this); }
void VM::CallValue(Value callee, int arg_count) {
  if (callee.IsObject()) {
    if (ObjClass *klass = callee.AsObject()->DynAs<ObjClass>()) {
      auto new_instance = Object::Make<ObjInstance>(klass);
      Value instance_value(new_instance);
      if (klass->methods.contains(SYMBOL_INIT)) {
        // call init will leave `this` on the stack.
        return CallClosure(new_instance, klass->methods[SYMBOL_INIT]->DynAs<ObjClosure>(), arg_count);
      } else if (arg_count != 0) {
        Error("Expected 0 arguments but got %d.", arg_count);
      } else {
        Pop();                 // pop the callee, which is the class
        Push(instance_value);  // leave instance on stack
      }
      return;
    }
    if (ObjBoundMethod *bound = callee.AsObject()->DynAs<ObjBoundMethod>()) {
      return CallClosure(bound->bounded_this, bound->method, arg_count);
    }
    if (ObjClosure *closure = callee.AsObject()->DynAs<ObjClosure>()) {
      return CallClosure(nullptr, closure, arg_count);
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
void VM::CallClosure(ObjInstance *this_instance, ObjClosure *callee, int arg_count) {
  if (arg_count != callee->function->arity) {
    Error("Expected %d arguments but got %d.", callee->function->arity, arg_count);
  }
  if ((active_frame_ - frames_) == (VM_FRAMES_LIMIT - 1)) {
    Error("Stack overflow.");
  }
  PushFrame(callee);
  if (this_instance) {
    active_frame_->slots[0] = Value(this_instance);  // a hack to replace slots[0] to this_instance
  } else {
    active_frame_->slots[0] = Value(callee);
  }
}
VM::VM() {
  active_frame_ = &frames_[0] - 1;  // to make the active_frame_ comparable, init with a dummy value
  ResetStack();
  DefineBuiltins();
  GC::Instance().markers[this] = [this]() { MarkGCRoots(); };
}
void VM::DefineBuiltins() {
  for (const auto &pair : AllNativeFn()) {
    auto key = Symbol::Intern(pair.first);
    assert(!globals_.contains(key));
    globals_[key] = Value(Object::Make<ObjNativeFunction>(pair.second));
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
  auto createdUpvalue = Object::Make<ObjUpvalue>(local_value_stack_pos);
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
void VM::MarkGCRoots() {
  VM *vm = this;

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

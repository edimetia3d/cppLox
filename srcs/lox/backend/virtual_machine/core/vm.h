//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_VM_VM_H_
#define CLOX_SRCS_CLOX_VM_VM_H_

#include "lox/err_code.h"
#include "lox/object/gc.h"

#include "lox/backend/virtual_machine/object/object.h"
#include "lox/backend/virtual_machine/core/chunk.h"

/**
 * The Core Component of the Virtual Machine are {CallFrame,Stack,Globals}
 *
 * The `stack` and `global` are unique to vm, CallFrame will only access `stack` and `global`.
 *
 * All execution start with a CallFrame. A CallFrame contains the function, which holds the chunk, a `ip`, which point
 * to somewhere in the chunk, a `slots` which is the `stack` of the CallFrame.
 *
 * When call a new function, vm will create a new CallFrame, set the ip to the first instruction of the chunk, and set
 * slots point to correct stack.
 *
 * When return from a function, vm just discard the frame, and leave only the last CallFrame's return value on stack.

 */
#define VM_FRAMES_MAX 64
#define VM_STACK_MAX (VM_FRAMES_MAX * STACK_LOOKUP_MAX)

namespace lox::vm {

struct CallFrame {
  ObjClosure *closure;  // the callee function
  uint8_t *ip;          // pointer to somewhere in function->chunk
  Value *slots;         // pointer to somewhere in VM::stack_
};

/**
 * A stack machine, it has is most important part of the virtual machine, and very important to the performance.
 */
class VM {
 public:
  VM();
  ~VM();
  static VM *Instance();
  ErrCode Interpret(ObjFunction *function);  // interpret a function
 private:
  ErrCode Run();  // the core vm dispatch loop.

  void Push(Value value);
  Value Peek(int distance = 0);
  Value Pop();

  void ResetStack();
  void DefineBuiltins();

  CallFrame *CurrentFrame();

  void RuntimeError(const char *format, ...);
  bool CallValue(Value callee, int arg_count);
  bool CallClosure(ObjClosure *callee, int arg_count);

  ObjUpvalue *MarkValueNeedToClose(Value *local_value_stack_pos);
  void CloseValuesAfterStack(Value *stack_position);

  bool TryGetBoundMethod(ObjClass *klass, Symbol *name);
  bool DispatchInvoke(Symbol *method_name, int arg_count);  // just a dispatcher of `ClassName.foo()`/`instance.foo()`
  bool InvokeMethod(ObjClass *klass, Symbol *method_name, int arg_count);

  static void MarkGCRoots(void *vm);
  void TryGC() const;  // only vm will trigger gc at a suitable time.

  friend void DumpStack(const VM *vm);
  friend void DumpGlobal(const VM *vm);

 private:
  CallFrame frames_[VM_FRAMES_MAX];
  int frame_count_ = 0;
  Value stack_[VM_STACK_MAX];
  Value *sp_ = nullptr;  // pointer to somewhere in stack_
  std::unordered_map<Symbol *, Value> globals_;

  ObjUpvalue *open_upvalues;  // a linked-list that stores all the upvalues that has not been closed
  Symbol *const SYMBOL_INIT{Symbol::Intern("init")};  // just used to avoid repeated symbol creation
  GC::RegisterMarkerGuard marker_register_guard;      // used to register the marker function
};
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_VM_VM_H_

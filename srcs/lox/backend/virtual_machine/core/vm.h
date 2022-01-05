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

/**
 * A explicit call frame, which contains the function, return_address, and stack. (We call it "explicit", because some
 * implementation will store these values on stack. Also, our implementation will store the closure on stack too, but
 * only used to reserve a stack slot to support some runtime hack.)
 *
 * slots is the `local stack` of the call frame, it is a part of the "global vm stack", for we do not use fixed size
 * call frame, the size of the slots is dynamic.
 *
 * The slot of the call frame will always begin with the `self` of the function, and the `argv` of the function, that is
 * slots[0] is the ObjClosure, and slots[1] to slots[argc+1] are the arguments.
 *
 * We save the closure explicitly, because we will use runtime hacks sometimes, and the closure stored in the stack will
 * possibly be changed.
 *
 * A OP::RETURN will cause a frame switch, the frame switch will only leave the return value on caller's stack.
 * And, a return frame switch will also discard the frame's stack, because the sp_ of vm will update to the caller's
 * slots, thus all the value in the frame's stack will be discarded implicitly.
 */
struct CallFrame {
  ObjClosure *closure;      // the callee function
  uint8_t *return_address;  // pointer to somewhere in caller's chunk
  Value *slots;             // pointer to somewhere in VM::stack_
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

  void RuntimeError(const char *format, ...);
  bool CallValue(Value callee, int arg_count);
  bool CallClosure(ObjClosure *callee, int arg_count);

  ObjUpvalue *MarkValueNeedToClose(Value *local_value_stack_pos);
  void CloseValuesFromStackPosition(Value *stack_position);

  bool TryGetBoundMethod(ObjClass *klass, Symbol *name);
  bool DispatchInvoke(Symbol *method_name, int arg_count);  // just a dispatcher of `ClassName.foo()`/`instance.foo()`
  bool InvokeMethod(ObjClass *klass, Symbol *method_name, int arg_count);

  static void MarkGCRoots(void *vm);
  void TryGC() const;  // only vm will trigger gc at a suitable time.

  void PushFrame(ObjClosure *closure);  // push a new call frame and active it
  void PopFrame();                      // pop the current call frame and discard it

  friend void DumpStack(const VM *vm);
  friend void DumpGlobal(const VM *vm);

 private:
  CallFrame frames_[VM_FRAMES_MAX];
  CallFrame *active_frame_ = nullptr;  // point to the current active frame
  Value stack_[VM_STACK_MAX];
  Value *sp_ = nullptr;    // the pointer point to global stack top.(also the top of last call-frame's local stack)
  uint8_t *ip_ = nullptr;  // pointer to the next instruction to be executed
  std::unordered_map<Symbol *, Value> globals_;

  ObjUpvalue *open_upvalues;  // a linked-list that stores all the upvalues that has not been closed
  Symbol *const SYMBOL_INIT{Symbol::Intern("init")};  // just used to avoid repeated symbol creation
  GC::RegisterMarkerGuard marker_register_guard;      // used to register the marker function
};
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_VM_VM_H_

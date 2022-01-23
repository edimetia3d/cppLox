//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_VM_VM_H_
#define CLOX_SRCS_CLOX_VM_VM_H_

#include "lox/object/gc.h"

#include "lox/backend/virtual_machine/object/object.h"
#include "lox/backend/virtual_machine/core/chunk.h"

#define VM_FRAMES_LIMIT 64
#define VM_STACK_LIMIT (VM_FRAMES_LIMIT * STACK_COUNT_LIMIT)

namespace lox::vm {

/**
 * A explicit call frame, which contains the function, return_address, and stack. (We call it "explicit", because some
 * implementation will store these values on stack. Also, our implementation will store the closure on stack too, but
 * only used to reserve a stack slot to support some runtime hack.)
 *
 * The `slots` is the `local stack` of the call frame, it is a part of the "global vm stack", for we do not use fixed
 * size call frame, the size of the slots is dynamic. The `slots` is mainly used to support random access to the stack.
 *
 * The `slots` of the call frame will always begin with the `self` of the function, and then the `argv` of the function,
 * that is slots[0] is ObjClosure, and slots[1] to slots[argc+1] are the arguments.
 *
 * Though there is a clousre object on stack, we save the closure explicitly, because the one on stack might get
 * changed.
 *
 * A OP::RETURN will cause a frame switch, the frame switch will only leave the return value on caller's stack.
 * And, a return frame switch will also discard everything in the callee frame's local stack.
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
  void Interpret(ObjFunction *function);  // interpret a script
 private:
  void Run();  // the core vm dispatch loop.

  inline void Push(Value value);
  inline Value Peek(int distance = 0);
  inline Value Pop();

  inline void ResetStack();
  void DefineBuiltins();

  void Error(const char *format, ...);
  void CallValue(Value callee, int arg_count);
  void CallClosure(ObjInstance *this_instance, ObjClosure *callee, int arg_count);

  ObjUpvalue *MarkValueNeedToClose(Value *local_value_stack_pos);
  void CloseValuesFromStackPosition(Value *stack_position);

  static void MarkGCRoots(void *vm);
  void TryGC() const;  // only vm will trigger gc at a suitable time.

  void PushFrame(ObjClosure *closure);  // push a new call frame and active it
  void PopFrame();                      // pop the current call frame and discard it

  friend void DumpStack(const VM *vm);
  friend void DumpGlobal(const VM *vm);

 private:
  CallFrame frames_[VM_FRAMES_LIMIT];
  CallFrame *active_frame_ = nullptr;  // point to the current active frame
  Value stack_[VM_STACK_LIMIT];
  Value *sp_ = nullptr;    // the pointer point to global stack top.(also the top of last call-frame's local stack)
  uint8_t *ip_ = nullptr;  // pointer to the next instruction to be executed
  std::unordered_map<Symbol *, Value> globals_;

  ObjUpvalue *open_upvalues;  // a linked-list that stores all the upvalues that has not been closed
  Symbol *const SYMBOL_INIT{Symbol::Intern("init")};  // just used to avoid repeated symbol creation
  GC::RegisterMarkerGuard marker_register_guard;      // used to register the marker function
};
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_VM_VM_H_

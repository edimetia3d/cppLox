//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_VM_VM_H_
#define CLOX_SRCS_CLOX_VM_VM_H_
#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/common/clox_value.h"
#include "lox/backend/virtual_machine/common/err_code.h"

#define VM_FRAMES_MAX 64
#define STACK_LOOKUP_OFFSET_MAX (UINT8_MAX + 1)  // same as the value in compiler
#define VM_STACK_MAX (VM_FRAMES_MAX * IMMEDIATE_NUMBER_MAX)
// Stack uses relative address , and command has a 256 limit,
// so a command can find only in [frame_pointer,frame_pointer+256]

namespace lox {
namespace vm {

struct CallFrame {
  ObjRuntimeFunction *closure;  // the callee function
  uint8_t *ip;                  // pointer to somewhere in function->chunk
  Value *slots;                 // pointer to somewhere in VM::stack_
};

/**
 * A stack machine
 */
class VM {
 public:
  static VM *Instance();
  ErrCode Interpret(ObjFunction *function);

 private:
  void ResetStack();
  VM();
  void defineBultins();
  ErrCode Run();
  void Push(Value value);
  Value Peek(int distance = 0);
  Value Pop();
  CallFrame frames[VM_FRAMES_MAX];
  CallFrame *currentFrame() {
    assert(frameCount > 0);
    return &frames[frameCount - 1];
  }
  int frameCount = 0;
  Value stack_[STACK_LOOKUP_OFFSET_MAX];
  HashMap<ObjInternedString *, Value, ObjInternedString::Hash> globals_;
  Value *sp_ = nullptr;  // stack pointer
  void DumpStack() const;
  void DumpGlobals();
  void runtimeError(const char *format, ...);
  ~VM();
  bool callValue(Value callee, int count);
  bool call(ObjRuntimeFunction *closure, int arg_count);
  void defineNativeFunction(const std::string &name, ObjNativeFunction::NativeFn function);
  ObjUpvalue *captureUpvalue(Value *pValue);
};
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_VM_VM_H_

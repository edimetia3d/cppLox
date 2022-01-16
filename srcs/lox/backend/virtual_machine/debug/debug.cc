//
// LICENSE: MIT
//
#include <spdlog/spdlog.h>

#include "lox/backend/virtual_machine/core/chunk.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/backend/virtual_machine/object/object.h"
#include "lox/object/value.h"

namespace lox::vm {

class ChunkDump {
 public:
  explicit ChunkDump(const Chunk &chunk) : chunk(&chunk), latest_line_buf(1000, '\0') {}

  int DumpCodeAt(int offset) {
    if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
      AppendLatestLine("%04d    | ", offset);
    } else {
      AppendLatestLine("%04d %4d ", offset, chunk->lines[offset] + 1);
    }
    auto new_offset = AppendInstruction(offset);
    SPDLOG_DEBUG(ReleaseLatestLine());
    return new_offset;
  }

  void DumpAllCode() {
    SPDLOG_DEBUG("=========== CODE BEGIN ===========");
    for (int offset = 0; offset < chunk->ChunkSize();) {
      offset = DumpCodeAt(offset);
    }
    SPDLOG_DEBUG("=========== CODE END ===========");
  }

  void DumpConstant() {
    AppendLatestLine("Constant: ");
    for (const Value *p = chunk->constants.data(); p != chunk->constants.data() + chunk->constants.size(); ++p) {
      AppendLatestLine("[ %s ]", p->Str().c_str());
    }
    SPDLOG_DEBUG(ReleaseLatestLine());
  }

 private:
  int AppendInstruction(int offset) {
    auto instruction = static_cast<OpCode>(chunk->code[offset]);
    switch (instruction) {
      case OpCode::OP_CONSTANT:
        return ConstantInstruction("OP_CONSTANT", offset);
      case OpCode::OP_NIL:
        return SimpleInstruction("OP_NIL", offset);
      case OpCode::OP_TRUE:
        return SimpleInstruction("OP_TRUE", offset);
      case OpCode::OP_FALSE:
        return SimpleInstruction("OP_FALSE", offset);
      case OpCode::OP_EQUAL:
        return SimpleInstruction("OP_EQUAL", offset);
      case OpCode::OP_GREATER:
        return SimpleInstruction("OP_GREATER", offset);
      case OpCode::OP_LESS:
        return SimpleInstruction("OP_LESS", offset);
      case OpCode::OP_ADD:
        return SimpleInstruction("OP_ADD", offset);
      case OpCode::OP_SUBTRACT:
        return SimpleInstruction("OP_SUBTRACT", offset);
      case OpCode::OP_MULTIPLY:
        return SimpleInstruction("OP_MULTIPLY", offset);
      case OpCode::OP_DIVIDE:
        return SimpleInstruction("OP_DIVIDE", offset);
      case OpCode::OP_NOT:
        return SimpleInstruction("OP_NOT", offset);
      case OpCode::OP_NEGATE:
        return SimpleInstruction("OP_NEGATE", offset);
      case OpCode::OP_RETURN:
        return SimpleInstruction("OP_RETURN", offset);
      case OpCode::OP_PRINT:
        return SimpleInstruction("OP_PRINT", offset);
      case OpCode::OP_POP:
        return SimpleInstruction("OP_POP", offset);
      case OpCode::OP_DEFINE_GLOBAL:
        return ConstantInstruction("OP_DEFINE_GLOBAL", offset);
      case OpCode::OP_GET_GLOBAL:
        return ConstantInstruction("OP_GET_GLOBAL", offset);
      case OpCode::OP_SET_GLOBAL:
        return ConstantInstruction("OP_SET_GLOBAL", offset);
      case OpCode::OP_GET_LOCAL:
        return ByteInstruction("OP_GET_LOCAL", offset);
      case OpCode::OP_SET_LOCAL:
        return ByteInstruction("OP_SET_LOCAL", offset);
      case OpCode::OP_JUMP:
        return JumpInstruction("OP_JUMP", 1, offset);
      case OpCode::OP_JUMP_IF_FALSE:
        return JumpInstruction("OP_JUMP_IF_FALSE", 1, offset);
      case OpCode::OP_JUMP_BACK:
        return JumpInstruction("OP_JUMP_BACK", -1, offset);
      case OpCode::OP_CALL:
        return ByteInstruction("OP_CALL", offset);
      case OpCode::OP_CLOSURE:
        return ClosureInstruction(offset);
      case OpCode::OP_GET_UPVALUE:
        return ByteInstruction("OP_GET_UPVALUE", offset);
      case OpCode::OP_SET_UPVALUE:
        return ByteInstruction("OP_SET_UPVALUE", offset);
      case OpCode::OP_CLASS:
        return ConstantInstruction("OP_CLASS", offset);
      case OpCode::OP_GET_ATTR:
        return ConstantInstruction("OP_GET_ATTR", offset);
      case OpCode::OP_SET_ATTR:
        return ConstantInstruction("OP_SET_ATTR", offset);
      case OpCode::OP_METHOD:
        return ConstantInstruction("OP_METHOD", offset);
      case OpCode::OP_INVOKE:
        return InvokeInstruction("OP_INVOKE", offset);
      case OpCode::OP_INHERIT:
        return SimpleInstruction("OP_INHERIT", offset);
      default:
        AppendLatestLine("Unknown opcode %d\n", (int)instruction);
        return offset + 1;
    }
  }

  int SimpleInstruction(const char *name, int offset) {
    AppendLatestLine("%s", name);
    return offset + 1;
  }

  int ConstantInstruction(const char *name, int offset) {
    uint8_t const_index = chunk->code[offset + 1];
    AppendLatestLine("%-16s %4d->[%s]", name, const_index, chunk->constants[const_index].Str().c_str());
    return offset + 2;
  }

  int ByteInstruction(const char *name, int offset) {
    uint8_t slot = chunk->code[offset + 1];
    AppendLatestLine("%-16s %4d", name, slot);
    return offset + 2;
  }

  int JumpInstruction(const char *name, int sign, int offset) {
    uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
    jump |= chunk->code[offset + 2];
    AppendLatestLine("%-16s %4d -> %d", name, offset, offset + 3 + sign * jump);
    return offset + 3;
  }

  int InvokeInstruction(const char *op_name, int offset) {
    uint8_t constant = chunk->code[offset + 1];
    uint8_t argCount = chunk->code[offset + 2];
    AppendLatestLine("%-16s (%d args) %4d->[%s]", op_name, argCount, constant,
                     chunk->constants[constant].Str().c_str());
    return offset + 3;
  }

  int ClosureInstruction(int offset) {
    offset++;
    uint8_t constant = chunk->code[offset++];
    AppendLatestLine("\n%-16s %4d->[%s]\n", "OP_CLOSURE", constant, chunk->constants[constant].Str().c_str());
    auto function = chunk->constants[constant].AsObject()->DynAs<ObjFunction>();
    int upvalue_count = chunk->code[offset++];
    for (int j = 0; j < upvalue_count; j++) {
      int isLocal = chunk->code[offset++];
      int index = chunk->code[offset++];
      AppendLatestLine("%04d      |                     %s %d\n", offset - 2, isLocal ? "local" : "upvalue", index);
    }
    return offset;
  };
  template <class... Args>
  void AppendLatestLine(Args... args) {
    int new_offset = snprintf(latest_line_buf.data() + valid_line_content_size,
                              latest_line_buf.size() - valid_line_content_size, args...);
    valid_line_content_size += new_offset;
    assert(valid_line_content_size <= latest_line_buf.size());
  }

  const char *ReleaseLatestLine() {
    valid_line_content_size = 0;
    return latest_line_buf.data();
  }

  const Chunk *chunk;
  std::vector<char> latest_line_buf;
  int valid_line_content_size = 0;
};

void DumpStack(const VM *vm) {
  std::vector<char> buf(1000);
  auto head = snprintf(buf.data(), buf.size(), "Stack: ");
  for (const Value *slot = vm->stack_; slot != vm->sp_; ++slot) {
    head += snprintf(buf.data() + head, buf.size() - head, "[ %s ]", slot->Str().c_str());
  }
  SPDLOG_DEBUG(buf.data());
}

void DumpGlobal(const VM *vm) {
  std::vector<char> buf(1000);
  auto head = snprintf(buf.data(), buf.size(), "Global: ");
  auto iter = vm->globals_.begin();
  while (iter != vm->globals_.end()) {
    head +=
        snprintf(buf.data() + head, buf.size() - head, "{ %s : %s }", iter->first->c_str(), iter->second.Str().c_str());
    ++iter;
  }
  SPDLOG_DEBUG(buf.data());
}

int DumpInstruction(const Chunk *chunk, int offset) {
  ChunkDump dump(*chunk);
  dump.DumpCodeAt(offset);
}

void DumpChunkCode(const Chunk *chunk) {
  ChunkDump dump(*chunk);
  dump.DumpAllCode();
}

void DumpChunkConstant(const Chunk *chunk) {
  ChunkDump dump(*chunk);
  dump.DumpConstant();
}

}  // namespace lox::vm

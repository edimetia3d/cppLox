//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/clox_value.h"

#include <spdlog/spdlog.h>

#include <cstdio>

#include "lox/backend/virtual_machine/bytecode/chunk.h"

namespace lox {

namespace vm {

void printValue(const Value &value, bool print_to_debug) {
#define q_printf(...)          \
  if (print_to_debug) {        \
    SPDLOG_DEBUG(__VA_ARGS__); \
  } else {                     \
    printf(__VA_ARGS__);       \
  };                           \
  break
  switch (value.Type()) {
    case ValueType::BOOL:
      q_printf((value.AsBool() ? "true" : "false"));
    case ValueType::NUMBER:
      q_printf("%f", value.AsNumber());
    case ValueType::NIL:
      q_printf("nil");
    case ValueType::OBJ:
      value.AsObj()->Print(print_to_debug);
      break;
    default:
      q_printf("Unkown types");
  }
#undef q_printf
}

static uint32_t fnv_1a(uint8_t *data, int size) {
  uint32_t new_hash = 2166136261u;
  for (int i = 0; i < size; i++) {
    new_hash ^= data[i];
    new_hash *= 16777619;
  }
  return new_hash;
}
bool lox::vm::Obj::Equal(const lox::vm::Obj *rhs) const {
  return this == rhs;  // all string are interned, so we can compare directly
}
void Obj::Print(bool print_to_debug) const {
#define q_printf(...)          \
  if (print_to_debug) {        \
    SPDLOG_DEBUG(__VA_ARGS__); \
  } else {                     \
    printf(__VA_ARGS__);       \
  };                           \
  break
  switch (type) {
    case ObjType::OBJ_STRING:
      q_printf("%s", As<ObjInternedString>()->c_str());
    case ObjType::OBJ_FUNCTION:
      q_printf("<fn %s>", As<ObjFunction>()->name.c_str());
    case ObjType::OBJ_NATIVE_FUNCTION:
      q_printf("<native fn>");
    case ObjType::OBJ_RUNTIME_FUNCTION: {
      auto p = As<ObjRuntimeFunction>();
      if (p->isClosure()) {
        q_printf("<closure %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      } else {
        q_printf("<fn %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      }
    }
    case ObjType::OBJ_UPVALUE:
      q_printf("upvalue");
    default:
      q_printf("Unknown Obj type");
  }
#undef q_printf
}
LinkList<Obj *> &Obj::AllCreatedObj() {
  static LinkList<Obj *> ret;
  return ret;
}
int &Obj::ObjCount() {
  static int value;
  return value;
}
uint32_t ObjInternedString::UpdateHash() {
  hash = fnv_1a((uint8_t *)c_str(), size());
  return hash;
}
ObjInternedString *ObjInternedString::Concat(const ObjInternedString *lhs, const ObjInternedString *rhs) {
  Buffer<char> buf;
  buf.reserve(lhs->size() + rhs->size() + 1);
  buf.push_buffer(lhs->c_str(), lhs->size());
  buf.push_buffer(rhs->c_str(), rhs->size());
  buf.push_back('\0');
  InternView view{.data = buf.data(), .hash = fnv_1a((uint8_t *)buf.data(), buf.size() - 1), .size = (buf.size() - 1)};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(std::move(buf));
  }
}
int ObjInternedString::size() const {
  return data.size() - 1;  // this is a c style str
}
ObjInternedString::ObjInternedString(const char *buf, int size) {
  data.push_buffer(buf, size);
  data.push_back('\0');
  UpdateHash();
  TryInternThis();
}
void ObjInternedString::TryInternThis() {
  if (!GetInternMap().Get(GetInternView())) {
    GetInternMap().Set(GetInternView(), this);
  } else {
    printf("Warning: Get repeated intern string\n");
  }
}
ObjInternedString::ObjInternedString(Buffer<char> &&buffer) {
  data = std::move(buffer);
  UpdateHash();
  TryInternThis();
}
ObjInternedString::InternMap &ObjInternedString::GetInternMap() {
  static InternMap interning_map(32);
  return interning_map;
}
ObjInternedString *ObjInternedString::Make(const char *data, int size) {
  InternView view{.data = data, .hash = fnv_1a((uint8_t *)data, size), .size = size};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(data, size);
  }
}
ObjInternedString::~ObjInternedString() { GetInternMap().Del(GetInternView()); }

void Obj::Destroy(Obj *obj) {
  SPDLOG_DEBUG("Object [%p] with type [%d] deleted.", obj, obj->type);
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      delete obj->As<ObjInternedString>();
      break;
    case ObjType::OBJ_FUNCTION:
      delete obj->As<ObjFunction>();
      break;
    case ObjType::OBJ_NATIVE_FUNCTION:
      delete obj->As<ObjNativeFunction>();
      break;
    case ObjType::OBJ_RUNTIME_FUNCTION:
      delete obj->As<ObjRuntimeFunction>();
      break;
    case ObjType::OBJ_UPVALUE:
      delete obj->As<ObjUpvalue>();
      break;
    default:
      printf("Destroying Unknown Type.\n");
  }
}
Obj::Obj(ObjType type) : type(type), isMarked(false) {
  SPDLOG_DEBUG("Object [%p] with type [%d] created", this, type);
  AllCreatedObj().Insert(this);
  ++ObjCount();
}
Obj::~Obj() {
  --ObjCount();
  AllCreatedObj().Delete(this);
}
void Obj::MarkReference(Obj *obj) {
  auto &gc = GC::Instance();
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      break;
    case ObjType::OBJ_FUNCTION: {
      int n = obj->As<ObjFunction>()->chunk->constants.size();
      auto p = obj->As<ObjFunction>()->chunk->constants.data();
      for (int i = 0; i < n; ++i) {
        gc.markValue(p[i]);
      }
      break;
    }
    case ObjType::OBJ_NATIVE_FUNCTION:
      break;
    case ObjType::OBJ_RUNTIME_FUNCTION: {
      gc.markObject(obj->As<ObjRuntimeFunction>()->function);
      for (int i = 0; i < obj->As<ObjRuntimeFunction>()->upvalueCount; ++i) {
        gc.markObject(obj->As<ObjRuntimeFunction>()->upvalues[i]);
      }
      break;
    }
    case ObjType::OBJ_UPVALUE: {
      auto p = obj->As<ObjUpvalue>();
      gc.markValue(*p->location);
      break;
    }
    default:
      break;
  }
}

ObjFunction::ObjFunction() { chunk = new Chunk(); }
ObjFunction::~ObjFunction() { delete chunk; }
bool Value::Equal(Value rhs) {
  if (type != rhs.type) return false;
  switch (type) {
    case ValueType::BOOL:
      return AsBool() == rhs.AsBool();
    case ValueType::NIL:
      return true;
    case ValueType::NUMBER:
      return AsNumber() == rhs.AsNumber();
    case ValueType::OBJ:
      return rhs.IsObj() && AsObj()->Equal(rhs.AsObj());
    default:
      return false;  // Unreachable.
  }
}
GC &GC::Instance() {
  static GC obj;
  return obj;
}
void GC::collectGarbage() {
  SPDLOG_DEBUG("-- gc begin");
  markRoots();
  Sweep();
  SPDLOG_DEBUG("-- gc end");
}
void GC::markValue(Value value) {
  if (value.IsObj()) markObject(value.AsObj());
}
void GC::markObject(Obj *object) {
  if (object == nullptr || object->isMarked) return;
  SPDLOG_DEBUG("%p mark ", (void *)object);
  printValue(Value(object), true);
  object->isMarked = true;
  Obj::MarkReference(object);
}
void GC::markTable(HashMap<ObjInternedString *, Value, ObjInternedString::Hash> *table) {
  auto iter = table->GetAllItem();
  while (auto entry = iter.next()) {
    markObject(entry->key);
    markValue(entry->value);
  }
}
void GC::Sweep() {
  auto &list = Obj::AllCreatedObj();
  auto p = list.Head();
  LinkList<Obj *> to_del;
  while (p) {
    if (p->val->isMarked) {
      p->val->isMarked = false;
    } else {
      to_del.Insert(p->val);
    }
    p = p->next;
  }
  p = to_del.Head();
  while (p) {
    auto next = p->next;
    Obj::Destroy(p->val);
    p = next;
  }
}
}  // namespace vm
}  // namespace lox

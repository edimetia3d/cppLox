//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H
#include <functional>
#include <map>
#include <unordered_map>

#include "lox/object/object.h"

namespace lox {
struct GC {
  // singleton
  using RootMarker = std::function<void()>;
  GC(const GC&) = delete;
  GC(GC&&) = delete;
  GC& operator=(const GC&) = delete;
  GC& operator=(GC&&) = delete;

  static GC& Instance();
  bool TryCollet();
  void Collect();
  int ForceClearAll();
  void RecursiveMark(Object* object);

  int gc_threashold = 1024;
  std::map<void*, RootMarker> markers;

 private:
  GC() = default;
  void MarkRoots();
  void Sweep();

  std::unordered_map<void*, bool> marked;
};
}  // namespace lox

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H

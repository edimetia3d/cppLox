//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOLVE_MAP_H_
#define CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOLVE_MAP_H_

#include <map>
namespace lox {
class EnvResolveMap {
 public:
  void Set(void* p, int distance) { env_reslove_map[p] = distance; }
  int Get(void* p) { return env_reslove_map[p]; }

 private:
  std::map<void*, int> env_reslove_map;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOLVE_MAP_H_

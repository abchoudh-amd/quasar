#include "quasar/physics/pic/initial_fields.hpp"

#include <stdexcept>
#include <string>

namespace quasar::pic {

namespace {

// Deck-facing names in enumerator order. This is the single definition; the
// Python validator and the bindings read it rather than restating the list, so
// adding a generator means adding an enumerator, a kernel branch and a name --
// the deck schema is untouched.
constexpr const char* kNames[] = {
    "seed_perturbation",
    "seed_tm_cavity",
    "seed_em_wave",
};

}  // namespace

std::vector<std::string> registered_pic_initial_fields() {
  return std::vector<std::string>{std::begin(kNames), std::end(kNames)};
}

PicInitialFieldKind pic_initial_field_kind(std::string_view name) {
  for (std::size_t k = 0; k < std::size(kNames); ++k) {
    if (name == kNames[k]) return static_cast<PicInitialFieldKind>(k);
  }
  throw std::invalid_argument{
      "fields.initial.type '" + std::string{name} +
      "' is not a registered PIC initial field generator"};
}

}  // namespace quasar::pic

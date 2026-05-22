#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <iostream>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  quasar::pic::EmPicConfig cfg;
  cfg.grid = quasar::Grid2D{16, 16, 1.0, 1.0};
  cfg.fdtd_order = 2;
  cfg.shape_order = 1;
  std::cout << "quasar_pic ready: " << cfg.grid.nx << "x" << cfg.grid.ny << "\n";
  return 0;
}

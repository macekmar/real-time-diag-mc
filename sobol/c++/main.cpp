#include <iostream>
#include "sobol_c.hpp"


int main() {

 auto p = i4_sobol(3, 2894);
 std::cout << p.first << std::endl;
 for (auto& i : p.second) {
  std::cout << i << std::endl;
 }

 return 0;
};


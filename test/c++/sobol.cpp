
#include <iostream>
#include "../c++/sobol.hpp"
#include <triqs/arrays.hpp>
using namespace triqs::arrays;

#include <ctime>
#include <vector>

int main()
{

 std::vector<int> dims {1,2,3,4,5,10,20};
 std::vector<int> lens {1000, 10000, 100000, 1000000, 10000000};
 double elapsed_secs;
 
 for (auto n = std::begin(lens); n != std::end(lens); n++)
 {
  for (auto d = std::begin(dims); d != std::end(dims); d++) 
  {
   clock_t begin = clock();
   auto a = i4_sobol_generate(*d, *n, 1);
   clock_t end = clock();
   elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

   std::cout << " " << elapsed_secs;
  }
  std::cout << std::endl;
 }

 std::cout << " Skip 1e7" << std::endl;

 for (auto n = std::begin(lens); n != std::end(lens); n++)
  {
   for (auto d = std::begin(dims); d != std::end(dims); d++) 
   {
    clock_t begin = clock();
    auto a = i4_sobol_generate(*d, *n, 10000000);
    clock_t end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << " " << elapsed_secs;
   }
   std::cout << std::endl;
  }

 return 0;
}
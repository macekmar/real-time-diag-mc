#include <iostream>
#include <stdint.h>
#include <bitset>
#include <strings.h>
//---------------------------------------------------------------
/*// Find first bit set. replace by intrinsic ? ffs ? portable ?
 int _ffs(uint64_t x) {
  if (x == 0) return 0;
  int r = 1;
  for (uint64_t t = 1; (x & t) == 0; ++r) t = t << 1;
  return r;
 }
*/

int main () {
  int N = 6;
  auto two_to_N = uint64_t(1) << N;
  for (uint64_t n = 0; n < two_to_N; ++n) {
   int nlc = (n < two_to_N - 1 ? ffs(~n) : N); 
   int gc = n ^ (n>>1);
   // n is the Gray code iteration number and n^(n>>1) is the Gray code (^ is the logical operator .or.)
   std::cout << gc << " == "<<   std::bitset<8>(gc) << "  "<< nlc << std::endl;
  }
}

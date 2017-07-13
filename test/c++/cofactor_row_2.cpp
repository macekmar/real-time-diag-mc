#include "../c++/configuration.hpp"

class Function {
 public:
 Function() {};
 double operator()(const int& i, const int& j) const {return i;};
};

int main() {

 Function f;
 auto matrix = det_manip<Function>(f, 10);

 for (int i=0; i<3; ++i) {
  matrix.insert_at_end(i, i);
 }

 // This matrix and its comatrix are stricly NOT invertible.
 auto matrix_save = matrix.matrix();
 //std::cout << matrix.matrix() << std::endl;
 //std::cout << matrix.inverse_matrix() << std::endl;

 // test n=3
 matrix.regenerate();
 auto cofactors = cofactor_row(matrix, 1, 3);
 //std::cout << cofactors << std::endl;
 if (not (cofactors.rank == 1 and first_dim(cofactors) == 3)) return 1;
 if (not (cofactors(0) == 0. and cofactors(1) == 0. and cofactors(2) == 0.)) return 2;
 if (matrix.matrix() != matrix_save) return 3;

 return 0;
}


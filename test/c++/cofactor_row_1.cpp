#include "../c++/configuration.hpp"

class Function {
 public:
 Function() {};
 double operator()(const int& i, const int& j) const {return i+j*j;};
};

int main() {

 Function f;
 auto matrix = det_manip<Function>(f, 10);

 for (int i=0; i<3; ++i) {
  matrix.insert_at_end(i, i);
 }

 // This matrix is stricly NOT invertible. `cofactor_row` must work in this case.
 auto matrix_save = matrix.matrix();
 //std::cout << matrix.matrix() << std::endl;
 //std::cout << matrix.inverse_matrix() << std::endl;

 // test n=2
 matrix.regenerate();
 auto cofactors = cofactor_row(matrix, 1, 2);
 //std::cout << cofactors << std::endl;
 if (not (cofactors.rank == 1 and first_dim(cofactors) == 2)) return 1;
 if (not (cofactors(0) == 6. and cofactors(1) == -8.)) return 2;
 if (matrix.matrix() != matrix_save) return 3;

 // test n=3
 matrix.regenerate();
 cofactors = cofactor_row(matrix, 1, 3);
 //std::cout << cofactors << std::endl;
 if (not (cofactors.rank == 1 and first_dim(cofactors) == 3)) return 4;
 if (not (cofactors(0) == 6. and cofactors(1) == -8. and cofactors(2) == 2.)) return 5;
 if (matrix.matrix() != matrix_save) return 6;

 return 0;
}


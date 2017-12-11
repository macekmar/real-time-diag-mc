#include "../c++/configuration.hpp"

class Function {
 public:
 Function() {};
 double operator()(const int& i, const int& j) const {return i-3+j*j;};
};

bool are_equal_det_manip(det_manip<Function> m1, det_manip<Function> m2) {
 if (m1.size() != m2.size()) return false;
 bool output = true;
 for (int i=0; i<m1.size(); ++i) {
  output = output and m1.get_x(i) == m2.get_x(i) and m1.get_y(i) == m2.get_y(i);
 }
 return output;
};

void print_det_manip(det_manip<Function> m) {
 for (int i=0; i<m.size(); ++i) std::cout << m.get_x(i) << ", ";
 std::cout << std::endl;
 for (int i=0; i<m.size(); ++i) std::cout << m.get_y(i) << ", ";
 std::cout << std::endl;
};

int main() {

 Function f;
 auto matrix = det_manip<Function>(f, 10);
 auto matrix_ref = det_manip<Function>(f, 10);

 for (int i=0; i<3; ++i) {
  matrix.insert_at_end(i+3, i);
  matrix_ref.insert_at_end(i+3, i);
 }

 // This matrix is stricly NOT invertible. `cofactor_row` has to work in this case.
 std::cout << matrix.matrix() << std::endl;
 std::cout << matrix.inverse_matrix() << std::endl;

 // test n=2
 matrix.regenerate();
 auto cofactors = cofactor_row(matrix, 1, 2);
 print_det_manip(matrix);
 std::cout << cofactors << std::endl;
 if (not are_equal_det_manip(matrix, matrix_ref)) return 10;
 if (not (cofactors.rank == 1 and first_dim(cofactors) == 2)) return 11;
 if (not (cofactors(0) == 6. and cofactors(1) == -8.)) return 12;

 // test n=3
 matrix.regenerate();
 cofactors = cofactor_row(matrix, 1, 3);
 print_det_manip(matrix);
 std::cout << cofactors << std::endl;
 if (not are_equal_det_manip(matrix, matrix_ref)) return 20;
 if (not (cofactors.rank == 1 and first_dim(cofactors) == 3)) return 21;
 if (not (cofactors(0) == 6. and cofactors(1) == -8. and cofactors(2) == 2.)) return 22;

 std::cout << "success" << std::endl;
 return 0;
}


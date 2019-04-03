#pragma once
#include <utility>
#include <vector>

int i4_bit_hi1 (int n);
int i4_bit_lo0 (int n);
std::pair<int, std::vector<double> > i4_sobol (int dim_num, int seed);
//std::vector<std::vector<double>> i4_sobol_generate (int m, int n, int skip);

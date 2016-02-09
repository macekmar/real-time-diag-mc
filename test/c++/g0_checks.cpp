// The flat band
// The non interacting dot GF's in frequency (2*2 matrix with Keldysh indices)
auto G0_dd_w_test = [&](double w) {
 dcomplex fact = Gamma * 1_j / ((w - epsilon_d) * (w - epsilon_d) + Gamma * Gamma);
 dcomplex temp2 = (nf(w - muL) + nf(w - muR)) * fact;
 auto gdc00 = 1.0 / (w - epsilon_d + Gamma * 1_j) + temp2;
 auto gdc10 = temp2 - 2.0 * fact;
 auto gdc11 = -1.0 / (w - epsilon_d - Gamma * 1_j) + temp2;
 return array<dcomplex, 2>{{gdc00, temp2}, {gdc10, gdc11}};
};

// For one lead
auto G0_dc_w_test = [&](double w) {
 auto g = G0_dd_w1(w);
 // we take Gamma_L=Gamma_R=Gamma/2
 double nl = nf(w - muL);

 auto SR = -0.5_j * Gamma;
 auto SA = 0.5_j * Gamma;
 auto SK = 0.5_j * Gamma * (4 * nl - 2);
 auto sigma_00 = (SR + SA + SK) / 2;
 auto sigma_01 = (SR - SA - SK) / 2;
 auto sigma_10 = (-SR + SA - SK) / 2;
 auto sigma_11 = (-SR - SA + SK) / 2;

 auto gdc00 = g(0, 0) * sigma_00 + g(0, 1) * sigma_10;
 auto gdc01 = -g(0, 0) * sigma_01 - g(0, 1) * sigma_11;
 auto gdc10 = g(1, 0) * sigma_00 + g(1, 1) * sigma_10;
 auto gdc11 = -g(1, 0) * sigma_01 - g(1, 1) * sigma_11;
 return array<dcomplex, 2>{{gdc00, gdc01}, {gdc10, gdc11}};
};

int main () {

// check g0_flat_band code agrees with this

}


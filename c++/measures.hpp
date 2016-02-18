#pragma once
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/clef.hpp>
#include <triqs/arrays.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

struct measure_pn_sn {

 qmc_data_t const *data; // Pointer to the MC qmc_data_t
 array<double, 1> &pn;
 array<double, 1> &sn;
 array <long,1> z{0}; //Z is an array with 1 element
 array<double,1> &pn_over_z_errors; //An array with for each value of n the error associated to pn
 array<double,1> &sn_errors;
  

 measure_pn_sn(qmc_data_t const *data, array<double, 1> *pn, array<double, 1> *sn,array<double,1> *pn_over_z_errors,array<double,1> *sn_errors) : data(data), pn(*pn), sn(*sn),pn_over_z_errors(*pn_over_z_errors),sn_errors(*sn_errors) {}

 void accumulate(dcomplex sign) {
  z(0) += 1;
  int k = data->perturbation_order;
  pn(k) += 1;
  sn(k) += real(sign);
 }

 void collect_results(mpi::communicator c) {

   int number_cores = c.size();
   std::vector<std::vector<double> >  list_pn_over_z;//An array with #core colonnes with for each line n, the values of pn over the cores
   std::vector<std::vector<double> >  list_sn;

  for (int n = 0 ; n < first_dim(pn);n++){
     array<double,1> pn_over_z_n{pn(n)/z(0)}; //The values of pn/z over the cores, array with 1 element
     array<double,1> list_temp_pn = mpi_all_gather(pn_over_z_n,c);
     list_pn_over_z.push_back(std::vector<double>(number_cores,0)); //Push a line of 0 and with size the number of cores
     for (int j = 0;j<number_cores;j++)  list_pn_over_z[n][j] = list_temp_pn(j); //Fill the line  //FIXME : do not do the loop over the cores, use directly a placeholder

     //Now for sn
     array<double,1> sn_n{sn(n)/pn(n)};
     array<double,1> list_temp_sn = mpi_all_gather(sn_n,c);     
     list_sn.push_back(std::vector<double>(number_cores,0)); //Push a line of 0 and size the number of cores
     for (int j = 0;j< number_cores;j++)list_sn[n][j] = list_temp_sn(j); //FIXME : do not do the loop, use directly a placeholder
  }

 
 //Computation of the error values 
  for(int n = 0; n < first_dim(pn);n++)
  {
    triqs::statistics::observable<double> observable_pn_over_z;
    triqs::statistics::observable<double> observable_sn;
    for(int j = 0; j <number_cores;j++) //Loop over all the cores
    {
        observable_pn_over_z << list_pn_over_z[n][j];
        observable_sn << list_sn[n][j];
    }

    pn_over_z_errors(n) = triqs::statistics::average_and_error(observable_pn_over_z).error_bar;
    sn_errors(n) = triqs::statistics::average_and_error(observable_sn).error_bar;
  } 



  long z_tot = mpi_all_reduce(z(0), c);
  pn = mpi_all_reduce(pn, c);
  sn = mpi_all_reduce(sn, c);
  for (int i = 0; i < first_dim(pn); ++i) {
   if (std::isnormal(pn(i))) sn(i) /= pn(i);
   pn(i) /= z_tot;
  }
 }
};


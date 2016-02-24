#pragma once
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/clef.hpp>
#include <triqs/arrays.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using namespace triqs::statistics;

struct measure_pn_sn {

 qmc_data_t const *data; // Pointer to the MC qmc_data_t
 
 //Creating an array of observables 
 array<observable<double>, 1 > &observable_pn;
 array<observable<double>, 1> &observable_sn;

 //With the new observable type
 measure_pn_sn(qmc_data_t const *data, array<observable<double>, 1> *observable_pn, array<observable<double>, 1> *observable_sn)
    : data(data), observable_pn(*observable_pn), observable_sn(*observable_sn) {}
 
 void accumulate(dcomplex sign) {
  
  int k = data->perturbation_order;

  //Accumulating in pn
  for(int i = 0; i < first_dim(observable_pn); i++)
  {
          if (i==k) observable_pn(i) <<1; 
          else observable_pn(i) << 0;
  }

  //Accumulating in sn
  observable_sn(k) << real(sign);
 }

 void collect_results(mpi::communicator c) {
  
  //NEW CODE
  //Putting the values from all cores together
  auto observable_pn_gathered =  array<observable<double>, 1 >(first_dim(observable_pn)); 
  auto observable_sn_gathered =  array<observable<double>, 1 >(first_dim(observable_sn)); 
  for (int i = 0;i < first_dim(observable_pn_gathered);i++) 
    {
        auto &&series_pn = observable_pn(i).get_series();
        auto &&series_sn = observable_sn(i).get_series();

        observable_pn_gathered(i) = observable<double>(mpi_all_gather(series_pn,c));
        observable_sn_gathered(i) = observable<double>(mpi_all_gather(series_sn,c));
    }
 
  //We replace pn and sn by the gathered observables, to make then the treatment of the mean values and average errors in 
  //solver_core.cpp
  observable_pn = observable_pn_gathered;
  observable_sn = observable_sn_gathered;

 }
};

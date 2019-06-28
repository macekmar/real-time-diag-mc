// DO NOT EDIT
// Generated automatically using libclang using the command :
// c++2py.py ../c++/solver_core.hpp -p -mctint_keldysh -o ctint_keldysh --moduledoc "The ctint solver" --libclang_location /usr/lib/llvm-3.8/lib/libclang-3.8.so


// --- C++ Python converter for solve_parameters_t
#include <triqs/python_tools/converters/vector.hpp>
#include <triqs/python_tools/converters/string.hpp>
#include <algorithm>

namespace triqs { namespace py_tools {

template <> struct py_converter<solve_parameters_t> {
 static PyObject *c2py(solve_parameters_t const & x) {
  PyObject * d = PyDict_New();
  PyDict_SetItemString( d, "creation_ops"            , convert_to_python(x.creation_ops));
  PyDict_SetItemString( d, "annihilation_ops"        , convert_to_python(x.annihilation_ops));
  PyDict_SetItemString( d, "extern_alphas"           , convert_to_python(x.extern_alphas));
  PyDict_SetItemString( d, "nonfixed_op"             , convert_to_python(x.nonfixed_op));
  PyDict_SetItemString( d, "interaction_start"       , convert_to_python(x.interaction_start));
  PyDict_SetItemString( d, "alpha"                   , convert_to_python(x.alpha));
  PyDict_SetItemString( d, "nb_orbitals"             , convert_to_python(x.nb_orbitals));
  PyDict_SetItemString( d, "potential"               , convert_to_python(x.potential));
  PyDict_SetItemString( d, "U"                       , convert_to_python(x.U));
  PyDict_SetItemString( d, "w_ins_rem"               , convert_to_python(x.w_ins_rem));
  PyDict_SetItemString( d, "w_dbl"                   , convert_to_python(x.w_dbl));
  PyDict_SetItemString( d, "w_shift"                 , convert_to_python(x.w_shift));
  PyDict_SetItemString( d, "max_perturbation_order"  , convert_to_python(x.max_perturbation_order));
  PyDict_SetItemString( d, "min_perturbation_order"  , convert_to_python(x.min_perturbation_order));
  PyDict_SetItemString( d, "forbid_parity_order"     , convert_to_python(x.forbid_parity_order));
  PyDict_SetItemString( d, "length_cycle"            , convert_to_python(x.length_cycle));
  PyDict_SetItemString( d, "random_seed"             , convert_to_python(x.random_seed));
  PyDict_SetItemString( d, "random_name"             , convert_to_python(x.random_name));
  PyDict_SetItemString( d, "verbosity"               , convert_to_python(x.verbosity));
  PyDict_SetItemString( d, "method"                  , convert_to_python(x.method));
  PyDict_SetItemString( d, "nb_bins"                 , convert_to_python(x.nb_bins));
  PyDict_SetItemString( d, "singular_thresholds"     , convert_to_python(x.singular_thresholds));
  PyDict_SetItemString( d, "cycles_trapped_thresh"   , convert_to_python(x.cycles_trapped_thresh));
  PyDict_SetItemString( d, "store_configurations"    , convert_to_python(x.store_configurations));
  PyDict_SetItemString( d, "preferential_sampling"   , convert_to_python(x.preferential_sampling));
  PyDict_SetItemString( d, "ps_gamma"                , convert_to_python(x.ps_gamma));
  PyDict_SetItemString( d, "sampling_model_intervals", convert_to_python(x.sampling_model_intervals));
  PyDict_SetItemString( d, "sampling_model_coeff"    , convert_to_python(x.sampling_model_coeff));
  return d;
 }

 template <typename T, typename U> static void _get_optional(PyObject *dic, const char *name, T &r, U const &init_default) {
  if (PyDict_Contains(dic, pyref::string(name)))
   r = convert_from_python<T>(PyDict_GetItemString(dic, name));
  else
   r = init_default;
 }

 template <typename T> static void _get_optional(PyObject *dic, const char *name, T &r) {
  if (PyDict_Contains(dic, pyref::string(name)))
   r = convert_from_python<T>(PyDict_GetItemString(dic, name));
  else
   r = T{};
 }

 static solve_parameters_t py2c(PyObject *dic) {
  solve_parameters_t res;
  res.creation_ops = convert_from_python<std::vector<std::tuple<orbital_t, int, timec_t, int> >>(PyDict_GetItemString(dic, "creation_ops"));
  res.annihilation_ops = convert_from_python<std::vector<std::tuple<orbital_t, int, timec_t, int> >>(PyDict_GetItemString(dic, "annihilation_ops"));
  res.extern_alphas = convert_from_python<std::vector<dcomplex>>(PyDict_GetItemString(dic, "extern_alphas"));
  _get_optional(dic, "nonfixed_op"             , res.nonfixed_op                ,false);
  res.interaction_start = convert_from_python<double>(PyDict_GetItemString(dic, "interaction_start"));
  res.alpha = convert_from_python<double>(PyDict_GetItemString(dic, "alpha"));
  res.nb_orbitals = convert_from_python<int>(PyDict_GetItemString(dic, "nb_orbitals"));
  res.potential = convert_from_python<std::tuple<std::vector<double>, std::vector<orbital_t>, std::vector<orbital_t> >>(PyDict_GetItemString(dic, "potential"));
  res.U = convert_from_python<std::vector<double>>(PyDict_GetItemString(dic, "U"));
  _get_optional(dic, "w_ins_rem"               , res.w_ins_rem                  ,1.0);
  _get_optional(dic, "w_dbl"                   , res.w_dbl                      ,0.5);
  _get_optional(dic, "w_shift"                 , res.w_shift                    ,0.0);
  _get_optional(dic, "max_perturbation_order"  , res.max_perturbation_order     ,3);
  _get_optional(dic, "min_perturbation_order"  , res.min_perturbation_order     ,0);
  _get_optional(dic, "forbid_parity_order"     , res.forbid_parity_order        ,-1);
  _get_optional(dic, "length_cycle"            , res.length_cycle               ,50);
  _get_optional(dic, "random_seed"             , res.random_seed                ,34788+928374*triqs::mpi::communicator().rank());
  _get_optional(dic, "random_name"             , res.random_name                ,"");
  _get_optional(dic, "verbosity"               , res.verbosity                  ,((triqs::mpi::communicator().rank()==0)?3:0));
  _get_optional(dic, "method"                  , res.method                     ,1);
  _get_optional(dic, "nb_bins"                 , res.nb_bins                    ,10000);
  res.singular_thresholds = convert_from_python<std::pair<double, double>>(PyDict_GetItemString(dic, "singular_thresholds"));
  _get_optional(dic, "cycles_trapped_thresh"   , res.cycles_trapped_thresh      ,100);
  _get_optional(dic, "store_configurations"    , res.store_configurations       ,0);
  _get_optional(dic, "preferential_sampling"   , res.preferential_sampling      ,false);
  _get_optional(dic, "ps_gamma"                , res.ps_gamma                   ,1.);
  res.sampling_model_intervals = convert_from_python<std::vector<std::vector<double> >>(PyDict_GetItemString(dic, "sampling_model_intervals"));
  res.sampling_model_coeff = convert_from_python<std::vector<std::vector<std::vector<double> > >>(PyDict_GetItemString(dic, "sampling_model_coeff"));
  return res;
 }

 template <typename T>
 static void _check(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (!convertible_from_python<T>(PyDict_GetItemString(dic, name), false))
   fs << "\n" << ++err << " The parameter " << name << " does not have the right type : expecting " << tname
      << " in C++, but got '" << PyDict_GetItemString(dic, name)->ob_type->tp_name << "' in Python.";
 }

 template <typename T>
 static void _check_mandatory(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (!PyDict_Contains(dic, pyref::string(name)))
   fs << "\n" << ++err << " Mandatory parameter " << name << " is missing.";
  else _check<T>(dic,fs,err,name,tname);
 }

 template <typename T>
 static void _check_optional(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (PyDict_Contains(dic, pyref::string(name))) _check<T>(dic, fs, err, name, tname);
 }

 static bool is_convertible(PyObject *dic, bool raise_exception) {
  if (dic == nullptr or !PyDict_Check(dic)) {
   if (raise_exception) { PyErr_SetString(PyExc_TypeError, "The function must be called with named arguments");}
   return false;
  }
  std::stringstream fs, fs2; int err=0;

#ifndef TRIQS_ALLOW_UNUSED_PARAMETERS
  std::vector<std::string> ks, all_keys = {"creation_ops","annihilation_ops","extern_alphas","nonfixed_op","interaction_start","alpha","nb_orbitals","potential","U","w_ins_rem","w_dbl","w_shift","max_perturbation_order","min_perturbation_order","forbid_parity_order","length_cycle","random_seed","random_name","verbosity","method","nb_bins","singular_thresholds","cycles_trapped_thresh","store_configurations","preferential_sampling","ps_gamma","sampling_model_intervals","sampling_model_coeff"};
  pyref keys = PyDict_Keys(dic);
  if (!convertible_from_python<std::vector<std::string>>(keys, true)) {
   fs << "\nThe dict keys are not strings";
   goto _error;
  }
  ks = convert_from_python<std::vector<std::string>>(keys);
  for (auto & k : ks)
   if (std::find(all_keys.begin(), all_keys.end(), k) == all_keys.end())
    fs << "\n"<< ++err << " The parameter '" << k << "' is not recognized.";
#endif

  _check_mandatory<std::vector<std::tuple<orbital_t, int, timec_t, int> >                          >(dic, fs, err, "creation_ops"            , "std::vector<std::tuple<orbital_t, int, timec_t, int> >");
  _check_mandatory<std::vector<std::tuple<orbital_t, int, timec_t, int> >                          >(dic, fs, err, "annihilation_ops"        , "std::vector<std::tuple<orbital_t, int, timec_t, int> >");
  _check_mandatory<std::vector<dcomplex>                                                           >(dic, fs, err, "extern_alphas"           , "std::vector<dcomplex>");
  _check_optional <bool                                                                            >(dic, fs, err, "nonfixed_op"             , "bool");
  _check_mandatory<double                                                                          >(dic, fs, err, "interaction_start"       , "double");
  _check_mandatory<double                                                                          >(dic, fs, err, "alpha"                   , "double");
  _check_mandatory<int                                                                             >(dic, fs, err, "nb_orbitals"             , "int");
  _check_mandatory<std::tuple<std::vector<double>, std::vector<orbital_t>, std::vector<orbital_t> >>(dic, fs, err, "potential"               , "std::tuple<std::vector<double>, std::vector<orbital_t>, std::vector<orbital_t> >");
  _check_mandatory<std::vector<double>                                                             >(dic, fs, err, "U"                       , "std::vector<double>");
  _check_optional <double                                                                          >(dic, fs, err, "w_ins_rem"               , "double");
  _check_optional <double                                                                          >(dic, fs, err, "w_dbl"                   , "double");
  _check_optional <double                                                                          >(dic, fs, err, "w_shift"                 , "double");
  _check_optional <int                                                                             >(dic, fs, err, "max_perturbation_order"  , "int");
  _check_optional <int                                                                             >(dic, fs, err, "min_perturbation_order"  , "int");
  _check_optional <int                                                                             >(dic, fs, err, "forbid_parity_order"     , "int");
  _check_optional <int                                                                             >(dic, fs, err, "length_cycle"            , "int");
  _check_optional <int                                                                             >(dic, fs, err, "random_seed"             , "int");
  _check_optional <std::string                                                                     >(dic, fs, err, "random_name"             , "std::string");
  _check_optional <int                                                                             >(dic, fs, err, "verbosity"               , "int");
  _check_optional <int                                                                             >(dic, fs, err, "method"                  , "int");
  _check_optional <int                                                                             >(dic, fs, err, "nb_bins"                 , "int");
  _check_mandatory<std::pair<double, double>                                                       >(dic, fs, err, "singular_thresholds"     , "std::pair<double, double>");
  _check_optional <int                                                                             >(dic, fs, err, "cycles_trapped_thresh"   , "int");
  _check_optional <int                                                                             >(dic, fs, err, "store_configurations"    , "int");
  _check_optional <bool                                                                            >(dic, fs, err, "preferential_sampling"   , "bool");
  _check_optional <double                                                                          >(dic, fs, err, "ps_gamma"                , "double");
  _check_mandatory<std::vector<std::vector<double> >                                               >(dic, fs, err, "sampling_model_intervals", "std::vector<std::vector<double> >");
  _check_mandatory<std::vector<std::vector<std::vector<double> > >                                 >(dic, fs, err, "sampling_model_coeff"    , "std::vector<std::vector<std::vector<double> > >");
  if (err) goto _error;
  return true;

 _error:
   fs2 << "\n---- There " << (err > 1 ? "are " : "is ") << err<< " error"<<(err >1 ?"s" : "")<< " in Python -> C++ transcription for the class solve_parameters_t\n" <<fs.str();
   if (raise_exception) PyErr_SetString(PyExc_TypeError, fs2.str().c_str());
  return false;
 }
};

}}
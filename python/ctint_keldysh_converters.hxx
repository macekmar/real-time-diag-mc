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
  PyDict_SetItemString( d, "right_input_points"     , convert_to_python(x.right_input_points));
  PyDict_SetItemString( d, "interaction_start"      , convert_to_python(x.interaction_start));
  PyDict_SetItemString( d, "measure_state"          , convert_to_python(x.measure_state));
  PyDict_SetItemString( d, "measure_times"          , convert_to_python(x.measure_times));
  PyDict_SetItemString( d, "measure_keldysh_indices", convert_to_python(x.measure_keldysh_indices));
  PyDict_SetItemString( d, "alpha"                  , convert_to_python(x.alpha));
  PyDict_SetItemString( d, "U"                      , convert_to_python(x.U));
  PyDict_SetItemString( d, "w_ins_rem"              , convert_to_python(x.w_ins_rem));
  PyDict_SetItemString( d, "w_dbl"                  , convert_to_python(x.w_dbl));
  PyDict_SetItemString( d, "w_shift"                , convert_to_python(x.w_shift));
  PyDict_SetItemString( d, "max_perturbation_order" , convert_to_python(x.max_perturbation_order));
  PyDict_SetItemString( d, "min_perturbation_order" , convert_to_python(x.min_perturbation_order));
  PyDict_SetItemString( d, "length_cycle"           , convert_to_python(x.length_cycle));
  PyDict_SetItemString( d, "random_seed"            , convert_to_python(x.random_seed));
  PyDict_SetItemString( d, "random_name"            , convert_to_python(x.random_name));
  PyDict_SetItemString( d, "max_time"               , convert_to_python(x.max_time));
  PyDict_SetItemString( d, "verbosity"              , convert_to_python(x.verbosity));
  PyDict_SetItemString( d, "method"                 , convert_to_python(x.method));
  PyDict_SetItemString( d, "nb_bins"                , convert_to_python(x.nb_bins));
  PyDict_SetItemString( d, "singular_thresholds"    , convert_to_python(x.singular_thresholds));
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
  res.right_input_points = convert_from_python<std::vector<std::tuple<x_index_t, double, int> >>(PyDict_GetItemString(dic, "right_input_points"));
  res.interaction_start = convert_from_python<double>(PyDict_GetItemString(dic, "interaction_start"));
  _get_optional(dic, "measure_state"          , res.measure_state             ,0);
  res.measure_times = convert_from_python<std::vector<double>>(PyDict_GetItemString(dic, "measure_times"));
  res.measure_keldysh_indices = convert_from_python<std::vector<int>>(PyDict_GetItemString(dic, "measure_keldysh_indices"));
  res.alpha = convert_from_python<double>(PyDict_GetItemString(dic, "alpha"));
  res.U = convert_from_python<double>(PyDict_GetItemString(dic, "U"));
  _get_optional(dic, "w_ins_rem"              , res.w_ins_rem                 ,1.0);
  _get_optional(dic, "w_dbl"                  , res.w_dbl                     ,0.5);
  _get_optional(dic, "w_shift"                , res.w_shift                   ,0.0);
  _get_optional(dic, "max_perturbation_order" , res.max_perturbation_order    ,3);
  _get_optional(dic, "min_perturbation_order" , res.min_perturbation_order    ,0);
  _get_optional(dic, "length_cycle"           , res.length_cycle              ,50);
  _get_optional(dic, "random_seed"            , res.random_seed               ,34788+928374*triqs::mpi::communicator().rank());
  _get_optional(dic, "random_name"            , res.random_name               ,"");
  _get_optional(dic, "max_time"               , res.max_time                  ,-1);
  _get_optional(dic, "verbosity"              , res.verbosity                 ,0);
  _get_optional(dic, "method"                 , res.method                    ,5);
  _get_optional(dic, "nb_bins"                , res.nb_bins                   ,10000);
  res.singular_thresholds = convert_from_python<std::pair<double, double>>(PyDict_GetItemString(dic, "singular_thresholds"));
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
  std::vector<std::string> ks, all_keys = {"right_input_points","interaction_start","measure_state","measure_times","measure_keldysh_indices","alpha","U","w_ins_rem","w_dbl","w_shift","max_perturbation_order","min_perturbation_order","length_cycle","random_seed","random_name","max_time","verbosity","method","nb_bins","singular_thresholds"};
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

  _check_mandatory<std::vector<std::tuple<x_index_t, double, int> >>(dic, fs, err, "right_input_points"     , "std::vector<std::tuple<x_index_t, double, int> >");
  _check_mandatory<double                                          >(dic, fs, err, "interaction_start"      , "double");
  _check_optional <int                                             >(dic, fs, err, "measure_state"          , "int");
  _check_mandatory<std::vector<double>                             >(dic, fs, err, "measure_times"          , "std::vector<double>");
  _check_mandatory<std::vector<int>                                >(dic, fs, err, "measure_keldysh_indices", "std::vector<int>");
  _check_mandatory<double                                          >(dic, fs, err, "alpha"                  , "double");
  _check_mandatory<double                                          >(dic, fs, err, "U"                      , "double");
  _check_optional <double                                          >(dic, fs, err, "w_ins_rem"              , "double");
  _check_optional <double                                          >(dic, fs, err, "w_dbl"                  , "double");
  _check_optional <double                                          >(dic, fs, err, "w_shift"                , "double");
  _check_optional <int                                             >(dic, fs, err, "max_perturbation_order" , "int");
  _check_optional <int                                             >(dic, fs, err, "min_perturbation_order" , "int");
  _check_optional <int                                             >(dic, fs, err, "length_cycle"           , "int");
  _check_optional <int                                             >(dic, fs, err, "random_seed"            , "int");
  _check_optional <std::string                                     >(dic, fs, err, "random_name"            , "std::string");
  _check_optional <int                                             >(dic, fs, err, "max_time"               , "int");
  _check_optional <int                                             >(dic, fs, err, "verbosity"              , "int");
  _check_optional <int                                             >(dic, fs, err, "method"                 , "int");
  _check_optional <int                                             >(dic, fs, err, "nb_bins"                , "int");
  _check_mandatory<std::pair<double, double>                       >(dic, fs, err, "singular_thresholds"    , "std::pair<double, double>");
  if (err) goto _error;
  return true;

 _error:
   fs2 << "\n---- There " << (err > 1 ? "are " : "is ") << err<< " error"<<(err >1 ?"s" : "")<< " in Python -> C++ transcription for the class solve_parameters_t\n" <<fs.str();
   if (raise_exception) PyErr_SetString(PyExc_TypeError, fs2.str().c_str());
  return false;
 }
};

}}
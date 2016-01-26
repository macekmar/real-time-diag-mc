// DO NOT EDIT
// Generated automatically using libclang using the command :
// c++2py.py ../c++/ctint.hpp -p -mpytriqs.applications.impurity_solvers.ctint_new -o ctint_new --moduledoc "The ctint solver"


// --- C++ Python converter for solve_parameters_t

namespace triqs { namespace py_tools {

template <> struct py_converter<solve_parameters_t> {
 static PyObject *c2py(solve_parameters_t const & x) {
  PyObject * d = PyDict_New();
  PyDict_SetItemString( d, "U"                     , convert_to_python(x.U));
  PyDict_SetItemString( d, "L"                     , convert_to_python(x.L));
  PyDict_SetItemString( d, "tmax"                  , convert_to_python(x.tmax));
  PyDict_SetItemString( d, "alpha"                 , convert_to_python(x.alpha));
  PyDict_SetItemString( d, "p_dbl"                 , convert_to_python(x.p_dbl));
  PyDict_SetItemString( d, "max_perturbation_order", convert_to_python(x.max_perturbation_order));
  PyDict_SetItemString( d, "min_perturbation_order", convert_to_python(x.min_perturbation_order));
  PyDict_SetItemString( d, "n_cycles"              , convert_to_python(x.n_cycles));
  PyDict_SetItemString( d, "length_cycle"          , convert_to_python(x.length_cycle));
  PyDict_SetItemString( d, "n_warmup_cycles"       , convert_to_python(x.n_warmup_cycles));
  PyDict_SetItemString( d, "random_seed"           , convert_to_python(x.random_seed));
  PyDict_SetItemString( d, "random_name"           , convert_to_python(x.random_name));
  PyDict_SetItemString( d, "max_time"              , convert_to_python(x.max_time));
  PyDict_SetItemString( d, "verbosity"             , convert_to_python(x.verbosity));
  return d;
 }

 template <typename T, typename U> static void _get_optional(PyObject *dic, const char *name, T &r, U const &init_default) {
  if (PyDict_Contains(dic, pyref::string(name)))
   r = convert_from_python<T>(PyDict_GetItemString(dic, name));
  else
   r = init_default;
 }

 static solve_parameters_t py2c(PyObject *dic) {
  solve_parameters_t res;
  res.U = convert_from_python<double>(PyDict_GetItemString(dic, "U"));
  res.L = convert_from_python<int>(PyDict_GetItemString(dic, "L"));
  res.tmax = convert_from_python<double>(PyDict_GetItemString(dic, "tmax"));
  res.alpha = convert_from_python<double>(PyDict_GetItemString(dic, "alpha"));
  _get_optional(dic, "p_dbl"                 , res.p_dbl                   , 0.5);
  _get_optional(dic, "max_perturbation_order", res.max_perturbation_order  , 3);
  _get_optional(dic, "min_perturbation_order", res.min_perturbation_order  , 0);
  res.n_cycles = convert_from_python<int>(PyDict_GetItemString(dic, "n_cycles"));
  _get_optional(dic, "length_cycle"          , res.length_cycle            , 50);
  _get_optional(dic, "n_warmup_cycles"       , res.n_warmup_cycles         , 5000);
  _get_optional(dic, "random_seed"           , res.random_seed             , 34788+928374*triqs::mpi::communicator().rank());
  _get_optional(dic, "random_name"           , res.random_name             , "");
  _get_optional(dic, "max_time"              , res.max_time                , -1);
  _get_optional(dic, "verbosity"             , res.verbosity               , ((triqs::mpi::communicator().rank()==0)?3:0));
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
  if (!PyDict_Check(dic)) {
   if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Not a python dict");}
   return false;
  }
  std::stringstream fs, fs2; int err=0;

#ifndef TRIQS_ALLOW_UNUSED_PARAMETERS
  std::vector<std::string> ks, all_keys = {"U","L","tmax","alpha","p_dbl","max_perturbation_order","min_perturbation_order","n_cycles","length_cycle","n_warmup_cycles","random_seed","random_name","max_time","verbosity"};
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

  _check_mandatory<double     >(dic, fs, err, "U"                     , "double");
  _check_mandatory<int        >(dic, fs, err, "L"                     , "int");
  _check_mandatory<double     >(dic, fs, err, "tmax"                  , "double");
  _check_mandatory<double     >(dic, fs, err, "alpha"                 , "double");
  _check_optional <double     >(dic, fs, err, "p_dbl"                 , "double");
  _check_optional <int        >(dic, fs, err, "max_perturbation_order", "int");
  _check_optional <int        >(dic, fs, err, "min_perturbation_order", "int");
  _check_mandatory<int        >(dic, fs, err, "n_cycles"              , "int");
  _check_optional <int        >(dic, fs, err, "length_cycle"          , "int");
  _check_optional <int        >(dic, fs, err, "n_warmup_cycles"       , "int");
  _check_optional <int        >(dic, fs, err, "random_seed"           , "int");
  _check_optional <std::string>(dic, fs, err, "random_name"           , "std::string");
  _check_optional <int        >(dic, fs, err, "max_time"              , "int");
  _check_optional <int        >(dic, fs, err, "verbosity"             , "int");
  if (err) goto _error;
  return true;

 _error:
   fs2 << "\n---- There " << (err > 1 ? "are " : "is ") << err<< " error"<<(err >1 ?"s" : "")<< " in Python -> C++ transcription for the class solve_parameters_t\n" <<fs.str();
   if (raise_exception) PyErr_SetString(PyExc_TypeError, fs2.str().c_str());
  return false;
 }
};

}}

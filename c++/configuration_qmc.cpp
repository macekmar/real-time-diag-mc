#include "./qmc_data.hpp"
#include "./configuration.hpp"
#include <triqs/det_manip.hpp>

/**
 * Insert a vertex at position k (before index k).
 */
void ConfigurationQMC::insert(int k, vertex_t vtx) {
 Configuration::insert(k, vtx);
 auto pt = vtx.get_up_pt();
 matrices[up].insert(k, k, pt, pt);
 pt = vtx.get_down_pt();
 matrices[down].insert(k, k, pt, pt);
};

/**
 * Insert two vertices such that `vtx1` is now at position k1 and `vtx2` at position k2.
 */
void ConfigurationQMC::insert2(int k1, int k2, vertex_t vtx1, vertex_t vtx2) {
 Configuration::insert2(k1, k2, vtx1, vtx2);
 auto pt1 = vtx1.get_up_pt();
 auto pt2 = vtx2.get_up_pt();
 matrices[up].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 pt1 = vtx1.get_down_pt();
 pt2 = vtx2.get_down_pt();
 matrices[down].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
};

/**
 * Remove the vertex at position `k`.
 */
void ConfigurationQMC::remove(int k) {
 Configuration::remove(k);
 for (auto& m : matrices) m.remove(k, k);
};

/**
 * Remove the vertices at positions `k1` and `k2`.
 */
void ConfigurationQMC::remove2(int k1, int k2) {
 Configuration::remove2(k1, k2);
 for (auto& m : matrices) m.remove2(k1, k2, k1, k2);
};

/**
 * Change vertex at position `k` into `vtx`.
 */
void ConfigurationQMC::change_vertex(int k, vertex_t vtx) {
 Configuration::change_vertex(k, vtx);
 auto pt = vtx.get_up_pt();
 matrices[up].change_row(k, pt);
 matrices[up].change_col(k, pt);
 pt = vtx.get_down_pt();
 matrices[down].change_row(k, pt);
 matrices[down].change_col(k, pt);
};

// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <thread>

#include <boost/program_options.hpp>

#include <parallel_for.hpp>
#include <polynomialspacedimension.hpp>

#include "hho-magnetostatics.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "/home/aurelio/HArDCore3D/meshes/";
std::string default_mesh = mesh_dir + "Cubic-Cells/RF_fmt/gcube_2x2x2";
std::string default_meshtype = "RF";

//------------------------------------------------------------------------------
// Free function headers
//------------------------------------------------------------------------------

/// Compute the error on the curl
double compute_curl_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::MagneticFieldType & b,
                          const Eigen::VectorXd & uI,
                          bool use_threads = true
                          );

/// Compute the error on the gradient
double compute_grad_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::PressureGradientType & grad_p,
                          const Eigen::VectorXd & pI,
                          bool use_threads = true
                          );

//------------------------------------------------------------------------------
  
int main(int argc, const char* argv[])
{

  // Program options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<std::string>(), "Set the mesh")
    ("meshtype,t", boost::program_options::value<std::string>(), "Set the mesh type (TG,MSH,RF)")
    ("degree,k", boost::program_options::value<size_t>()->default_value(1), "The polynomial degree of the HHO space")
    ("pthread,p", boost::program_options::value<bool>()->default_value(true), "Use thread-based parallelism")
    ("export-matrix,e", "Export matrix to Matrix Market format")
    ("static-condensation,s", "Perform static condensation");
  
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  // Display the help options
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  // Select the mesh
  std::string mesh_file = (vm.count("mesh") ? vm["mesh"].as<std::string>() : default_mesh);
  std::string mesh_type = (vm.count("meshtype") ? vm["meshtype"].as<std::string>() : default_meshtype);

  std::cout << FORMAT(25) << "[main] Mesh file" << mesh_file << std::endl;
  
  // Select the degree 
  size_t K = (vm.count("degree") ? vm["degree"].as<size_t>() : 0);
  std::cout << FORMAT(25) << "[main] Degree" << K << std::endl;

  // Interpolate the exact solution
  HHOMagnetostatics::CurrentDensityType f = trigonometric_f;
  HHOMagnetostatics::VectorPotentialType u = trigonometric_u;
  HHOMagnetostatics::MagneticFieldType b = trigonometric_b;
  HHOMagnetostatics::PressureType p = trigonometric_p;
  HHOMagnetostatics::PressureGradientType grad_p = trigonometric_grad_p;

  // HHOMagnetostatics::CurrentDensityType f = linear_f;
  // HHOMagnetostatics::VectorPotentialType u = linear_u;
  // HHOMagnetostatics::MagneticFieldType b = linear_b;
  // HHOMagnetostatics::PressureType p = linear_p;
  // HHOMagnetostatics::PressureGradientType grad_p = linear_grad_p;

  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  // Reorder mesh faces so that boundary faces are put at the end
  BoundaryConditions bc("D", *mesh_ptr.get());
  bc.reorder_faces();
  
  // Create problem assembler
  bool use_threads = (vm.count("pthread") ? vm["pthread"].as<bool>() : true);
  std::cout << "[main] " << (use_threads ? "Parallel execution" : "Sequential execution") << std:: endl;
  HHOMagnetostatics pb(*mesh_ptr, K, bc, use_threads);

  // Count the number of degrees of freedom
  size_t nb_cell_dofs = mesh_ptr->n_cells() * pb.zSpace().numLocalDofsCell();
  size_t nb_boundary_face_dofs = mesh_ptr->n_b_faces() * pb.zSpace().numLocalDofsFace();
  size_t nb_internal_face_dofs = mesh_ptr->n_i_faces() * pb.zSpace().numLocalDofsFace();

  std::cout << "[main] Number of cell DOFs : " << nb_cell_dofs << std::endl;
  std::cout << "[main] Number of internal face DOFS : " << nb_internal_face_dofs << std::endl;
  std::cout << "[main] Number of boundary face DOFS : " << nb_boundary_face_dofs << std::endl;

  assert( pb.linearSystemDimension<HHOMagnetostatics::Full>() == nb_cell_dofs + nb_internal_face_dofs + nb_boundary_face_dofs &&
          pb.linearSystemDimension<HHOMagnetostatics::StaticCondensation>() == nb_internal_face_dofs + nb_boundary_face_dofs );

  // Interpolate the exact solution
  std::cout << "[main] Interpolating the exact solution" << std::endl;

  Eigen::VectorXd uI = pb.xSpace().interpolate(u);
  Eigen::VectorXd pI = pb.ySpace().interpolate(p);
  Eigen::VectorXd xI = pb.zSpace().merge(std::array<Eigen::VectorXd, 2>{{uI, pI}});
  Eigen::VectorXd xFI = xI.head(nb_internal_face_dofs + nb_boundary_face_dofs);
  
  // Print the meshsize
  std::cout << "[main] Meshsize " << mesh_ptr->h_max() << std::endl;

  // Compute the interpolation error for the discrete curl and gradient
  std::cout << "[main] Computing errors" << std::endl;
  double err_curl = compute_curl_error(pb, b, uI, use_threads);
  double err_grad = compute_grad_error(pb, grad_p, pI, use_threads);
  std::cout << "[main] Curl error " << err_curl << std::endl;
  std::cout << "[main] Grad error " << err_grad << std::endl;

    std::ofstream out("results.txt");
    out << "Mesh: " << mesh_file << "\n";
    out << "Degree: " << K << "\n";
    out << "Using threads: " << (use_threads ? "true" : "false") << "\n";
    out << "MeshSize: " << mesh_ptr->h_max() << "\n";
    out << "NbCells: " << mesh_ptr->n_cells() << "\n";
    out << "NbFaces: " << mesh_ptr->n_faces() << "\n";
    out << "MeshReg: " << mesh_ptr->regularity() << "\n";
    out << "NoDofs: " << nb_internal_face_dofs << "\n";

  if (vm.count("static-condensation")) { 
    std::cout << "[main] Assemble statically condensed linear system" << std::endl;
    
    // Assemble the problem
    pb.assembleLinearSystem<HHOMagnetostatics::StaticCondensation>(f, u, p);

    // Handle strongly enforced Dirichlet boundary conditions and solve the system
    std::cout << "[main] Enforcing boundary conditions" << std::endl;
    Eigen::SparseMatrix<double> Ah = pb.systemMatrix().topLeftCorner(nb_internal_face_dofs, nb_internal_face_dofs);
    Eigen::VectorXd bh = pb.systemVector() - pb.systemMatrix().rightCols(nb_boundary_face_dofs) * xFI.tail(nb_boundary_face_dofs);

    std::cout << "[main] Residual " << (bh.head(nb_internal_face_dofs) - Ah * xI.head(nb_internal_face_dofs)).norm() << std::endl;
    out << "Residual: " << (bh.head(nb_internal_face_dofs) - Ah * xI.head(nb_internal_face_dofs)).norm() << "\n";

    std::cout << "[main] Solving the linear system using a direct solver" << std::endl;
    Eigen::SparseLU<HHOMagnetostatics::SystemMatrixType> solver;
    solver.compute(Ah);
    if (solver.info() != Eigen::Success) {
    std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
    }
    Eigen::VectorXd xh = solver.solve(bh.head(nb_internal_face_dofs));

    // Compute error
    auto err_hI = xh - xI.head(nb_internal_face_dofs);
    double err_en = std::sqrt(err_hI.transpose() * Ah * err_hI);
    std::cout << "[main] Energy error " << err_en << std::endl;
    out << "EnergyError: " << err_en << "\n";

    // Export to Matrix Market format if requested  
    if (vm.count("export-matrix")) {
      std::cout << "[main] Exporting matrix to Matrix Market format" << std::endl;
      saveMarket(Ah, "A_hho-magnetostatics.mtx");
      saveMarket(bh, "b_hho-magnetostatics.mtx");
      saveMarket(xh, "x_hho-magnetostatics.mtx");
    }
  } else {
    std::cout << "[main] Assemble full linear system" << std::endl;
    
    pb.assembleLinearSystem<HHOMagnetostatics::Full>(f, u, p);
    Eigen::VectorXd rh = pb.systemVector() - pb.systemMatrix() * xI;
    std::cout << "[main] Residual "
              << rh.head(nb_internal_face_dofs).norm() + rh.tail(nb_cell_dofs).norm()
              << std::endl;
  }

    // --------------------------------------------------------------------------
    //                     Creates .txt file with data and results
    // --------------------------------------------------------------------------
    out << std::flush;
    out.close();

  std::cout << "[main] Done" << std::endl;
  
  return 0;
}

//------------------------------------------------------------------------------
// HHOMagnetostatics
//------------------------------------------------------------------------------

HHOMagnetostatics::HHOMagnetostatics(const Mesh & mesh, size_t K, const BoundaryConditions & bc, bool use_threads, std::ostream & output)
  : m_mesh(mesh),
    m_degree(K),
    m_bc(bc),
    m_use_threads(use_threads),
    m_output(output),
    m_xspace(mesh, K + 1, use_threads),
    m_yspace(mesh, K + 1, use_threads),
    m_zspace(typename ZSpace::CartesianFactorsArray{{ &m_xspace, &m_yspace }})
{
  m_output << "[HHOMagnetostatics] Initializing" << std::endl;
}

//------------------------------------------------------------------------------

void HHOMagnetostatics::_assemble_statically_condensed(
                                                       size_t iT,
                                                       const LocalContribution & lsT,
                                                       std::list<Eigen::Triplet<double> > & my_triplets,
                                                       Eigen::VectorXd & my_rhs
                                                       )
{
  const Cell & T = *mesh().cell(iT);

  size_t n_cell_dofs = zSpace().numLocalDofsCell();
  size_t n_cell_boundary_dofs = zSpace().numLocalDofsCellBoundary(T);

  const auto & AT_FF = lsT.first.topLeftCorner(n_cell_boundary_dofs, n_cell_boundary_dofs);
  const auto & AT_TF = lsT.first.bottomLeftCorner(n_cell_dofs, n_cell_boundary_dofs);
  const auto & AT_FT = lsT.first.topRightCorner(n_cell_boundary_dofs, n_cell_dofs);
  const auto & AT_TT = lsT.first.bottomRightCorner(n_cell_dofs, n_cell_dofs);

  Eigen::FullPivLU<Eigen::MatrixXd> fact_AT_TT(AT_TT);

  if (!fact_AT_TT.isInvertible()) {
    std::cerr << "[HHOMagnetostatics] ERROR: Non-invertible local matrix found in static condensation" << std::endl;
    exit(1);
  }

  Eigen::MatrixXd AT = AT_FF - AT_FT * fact_AT_TT.solve(AT_TF);
  Eigen::VectorXd bT = -AT_FT * fact_AT_TT.solve(lsT.second.tail(n_cell_dofs));

  auto I_pT = zSpace().globalCellBoundaryDofIndices(T);

  for (size_t i = 0; i < n_cell_boundary_dofs; i++) {
    my_rhs(I_pT(i)) += bT(i);
    for (size_t j = 0; j < n_cell_boundary_dofs; j++) {
      my_triplets.push_back( Eigen::Triplet<double>(I_pT(i), I_pT(j), AT(i, j)) );
    } // for j

  } // for i
}

//------------------------------------------------------------------------------

void HHOMagnetostatics::_assemble_full(
                                       size_t iT,
                                       const LocalContribution & lsT,
                                       std::list<Eigen::Triplet<double> > & my_triplets,
                                       Eigen::VectorXd & my_rhs
                                       )
{
  const Cell & T = *mesh().cell(iT);

  const Eigen::MatrixXd & AT = lsT.first;
  const Eigen::VectorXd & bT = lsT.second;
  
  // Create the vector of DOF indices
  auto I_boundary_T = zSpace().globalCellBoundaryDofIndices(T);
  auto I_internal_T = zSpace().globalCellDofIndices(T);

  typename CartesianProductHHOSpace<2>::DofIndexVector I_T(zSpace().dimensionCell(T));
  I_T.head(T.n_faces() * zSpace().numLocalDofsFace()) = I_boundary_T;
  I_T.tail(zSpace().numLocalDofsCell()) = I_internal_T;
  
  // Assemble
  for (size_t i = 0; i < zSpace().dimensionCell(T); i++) {
    my_rhs(I_T(i)) += bT(i);
    for (size_t j = 0; j < zSpace().dimensionCell(T); j++) {
      my_triplets.push_back( Eigen::Triplet<double>(I_T(i), I_T(j), AT(i,j)) );
    } // for j
  } // for i
}

//------------------------------------------------------------------------------

HHOMagnetostatics::LocalContribution
HHOMagnetostatics::_compute_local_contribution(
                                               size_t iT,
                                               const CurrentDensityType & f,
                                               const VectorPotentialType & u,
                                               const PressureType & p 
                                               )
{
  const Cell & T = *mesh().cell(iT);

  //------------------------------------------------------------------------------
  // Problem matrix
  
  Eigen::MatrixXd AT = Eigen::MatrixXd::Zero(zSpace().dimensionCell(T), zSpace().dimensionCell(T));

  // Stabilization contribution for aT and dT
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);

    double hF = F.diam();
    Eigen::Vector3d nTF = T.face_normal(iF);
    
    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * xSpace().degree());

    auto xspace_basis_T_quad = evaluate_quad<Function>::compute(xSpace().cellBasis(iT), quad_2kpo_F);
    // Take the tangent component of element basis functions
    std::transform(
                   xspace_basis_T_quad.data(),
                   xspace_basis_T_quad.data() + xspace_basis_T_quad.num_elements(),
                   xspace_basis_T_quad.data(),
                   [&nTF](const Eigen::Vector3d & x) -> Eigen::Vector3d { return nTF.cross(x.cross(nTF)); }
                   );
    auto xspace_basis_F_quad = evaluate_quad<Function>::compute(xSpace().faceBasis(F.global_index()), quad_2kpo_F);

    AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, F, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsFace())
      += 1./hF * compute_gram_matrix(xspace_basis_F_quad, quad_2kpo_F);
    Eigen::MatrixXd hFmo_xspace_MFT = 1./hF * compute_gram_matrix(xspace_basis_F_quad, xspace_basis_T_quad, quad_2kpo_F);
    AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsCell())
      -= hFmo_xspace_MFT;
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, F, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsFace())
      -= hFmo_xspace_MFT.transpose();      
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsCell())
      += 1./hF * compute_gram_matrix(xspace_basis_T_quad, quad_2kpo_F);

    // dT
    auto yspace_basis_T_quad = evaluate_quad<Function>::compute(ySpace().cellBasis(iT), quad_2kpo_F);
    auto yspace_basis_F_quad = evaluate_quad<Function>::compute(ySpace().faceBasis(F.global_index()), quad_2kpo_F);

    AT.block(zSpace().localOffset(T, F, 1), zSpace().localOffset(T, F, 1), ySpace().numLocalDofsFace(), ySpace().numLocalDofsFace())
      += hF * compute_gram_matrix(yspace_basis_F_quad, quad_2kpo_F);
    Eigen::MatrixXd hF_yspace_MFT = hF * compute_gram_matrix(yspace_basis_F_quad, yspace_basis_T_quad, quad_2kpo_F);
    AT.block(zSpace().localOffset(T, F, 1), zSpace().localOffset(T, 1), ySpace().numLocalDofsFace(), ySpace().numLocalDofsCell())
      -= hF_yspace_MFT;
    AT.block(zSpace().localOffset(T, 1), zSpace().localOffset(T, F, 1), ySpace().numLocalDofsCell(), ySpace().numLocalDofsFace())
      -= hF_yspace_MFT.transpose();      
    AT.block(zSpace().localOffset(T, 1), zSpace().localOffset(T, 1), ySpace().numLocalDofsCell(), ySpace().numLocalDofsCell())
      += hF * compute_gram_matrix(yspace_basis_T_quad, quad_2kpo_F);
  } // for iF

  // Consistency term for aT
  Eigen::MatrixXd aT_consistency = xSpace().curlReconstruction(iT).transpose() * xSpace().curlReconstructionRHS(iT);
  AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsCell())
    += aT_consistency.block(xSpace().localOffset(T), xSpace().localOffset(T), xSpace().numLocalDofsCell(), xSpace().numLocalDofsCell());
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, F, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsFace())
      += aT_consistency.block(xSpace().localOffset(T), xSpace().localOffset(T, F), xSpace().numLocalDofsCell(), xSpace().numLocalDofsFace());
    AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsCell())
      += aT_consistency.block(xSpace().localOffset(T, F), xSpace().localOffset(T), xSpace().numLocalDofsFace(), xSpace().numLocalDofsCell());

    for (size_t jF = 0; jF < T.n_faces(); jF++) {
      const Face & G = *T.face(jF);
      
      AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, G, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsFace())
        += aT_consistency.block(xSpace().localOffset(T, F), xSpace().localOffset(T, G), xSpace().numLocalDofsFace(), xSpace().numLocalDofsFace());
    } // for jF
  } // for iF

  //------------------------------------------------------------------------------
  // bT
  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * xSpace().degree());
  auto xspace_basis_T_quad = evaluate_quad<Function>::compute(xSpace().cellBasis(iT), quad_2kpo_T);
  // auto yspace_gradient_basis_T_quad = evaluate_quad<Function>::compute(ySpace().gradientReconstructionBasis(iT), quad_2kpo_T);
  // Eigen::MatrixXd bT = compute_gram_matrix(xspace_basis_T_quad, yspace_gradient_basis_T_quad, quad_2kpo_T) * ySpace().gradientReconstruction(iT);
  const Eigen::MatrixXd & bT = ySpace().gradientReconstructionRHS(iT);

  AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, 1), xSpace().numLocalDofsCell(), ySpace().numLocalDofsCell())
    += bT.block(0, ySpace().localOffset(T), xSpace().numLocalDofsCell(), ySpace().numLocalDofsCell());
  AT.block(zSpace().localOffset(T, 1), zSpace().localOffset(T, 0), ySpace().numLocalDofsCell(), xSpace().numLocalDofsCell())
    -= bT.block(0, ySpace().localOffset(T), xSpace().numLocalDofsCell(), ySpace().numLocalDofsCell()).transpose();
  
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, F, 1), xSpace().numLocalDofsCell(), ySpace().numLocalDofsFace())
      += bT.block(0, ySpace().localOffset(T, F), xSpace().numLocalDofsCell(), ySpace().numLocalDofsFace());
    AT.block(zSpace().localOffset(T, F, 1), zSpace().localOffset(T, 0), ySpace().numLocalDofsFace(), xSpace().numLocalDofsCell())
      -= bT.block(0, ySpace().localOffset(T, F), xSpace().numLocalDofsCell(), ySpace().numLocalDofsFace()).transpose();
  } // for iF

  //------------------------------------------------------------------------------
  // Source term
  Eigen::VectorXd rT = Eigen::VectorXd::Zero(zSpace().dimensionCell(T));
  rT.segment(zSpace().localOffset(T, 0), xSpace().numLocalDofsCell()) = integrate(f, xspace_basis_T_quad, quad_2kpo_T);  
  
  return std::make_pair(AT, rT);
}

//------------------------------------------------------------------------------
// Free functions
//------------------------------------------------------------------------------

double compute_curl_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::MagneticFieldType & b,
                          const Eigen::VectorXd & uI,
                          bool use_threads
                          )
{
  Eigen::ArrayXd err_curl = Eigen::ArrayXd::Zero(pb.mesh().n_cells());

  std::function<void(size_t, size_t)> compute_curl_errors
    = [&pb, &b, &uI, &err_curl](size_t start, size_t end)->void
      {
        for (size_t iT = 0; iT < pb.mesh().n_cells(); iT++) {
          const Cell & T = *pb.mesh().cell(iT);

          Eigen::VectorXd CT_uI_T = pb.xSpace().curlReconstruction(iT) * pb.xSpace().restrictCell(iT, uI);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * pb.xSpace().degree());
          auto curl_basis_quad = evaluate_quad<Function>::compute(pb.xSpace().curlReconstructionBasis(iT), quad_2k_T);

          double err_curl_T = 0.;
          for (size_t iqn = 0; iqn < quad_2k_T.size(); iqn++) {
            const double & w_iqn = quad_2k_T[iqn].w;
            
            const Eigen::Vector3d & b_iqn = b(quad_2k_T[iqn].vector());
            Eigen::Vector3d CT_uI_T_iqn = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < pb.xSpace().curlReconstructionBasis(iT).dimension(); i++) {
              CT_uI_T_iqn += CT_uI_T(i) * curl_basis_quad[i][iqn];
            } // for i
            
            err_curl_T += w_iqn * (b_iqn - CT_uI_T_iqn).squaredNorm();
          } // for iqn

          err_curl[iT] = err_curl_T;
        } // for iT
      };
  parallel_for(pb.mesh().n_cells(), compute_curl_errors, use_threads);
  
  return std::sqrt(err_curl.sum());
}

//------------------------------------------------------------------------------

double compute_grad_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::PressureGradientType & grad_p,
                          const Eigen::VectorXd & pI,
                          bool use_threads
                          )
{
  Eigen::ArrayXd err_grad = Eigen::ArrayXd::Zero(pb.mesh().n_cells());
  
  std::function<void(size_t, size_t)> compute_grad_errors
    = [&pb, &grad_p, &pI, &err_grad](size_t start, size_t end)->void
      {
        for (size_t iT = 0; iT < pb.mesh().n_cells(); iT++) {
          const Cell & T = *pb.mesh().cell(iT);

          Eigen::VectorXd GT_pI_T = pb.ySpace().gradientReconstruction(iT) * pb.ySpace().restrictCell(iT, pI);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * pb.ySpace().degree());
          auto grad_basis_quad = evaluate_quad<Function>::compute(pb.ySpace().gradientReconstructionBasis(iT), quad_2k_T);

          double err_grad_T = 0.;
          for (size_t iqn = 0; iqn < quad_2k_T.size(); iqn++) {
            const double & w_iqn = quad_2k_T[iqn].w;
            
            const Eigen::Vector3d & grad_p_iqn = grad_p(quad_2k_T[iqn].vector());
            Eigen::Vector3d GT_pI_T_iqn = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < pb.ySpace().gradientReconstructionBasis(iT).dimension(); i++) {
              GT_pI_T_iqn += GT_pI_T(i) * grad_basis_quad[i][iqn];
            } // for i
            
            err_grad_T += w_iqn * (grad_p_iqn - GT_pI_T_iqn).squaredNorm();
          } // for iqn

          err_grad[iT] = err_grad_T;
        } // for iT
      };
  parallel_for(pb.mesh().n_cells(), compute_grad_errors, use_threads);
  
  return std::sqrt(err_grad.sum());
}

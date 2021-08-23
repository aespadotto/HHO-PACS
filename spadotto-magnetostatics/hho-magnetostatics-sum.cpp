// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <thread>
#include <cmath>

#include <GetPot>

#include <parallel_for.hpp>
#include <polynomialspacedimension.hpp>

#include "hho-magnetostatics-sum.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "/mnt/c/Users/Aurelio/Desktop/Ubuntu/HArDCore3D/meshes/";
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
                          const Eigen::VectorXd & bI,
                          bool use_threads = true
                          );

/// Compute the error on the gradient
double compute_grad_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::PressureGradientType & grad_p,
                          const Eigen::VectorXd & pI,
                          const Eigen::VectorXd & gradpI,
                          bool use_threads = true
                          );

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{

// parse with GetPot
GetPot cl(argc, argv);
const std::string mesh_file = cl.follow("/mnt/c/Users/Aurelio/Desktop/HHO-muse/meshes/Cubic-Cells/RF_fmt/gcube_2x2x2", 2, "-m", "--mesh");
const std::string mesh_type = cl.follow ("RF", 2, "-t","--meshtype");
const size_t K = cl.follow (1, 2, "-k","--degree");
const bool use_threads = cl.follow (true, 2, "-p", "--pthread");
const bool static_condensation = cl.follow (true, 2, "-s","--static-condensation");
const bool use_linear  = cl.follow(false, 2, "-l", "--linear-solution");
const std::string bc_type = cl.follow ("inB", 2, "-bc", "--boundary-condition");

std::cout << FORMAT(25) << "[main] Degree" << K << std::endl;

  HHOMagnetostatics::CurrentDensityType f;
  HHOMagnetostatics::VectorPotentialType u;
  HHOMagnetostatics::MagneticFieldType b;
  HHOMagnetostatics::PressureType p;
  HHOMagnetostatics::PressureGradientType grad_p;

  if (bc_type=="inH") {
      f = trigonometric_f_H;
      u = trigonometric_u_H;
      b = trigonometric_b_H;
      p = trigonometric_p_H;
      grad_p = trigonometric_grad_p_H;
  }
  else if(bc_type=="Mixed"){
      f = trigonometric_f_M;
      u = trigonometric_u_M;
      b = trigonometric_b_M;
      p = trigonometric_p_M;
      grad_p = trigonometric_grad_p_M;
  }
  else if (bc_type=="inB" && !use_linear) {
      f = trigonometric_f;
      u = trigonometric_u;
      b = trigonometric_b;
      p = trigonometric_p;
      grad_p = trigonometric_grad_p; }

  else {
      f = linear_f;
      u = linear_u;
      b = linear_b;
      p = linear_p;
      grad_p = linear_grad_p; }

  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  // Reorder mesh faces so that boundary faces are put at the end
  // here the flag "D" doesn't really stand for Dirichlet boundary condition,
  // but it is still necessary to distinguish border faces

  BoundaryConditions bc(bc_type=="Mixed"? "M0":"D", *mesh_ptr.get());
  bc.reorder_faces();

  // Create problem assembler
  //bool use_threads = (vm.count("pthread") ? vm["pthread"].as<bool>() : true);
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
  Eigen::VectorXd bI = pb.xSpace().interpolate_curl(b);
  Eigen::VectorXd gradpI = pb.ySpace().interpolate_grad(grad_p);
  Eigen::VectorXd xI = pb.zSpace().merge(std::array<Eigen::VectorXd, 2>{{uI, pI}});
  Eigen::VectorXd xFI = xI.head(nb_internal_face_dofs + nb_boundary_face_dofs);

  // Print the meshsize
  std::cout << "[main] Meshsize " << mesh_ptr->h_max() << std::endl;

  // Compute the interpolation error for the discrete curl and gradient
  std::cout << "[main] Computing errors" << std::endl;
  double err_curl = compute_curl_error(pb, b, uI, bI, use_threads);
  std::cout << "[main] Curl error " << err_curl << std::endl;
  double err_grad = compute_grad_error(pb, grad_p, pI, gradpI, use_threads);
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

if (static_condensation)
{
    std::cout << "[main] Assemble statically condensed linear system" << std::endl;

    // Assemble the problem (matrices differ based on boundary conditions)
    pb.assembleLinearSystem<HHOMagnetostatics::StaticCondensation>(*mesh_ptr, pb.zSpace(), pb.ySpace(), f, u, p, bc_type, b);
    Eigen::VectorXd xh;                /// vector of unknowns
    Eigen::SparseMatrix<double> Ah;    /// system matrix

   if (bc_type=="inB") {
       // Handle strongly enforced Dirichlet boundary conditions and solve the system
       std::cout << "[main] Enforcing boundary conditions in B" << std::endl;

       Ah = pb.systemMatrix().topLeftCorner(nb_internal_face_dofs, nb_internal_face_dofs);
       Eigen::VectorXd bh =
               pb.systemVector() - pb.systemMatrix().rightCols(nb_boundary_face_dofs) * xFI.tail(nb_boundary_face_dofs);

       std::cout << "[main] Residual " << (bh.head(nb_internal_face_dofs) - Ah * xI.head(nb_internal_face_dofs)).norm()
                 << std::endl;
       out << "Residual: " << (bh.head(nb_internal_face_dofs) - Ah * xI.head(nb_internal_face_dofs)).norm() << "\n";

       std::cout << "[main] Solving the linear system using a direct solver" << std::endl;
       Eigen::SparseLU<HHOMagnetostatics::SystemMatrixType> solver;
       solver.compute(Ah);
       if (solver.info() != Eigen::Success) {
           std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
           if (solver.info() == Eigen::NumericalIssue) std::cout << "[main] Kind of error: numerical" << std::endl;
           if (solver.info() == Eigen::InvalidInput) std::cout << "[main] Kind of error: invalid input" << std::endl;
       }
       xh = solver.solve(bh.head(nb_internal_face_dofs));
   }
   else if (bc_type=="Mixed") {
       // Handle strongly enforced Dirichlet boundary conditions and solve the system
       std::cout << "[main] Enforcing boundary conditions in B on Gamma_B" << std::endl;
       size_t nb_unknowns_mixed = nb_internal_face_dofs+ nb_boundary_face_dofs - bc.n_dir_faces()*pb.zSpace().numLocalDofsFace();
       size_t nb_dirichlet_dofs = bc.n_dir_faces()*pb.zSpace().numLocalDofsFace();
       Ah = pb.systemMatrix().topLeftCorner(nb_unknowns_mixed, nb_unknowns_mixed);
       Eigen::VectorXd bh =
               pb.systemVector() - pb.systemMatrix().rightCols(nb_dirichlet_dofs) * xFI.tail(nb_dirichlet_dofs);

       std::cout << "[main] Residual " << (bh.head(nb_unknowns_mixed) - Ah * xI.head(nb_unknowns_mixed)).norm()
                 << std::endl;
       out << "Residual: " << (bh.head(nb_unknowns_mixed) - Ah * xI.head(nb_unknowns_mixed)).norm() << "\n";

       std::cout << "[main] Solving the linear system using a direct solver" << std::endl;
       Eigen::SparseLU<HHOMagnetostatics::SystemMatrixType> solver;
       solver.compute(Ah);
       if (solver.info() != Eigen::Success) {
           std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
           if (solver.info() == Eigen::NumericalIssue) std::cout << "[main] Kind of error: numerical" << std::endl;
           if (solver.info() == Eigen::InvalidInput) std::cout << "[main] Kind of error: invalid input" << std::endl;
       }
       xh = solver.solve(bh.head(nb_unknowns_mixed));
   }
   else if (bc_type=="inH") {
       Ah = pb.systemMatrix();
       Eigen::VectorXd bh = pb.systemVector();
       std::cout<<"[main] Boundary conditions in H (no enforcing needed)"<<std::endl;
       std::cout << "[main] Residual " << (bh - Ah * xFI).norm() << std::endl;
       out << "Residual: " << (bh - Ah * xFI).norm() << "\n";

       std::cout << "[main] Solving the linear system using a direct solver" << std::endl;
       Eigen::SparseLU<HHOMagnetostatics::SystemMatrixType> solver;
       solver.compute(Ah);
       if (solver.info() != Eigen::Success) {
           std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
           if (solver.info() == Eigen::NumericalIssue) std::cout << "[main] Kind of error: numerical" << std::endl;
           if (solver.info() == Eigen::InvalidInput) std::cout << "[main] Kind of error: invalid input" << std::endl;
       }
       xh = solver.solve(bh);
   }

    /*
    std::cout << "[main] Solving the linear system using BiCGSTAB" << std::endl;
    Eigen::BiCGSTAB<HHOMagnetostatics::SystemMatrixType> solver;
    solver.compute(Ah);
    Eigen::VectorXd xh = solver.solve(bh.head(nb_internal_face_dofs));
    */

    // in the following the number of dofs of the solution may vary according to bc type
    size_t nb_sol_dofs;
    if (bc_type=="inB") nb_sol_dofs = (nb_internal_face_dofs);
    if (bc_type=="inH") nb_sol_dofs = (nb_internal_face_dofs+nb_boundary_face_dofs);
    if (bc_type=="Mixed") nb_sol_dofs = (nb_internal_face_dofs+nb_boundary_face_dofs-bc.n_dir_faces()*pb.zSpace().numLocalDofsFace());
    // Compute energy error
    auto err_hI = xh - xFI.head(nb_sol_dofs);
    std::cout<<"[debug] Euclidean Norm Error "<<err_hI.norm()<<std::endl;
    double err_en = std::sqrt(err_hI.transpose() * Ah * err_hI);
    std::cout << "[main] Energy error " << err_en << std::endl;
    out << "EnergyError: " << err_en << "\n";

    // Compute L2 error
    //Reconstruct cell unknowns from static condensation solution
    std::cout<<"[main] Calculating L2 Error "<<std::endl;
    Eigen::VectorXd face_unknowns = Eigen::VectorXd::Zero(pb.zSpace().dimension());
    Eigen::VectorXd cell_unknowns = Eigen::VectorXd::Zero(pb.zSpace().dimension());
    if (bc_type=="inB") {
        face_unknowns.head(nb_internal_face_dofs) = xh;
        face_unknowns.segment(nb_internal_face_dofs, nb_boundary_face_dofs) = xFI.tail(nb_boundary_face_dofs);
    }
    else if (bc_type=="inH"){
        face_unknowns = xh;
    }

    else if (bc_type=="Mixed"){
        face_unknowns.head(nb_sol_dofs)= xh;
        face_unknowns.segment(nb_sol_dofs, bc.n_dir_faces()) = xFI.tail(bc.n_dir_faces());
    }

    std::function <void (size_t, size_t)> cell_unknown_rec =
                  [&f,&u,&p,&pb, &face_unknowns, &cell_unknowns, &bc_type, &b] (size_t start, size_t end)->void {
                    //std::cout<<"zig"<<start<<std::endl;
                    for (size_t iT = start; iT<end; iT++){
                          //std::cout<<"zag"<<std::endl;
                          pb._reconstruct_cell_unknowns(iT, face_unknowns, cell_unknowns, f, u, p, bc_type, b);
                     }
                  };

/*  std::cout<<"[main] Make some checks"<<std::endl;
    std::cout<<face_unknowns.rows()<<" = "<<nb_internal_face_dofs + nb_boundary_face_dofs<<std::endl;
    std::cout<<cell_unknowns.rows()<<" = "<<nb_cell_dofs<<std::endl;
    std::cout<<pb.zSpace().numLocalDofsFace()<<" "<<pb.zSpace().numLocalDofsCell()<<std::endl;
    std::cout<<mesh_ptr->n_faces()<<" "<<mesh_ptr->n_cells()<<std::endl; */
    //std::cout<<"Tot Cell: "<<mesh_ptr->n_cells()<<std::endl;
    //std::cout<<"Threads: "<<std::thread::hardware_concurrency()<<std::endl;


  /*  for (int iT = 0; iT<mesh_ptr->n_cells(); ++iT) {
      std::cout<<iT<<" faces: "<<(mesh_ptr->cell(iT))->n_faces()<<std::endl;
      //const Cell & T = *pb.mesh().cell(iT);
      pb._reconstruct_cell_unknowns(iT,face_unknowns, cell_unknowns, f,u,p);
              //auto xxx = pb._compute_local_contribution(iT,f,u,p);
    //const Cell & T = *mesh_ptr->cell(iT);
  }*/

    parallel_for (mesh_ptr->n_cells(), cell_unknown_rec, use_threads);
    std::cout<<"[main] Reconstruct Cell Unknowns "<<std::endl;
      // determine error

    Eigen::VectorXd err_cell = cell_unknowns.tail(nb_cell_dofs)-xI.tail(nb_cell_dofs);
      // if I can assume the ordering is right (check) I can assume the unknowns vector as a row of a family, and calculate square of  L2 norm by taking out
      // the trace of Gram Matrix. Notice that the vector contains both u and p cell unknowns, so extract

    double l2_err_squared = 0.;

    std::function <void(size_t,size_t)> add_local_l2_err =
    [&pb, &l2_err_squared, K, &err_cell] (size_t start, size_t end)->void {
      for (size_t iT =start; iT<end; iT++) {
          const Cell & T = *(pb.mesh().cell(iT));

          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * K);
          Eigen::VectorXd err_cell_u = err_cell.segment(iT*pb.numLocalDofsCell(), pb.xSpace().numLocalDofsCell());
          Eigen::VectorXd err_cell_p = err_cell.segment(iT*pb.numLocalDofsCell()+pb.xSpace().numLocalDofsCell(), pb.ySpace().numLocalDofsCell());
          auto err_quad_u = evaluate_quad<Function>::compute(
            Family (pb.xSpace().cellBasis(iT) , err_cell_u.transpose()),
            quad_2k_T);
          auto err_quad_p = evaluate_quad<Function>::compute(
            Family (pb.ySpace().cellBasis(iT) , err_cell_p.transpose()),
            quad_2k_T);

          Eigen::MatrixXd err_mat_u = compute_gram_matrix(
                                                        err_quad_u,
                                                        quad_2k_T
                                                       );
          Eigen::MatrixXd err_mat_p = compute_gram_matrix(
                                                        err_quad_p,
                                                        quad_2k_T
                                                       );
         // focus temporarily on u only
         l2_err_squared+=err_mat_u.trace();
      }
    };
    std::cout<<"[main] Building L2 Error "<<std::endl;
    parallel_for (mesh_ptr->n_cells(), add_local_l2_err, use_threads);
    out<< "L2Error: "<<std::sqrt(l2_err_squared)<<"\n";
    std::cout<<"L2 Error: "<<std::sqrt(l2_err_squared)<<std::endl;
    std::cout<<"Euclidean L2 Error: "<<std::sqrt(err_cell.transpose()*err_cell)<<std::endl;


    // Export to Matrix Market format if requested
/*    if (vm.count("export-matrix")) {
      std::cout << "[main] Exporting matrix to Matrix Market format" << std::endl;
      saveMarket(Ah, "A_hho-magnetostatics.mtx");
      saveMarket(bh, "b_hho-magnetostatics.mtx");
      saveMarket(xh, "x_hho-magnetostatics.mtx");

    }
*/
  } else {
    std::cout << "[main] Assemble full linear system" << std::endl;
/*
    pb.assembleLinearSystem<HHOMagnetostatics::Full>(f, u, p);

    Eigen::SparseMatrix<double> Ah (nb_internal_face_dofs+nb_cell_dofs, nb_internal_face_dofs+nb_cell_dofs);
    (Ah.leftCols(nb_internal_face_dofs)).topRows(nb_internal_face_dofs) = (pb.systemMatrix().leftCols(nb_internal_face_dofs)).topRows(nb_internal_face_dofs);
    (Ah.rightCols(nb_internal_face_dofs)).topRows(nb_cell_dofs)= (pb.systemMatrix().rightCols(nb_internal_face_dofs)).topRows(nb_cell_dofs);
    (Ah.leftCols(nb_internal_face_dofs)).bottomRows(nb_cell_dofs) = (pb.systemMatrix().leftCols(nb_internal_face_dofs)).bottomRows(nb_cell_dofs);
    (Ah.rightCols(nb_cell_dofs)).bottomRows(nb_cell_dofs) = (pb.systemMatrix().rightCols(nb_cell_dofs)).bottomRows(nb_cell_dofs);

    Eigen::VectorXd bh = Eigen::VectorXd::Zero(nb_internal_face_dofs+nb_cell_dofs);
    bh.head(nb_internal_face_dofs)=pb.systemVector().head(nb_internal_face_dofs);
    bh.tail(nb_cell_dofs)=pb.systemVector().tail(nb_cell_dofs);

    std::cout<<"[main] Enforcing Boundary Conditions"<<std::endl;
    bh.head(nb_internal_face_dofs)-=(pb.systemMatrix().middleCols(nb_internal_face_dofs, nb_boundary_face_dofs)).topRows(nb_internal_face_dofs)
                                   *xI.segment(nb_internal_face_dofs, nb_boundary_face_dofs);
    bh.tail(nb_cell_dofs)-=(pb.systemMatrix().middleCols(nb_internal_face_dofs, nb_boundary_face_dofs)).bottomRows(nb_cell_dofs)
                          *xI.segment(nb_internal_face_dofs, nb_boundary_face_dofs);

    Eigen::VectorXd rh = bh - Ah.leftCols(nb_internal_face_dofs)*xI.head(nb_internal_face_dofs)-Ah.rightCols(nb_cell_dofs)*xI.tail(nb_cell_dofs);
    std::cout << "[main] Residual "
              << rh.head(nb_internal_face_dofs).norm() + rh.tail(nb_cell_dofs).norm()
              << std::endl;

    std::cout << "[main] Solving the linear system using a direct solver" << std::endl;
    Eigen::SparseLU<HHOMagnetostatics::SystemMatrixType> solver;
    solver.compute(Ah);
    if (solver.info() != Eigen::Success) {
    std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
    if (solver.info()== Eigen::NumericalIssue) std::cout<<"[main] Kind of error: numerical"<<std::endl;
    if (solver.info()== Eigen::InvalidInput)   std::cout<<"[main] Kind of error: invalid input"<<std::endl;
    }
    Eigen::VectorXd xh = solver.solve(bh);

    // Compute error
    Eigen::VectorXd xxI = Eigen::VectorXd::Zero(nb_internal_face_dofs+nb_cell_dofs);
    xxI.head(nb_internal_face_dofs) = xI.head(nb_internal_face_dofs);
    xxI.tail(nb_cell_dofs) = xI.tail (nb_cell_dofs);
    auto err_hI = xh - xxI;
    double err_en = std::sqrt(err_hI.transpose() * Ah* err_hI);
    std::cout << "[main] Energy error " << err_en << std::endl;
    out << "EnergyError: " << err_en << "\n";

*/


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
                                                       Eigen::VectorXd & my_rhs,
                                                       const std::string bc_type
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

/////////////// new method
void HHOMagnetostatics::_reconstruct_cell_unknowns(
                               size_t iT,
                               Eigen::VectorXd & face_unknowns,
                               Eigen::VectorXd & cell_unknowns,
                               const CurrentDensityType & f,
                               const VectorPotentialType & u,
                               const PressureType & p,
                               const std::string bc_type,
                               const MagneticFieldType & b
                             ) {

    
    const Cell & T = *this->mesh().cell(iT);

    LocalContribution lsT = this->_compute_local_contribution(iT, f, u, p, bc_type, b);
    

    // defining useful sizes
    size_t n_cell_dofs = zSpace().numLocalDofsCell();
    size_t n_cell_boundary_dofs = zSpace().numLocalDofsCellBoundary(T);
    ZSpace::DofIndexVector globalCellDofIndices = zSpace().globalCellDofIndices(T);
    ZSpace::DofIndexVector globalCellBoundaryDofIndices = zSpace().globalCellBoundaryDofIndices(T);

           //const auto & AT_FF = lsT.first.topLeftCorner(n_cell_boundary_dofs, n_cell_boundary_dofs);
           const auto & AT_TF = lsT.first.bottomLeftCorner(n_cell_dofs, n_cell_boundary_dofs);
           //const auto & AT_FT = lsT.first.topRightCorner(n_cell_boundary_dofs, n_cell_dofs);
           const auto & AT_TT = lsT.first.bottomRightCorner(n_cell_dofs, n_cell_dofs);

   Eigen::FullPivLU<Eigen::MatrixXd> fact_AT_TT(AT_TT);

  if (!fact_AT_TT.isInvertible()) {
      std::cerr << "[HHOMagnetostatics] ERROR: Non-invertible local matrix found in static condensation" << std::endl;
      exit(1);
     }


  Eigen::VectorXd rhsT = lsT.second.tail(n_cell_dofs);
  Eigen::VectorXd local_face_unknowns =Eigen::VectorXd::Zero (n_cell_boundary_dofs);

  //std::cout<<globalCellBoundaryDofIndices.rows()<<" = "<<n_cell_boundary_dofs<<std::endl;
  for (size_t i = 0; i<n_cell_boundary_dofs; ++i) {
    local_face_unknowns (i) = face_unknowns(globalCellBoundaryDofIndices(i));
  }
  //std::cout<<"[_reconstruct_cell_unknowns] Retrieve face unknowns from global vector"<<std::endl;


  Eigen::VectorXd rhs = rhsT -AT_TF*local_face_unknowns;
  //std::cout<<"[_reconstruct_cell_unknowns] Computing rhs vector for local rec problem "<<std::endl;
  Eigen::VectorXd local_cell_unknowns         = fact_AT_TT.solve(rhs);
  //std::cout<<"[_reconstruct_cell_unknowns] Solve local recostruction problem "<<std::endl;
      for (size_t i = 0; i<n_cell_dofs;++i) {
      cell_unknowns(globalCellDofIndices(i)) = local_cell_unknowns(i);
      
  }
}


//------------------------------------------------------------------------------

void HHOMagnetostatics::_assemble_full(
                                       size_t iT,
                                       const LocalContribution & lsT,
                                       std::list<Eigen::Triplet<double> > & my_triplets,
                                       Eigen::VectorXd & my_rhs,
                                       const std::string bc_type
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
////////////////// modified method
HHOMagnetostatics::LocalContribution
HHOMagnetostatics::_compute_local_contribution(
                                               size_t iT,
                                               const CurrentDensityType & f,
                                               const VectorPotentialType & u,
                                               const PressureType & p,
                                               std::string bc_type,
                                               const MagneticFieldType & b
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
    // Take the tangent component of element basis functions (actually the following isn't used. Projection onto face space is already tangent)
    /*std::transform(
                   xspace_basis_T_quad.data(),
                   xspace_basis_T_quad.data() + xspace_basis_T_quad.num_elements(),
                   xspace_basis_T_quad.data(),
                   [&nTF](const Eigen::Vector3d & x) -> Eigen::Vector3d { return nTF.cross(x.cross(nTF)); }
                   );*/
    auto xspace_basis_F_quad = evaluate_quad<Function>::compute(xSpace().faceBasis(F.global_index()), quad_2kpo_F);
    // here I must project onto face space
    AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, F, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsFace())
      += 1./hF * compute_gram_matrix(xspace_basis_F_quad, quad_2kpo_F);
    Eigen::MatrixXd hFmo_xspace_MFT = 1./hF * compute_gram_matrix(xspace_basis_F_quad, quad_2kpo_F)*xSpace().cell_to_face_projection(iT,iF);
    AT.block(zSpace().localOffset(T, F, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsFace(), xSpace().numLocalDofsCell())
      -= hFmo_xspace_MFT;
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, F, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsFace())
      -= hFmo_xspace_MFT.transpose();
    // I must project cell basis onto face space
    AT.block(zSpace().localOffset(T, 0), zSpace().localOffset(T, 0), xSpace().numLocalDofsCell(), xSpace().numLocalDofsCell())
      += 1./hF * xSpace().cell_to_face_projection(iT,iF).transpose()*compute_gram_matrix(xspace_basis_F_quad, quad_2kpo_F)*xSpace().cell_to_face_projection(iT,iF);

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
  // here second factor was curlReconstructionRHS(iT). moreover added then added central factor
  QuadratureRule quad_2K_T  = generate_quadrature_rule (T, 2*xSpace().degree());
  Eigen::MatrixXd aT_consistency = xSpace().curlReconstruction(iT).transpose()
                                   * compute_gram_matrix(evaluate_quad<Function>::compute(xSpace().curlReconstructionBasis(iT),quad_2K_T),
                                                         quad_2K_T)
                                   * xSpace().curlReconstruction(iT);
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
  // here I uncommented 2 lines: changed definition of bT
  auto yspace_gradient_basis_T_quad = evaluate_quad<Function>::compute(ySpace().gradientReconstructionBasis(iT), quad_2kpo_T);
  Eigen::MatrixXd bT = compute_gram_matrix(xspace_basis_T_quad, yspace_gradient_basis_T_quad, quad_2kpo_T) * ySpace().gradientReconstruction(iT);
  // const Eigen::MatrixXd & bT = ySpace().gradientReconstructionRHS(iT);

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
  // the higher the exactness the better
  Eigen::VectorXd rT = Eigen::VectorXd::Zero(zSpace().dimensionCell(T));
  QuadratureRule quad_very_exact = generate_quadrature_rule(T, 10);
  rT.segment(zSpace().localOffset(T, 0), xSpace().numLocalDofsCell()) =
          integrate(f,
                    evaluate_quad<Function>::compute(xSpace().cellBasis(iT), quad_very_exact),
                    quad_very_exact);

  // Additional boundary term when non homogeneous inH boundary condition is valid for a portion of the boundary
  if (bc_type!="inB") {
            for (size_t jF = 0; jF < T.n_faces(); jF++) {
          const Face &F = *T.face(jF);
          VectorRd nTF = F.normal();
          if (m_bc.type(F)=="neu"){
              QuadratureRule quad_F = generate_quadrature_rule(F, 10);
              auto xspace_basis_F_quad = evaluate_quad<Function>::compute(xSpace().faceBasis(F.global_index()), quad_F);
              std::transform(
                      xspace_basis_F_quad.data(),
                      xspace_basis_F_quad.data() + xspace_basis_F_quad.num_elements(),
                      xspace_basis_F_quad.data(),
                      [&nTF](const Eigen::Vector3d & x) -> Eigen::Vector3d { return nTF.cross(x.cross(nTF)); }
              );
              std::function<VectorRd(const VectorRd&)> phi_cross_n = [&b, nTF] (const VectorRd &x){return nTF.cross(b(x));};
              rT.segment(zSpace().localOffset(T, F, 0), xSpace().numLocalDofsFace()) =
                      integrate (phi_cross_n, xspace_basis_F_quad, quad_F);
          }
      }
  }
  //-------------------------------------------------------------------------------
  // Additional contribution to lhs in case of boundary conditions on H: (p,q)_T
  if (bc_type !="inB"){

      QuadratureRule quad_T = generate_quadrature_rule(T, 10);
      auto eval_Y_quad_T= evaluate_quad<Function>::compute (ySpace().cellBasis(iT), quad_T);
      AT.block(zSpace().localOffset(T, 1), zSpace().localOffset(T, 1), ySpace().numLocalDofsCell(), ySpace().numLocalDofsCell())
              +=  compute_gram_matrix(eval_Y_quad_T, eval_Y_quad_T, quad_T);
  }

  return std::make_pair(AT, rT);
}

//------------------------------------------------------------------------------
// Free functions
//------------------------------------------------------------------------------
///////////// new free function
double compute_curl_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::MagneticFieldType & b,
                          const Eigen::VectorXd & uI,
                          const Eigen::VectorXd & bI,
                          bool use_threads
                          )
{
  Eigen::ArrayXd err_curl = Eigen::ArrayXd::Zero(pb.mesh().n_cells());
  size_t curl_basis_dim = pb.xSpace().curlReconstructionBasis(0).dimension();
  std::function<void(size_t, size_t)> compute_curl_errors
    = [&pb, &b, &uI, &bI,&err_curl,&curl_basis_dim](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT< end; iT++) {
          const Cell & T = *pb.mesh().cell(iT);

          Eigen::VectorXd CT_uI_T = pb.xSpace().curlReconstruction(iT) * pb.xSpace().restrictCell(iT, uI);
          Eigen::VectorXd INT_pI_T = bI.segment(iT*curl_basis_dim,curl_basis_dim);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * pb.xSpace().degree());
          auto curl_basis_quad = evaluate_quad<Function>::compute(pb.xSpace().curlReconstructionBasis(iT), quad_2k_T);
          double err_curl_T = 0.;
          for (size_t iqn = 0; iqn < quad_2k_T.size(); iqn++) {
            const double & w_iqn = quad_2k_T[iqn].w;

            Eigen::Vector3d CT_uI_T_iqn = Eigen::Vector3d::Zero();
            Eigen::Vector3d INT_bI_T_iqn = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < pb.xSpace().curlReconstructionBasis(iT).dimension(); i++) {
              CT_uI_T_iqn += CT_uI_T(i) * curl_basis_quad[i][iqn];
              INT_bI_T_iqn+=INT_pI_T(i) * curl_basis_quad[i][iqn];
            } // for i

            err_curl_T += w_iqn * (INT_bI_T_iqn - CT_uI_T_iqn).squaredNorm();
          } // for iqn

          err_curl[iT] = err_curl_T;
        } // for iT
      };
  parallel_for(pb.mesh().n_cells(), compute_curl_errors, use_threads);

  return std::sqrt(err_curl.sum());
}

//------------------------------------------------------------------------------
/////////////new free function
double compute_grad_error(
                          const HHOMagnetostatics & pb,
                          const HHOMagnetostatics::PressureGradientType & grad_p,
                          const Eigen::VectorXd & pI,
                          const Eigen::VectorXd & gradpI,
                          bool use_threads
                          )
{
  Eigen::ArrayXd err_grad = Eigen::ArrayXd::Zero(pb.mesh().n_cells());
  size_t grad_basis_dim = pb.ySpace().gradientReconstructionBasis(0).dimension();

  std::function<void(size_t, size_t)> compute_grad_errors
    = [&pb, &grad_p, &pI, &gradpI, &err_grad, &grad_basis_dim](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *pb.mesh().cell(iT);

          Eigen::VectorXd GT_pI_T = pb.ySpace().gradientReconstruction(iT) * pb.ySpace().restrictCell(iT, pI);
          Eigen::VectorXd INT_gradpI_T = gradpI.segment(iT*grad_basis_dim, grad_basis_dim);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * pb.ySpace().degree());
          auto grad_basis_quad = evaluate_quad<Function>::compute(pb.ySpace().gradientReconstructionBasis(iT), quad_2k_T);

          double err_grad_T = 0.;
          for (size_t iqn = 0; iqn < quad_2k_T.size(); iqn++) {
            const double & w_iqn = quad_2k_T[iqn].w;

            // this way we evaluate the exact gradient. To check commutation property
            // try evaluating its interpolate on gradient reconstruction space
            //const Eigen::Vector3d & grad_p_iqn = grad_p(quad_2k_T[iqn].vector());
            Eigen::Vector3d GT_pI_T_iqn = Eigen::Vector3d::Zero();
            Eigen::Vector3d INT_gradpI_T_iqn = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < pb.ySpace().gradientReconstructionBasis(iT).dimension(); i++) {
              GT_pI_T_iqn += GT_pI_T(i) * grad_basis_quad[i][iqn];
              INT_gradpI_T_iqn += INT_gradpI_T(i) * grad_basis_quad[i][iqn];
            } // for i

            err_grad_T += w_iqn * (INT_gradpI_T_iqn - GT_pI_T_iqn).squaredNorm();
          } // for iqn

          err_grad[iT] = err_grad_T;
        } // for iT
      };
  parallel_for(pb.mesh().n_cells(), compute_grad_errors, use_threads);

  return std::sqrt(err_grad.sum());
}

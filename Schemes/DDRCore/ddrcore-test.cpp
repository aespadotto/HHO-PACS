// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/math/constants/constants.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <basis.hpp>

#include <ddrcore.hpp>
#include <parallel_for.hpp>

#include "l2projection.hpp"
#include "ddrcore-test.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;
using namespace HArDCore3D::Tests;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "../HArDCore3D/meshes/";
std::string default_mesh = mesh_dir + "Voro-small-0/RF_fmt/voro-2";
std::string default_meshtype = "RF";

//------------------------------------------------------------------------------

int main(int argc, const char* argv[])
{
  // Program options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<std::string>(), "Set the mesh")
    ("meshtype,t", boost::program_options::value<std::string>(), "Set the mesh type (TG,MSH,RF)")
    ("degree,k", boost::program_options::value<size_t>()->default_value(1), "The polynomial degree of the sequence")
    ("pthread,p", boost::program_options::value<bool>()->default_value(true), "Use thread-based parallelism");

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
  size_t K = vm["degree"].as<size_t>();

  std::cout << FORMAT(25) << "[main] Degree" << K << std::endl;
  
  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  // Create DDR core
  bool use_threads = (vm.count("pthread") ? vm["pthread"].as<bool>() : true);
  std::cout << "[main] " << (use_threads ? "Parallel execution" : "Sequential execution") << std:: endl;
  DDRCore ddr_core(*mesh_ptr, K, use_threads);

  //------------------------------------------------------------------------------
  // Check the approximation properties of the element spaces
  //------------------------------------------------------------------------------
  
  Eigen::MatrixXd errors_cells = Eigen::MatrixXd::Zero(5, mesh_ptr->n_cells());

  auto compute_errors_cells
    = [&mesh_ptr, &ddr_core, K, &errors_cells](size_t start, size_t end) {
	for (size_t iT = start; iT < end; iT++) {
	  // Retrieve local element bases
	  const DDRCore::CellBases & bases_T = ddr_core.cellBases(iT);

	  // Generate the quadrature rules required for the following tests
	  const Cell & T = *mesh_ptr->cell(iT);
	  QuadratureRule qr_2k_T = generate_quadrature_rule(T, 2 * K);
	  QuadratureRule qr_2kpo_T = generate_quadrature_rule(T, 2 * (K + 1));

	  // Check the approximation properties of Pk+1(T)
	  assert( bases_T.Polykpo );
   
	  L2Projection<DDRCore::PolyBasisCellType>
	    l2_Pkpo_T(*bases_T.Polykpo, qr_2kpo_T, evaluate_quad<Function>::compute(*bases_T.Polykpo, qr_2kpo_T));
	  auto l2_Pkpo_q = l2_Pkpo_T.compute(q);
	  errors_cells(0, iT) += l2_Pkpo_T.squared_error(q, l2_Pkpo_q);

	  // Check the approximation properties of Gk-1(T)
	  if (bases_T.Golykmo) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Cell>::Goly(K - 1) > 0 );
	     
	    QuadratureRule qr_2kmo_T = generate_quadrature_rule(T, 2 * (K - 1));
	    L2Projection<DDRCore::GolyBasisCellType>
	      l2_Gkmo_T(*bases_T.Golykmo, qr_2kmo_T, evaluate_quad<Function>::compute(*bases_T.Golykmo, qr_2kmo_T));
	    auto l2_Gkmo_grad_q = l2_Gkmo_T.compute(grad_q);
	    errors_cells(1, iT) += l2_Gkmo_T.squared_error(grad_q, l2_Gkmo_grad_q);
	  }
    
	  // Check the approximation properties of GOk(T)
	  if (bases_T.GolyComplk) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Cell>::GolyCompl(K) > 0 );

	    L2Projection<DDRCore::GolyComplBasisCellType>
	      l2_GOk_T(*bases_T.GolyComplk, qr_2k_T, evaluate_quad<Function>::compute(*bases_T.GolyComplk, qr_2k_T));
	    auto l2_GOk_grad_q = l2_GOk_T.compute(grad_q);
	    errors_cells(2, iT) += squared_l2_norm(l2_GOk_grad_q, qr_2k_T, l2_GOk_T.basisQuad());
	  }
    
	  // Check the approximation properties of Rk-1(T)
	  if (bases_T.Rolykmo) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Cell>::Roly(K - 1) > 0 );

	    QuadratureRule qr_2kmo_T = generate_quadrature_rule(T, 2 * (K - 1));
	    L2Projection<DDRCore::RolyBasisCellType>
	      l2_Rkmo_T(*bases_T.Rolykmo, qr_2kmo_T, evaluate_quad<Function>::compute(*bases_T.Rolykmo, qr_2kmo_T));
	    auto l2_Rkmo_curl_v = l2_Rkmo_T.compute(curl_v);
	    errors_cells(3, iT) += l2_Rkmo_T.squared_error(curl_v, l2_Rkmo_curl_v);
	  }
    
	  // Check the approximation properties of ROk(T)
	  if (bases_T.RolyComplk) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Cell>::RolyCompl(K) > 0 );

	    L2Projection<DDRCore::RolyComplBasisCellType>
	      l2_ROk_T(*bases_T.RolyComplk, qr_2k_T, evaluate_quad<Function>::compute(*bases_T.RolyComplk, qr_2k_T));
	    auto l2_ROk_curl_v = l2_ROk_T.compute(curl_v);
	    errors_cells(4, iT) += squared_l2_norm(l2_ROk_curl_v, qr_2k_T, l2_ROk_T.basisQuad());
	  }
    
	} // for iT
      };

  std::cout << "[main] Checking the approximation properties of the DDR spaces on cells" << std::endl;
  parallel_for(mesh_ptr->n_cells(), compute_errors_cells, use_threads);

  // Compute the errors by summing elementary contributions
  double err_l2_Pkpo_Th = errors_cells.row(0).sum();
  double err_l2_Gkmo_Th = errors_cells.row(1).sum();
  double err_l2_GOk_Th  = errors_cells.row(2).sum();
  double err_l2_Rkmo_Th = errors_cells.row(3).sum();
  double err_l2_ROk_Th  = errors_cells.row(4).sum();

  // Print projection errors
  std::cout << FORMAT(25) << "[main] err_l2_Pkpo_Th" << std::scientific << std::sqrt(err_l2_Pkpo_Th) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_Gkmo_Th" << std::scientific << std::sqrt(err_l2_Gkmo_Th) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_GOk_Th" << std::scientific << std::sqrt(err_l2_GOk_Th) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_Rkmo_Th" << std::scientific << std::sqrt(err_l2_Rkmo_Th) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_ROk_Th" << std::scientific << std::sqrt(err_l2_ROk_Th) << std::endl;

  //------------------------------------------------------------------------------
  // Check the approximation properties of the face spaces
  //------------------------------------------------------------------------------
  
  Eigen::MatrixXd errors_faces = Eigen::MatrixXd::Zero(3, mesh_ptr->n_faces());

  auto compute_errors_faces
    = [&mesh_ptr, &ddr_core, K, &errors_faces](size_t start, size_t end) {
	for (size_t iF = start; iF < end; iF++) {

	  // Retrieve local face bases
	  const DDRCore::FaceBases & bases_F = ddr_core.faceBases(iF);

	  // Create the function grad q \times nF
	  const Face & F = *mesh_ptr->face(iF);
	  const Eigen::Vector3d & nF = F.normal();
	  auto grad_q_cross_nF  = [nF](const Eigen::Vector3d & x)->Eigen::Vector3d { return grad_q(x).cross(nF); };
	  
	  // Generate the quadrature rules required for the following tests
	  QuadratureRule qr_2kpo_F = generate_quadrature_rule(F, 2 * (K + 1));

	  // Check the approximation properties of Pk+1(F)
	  assert( bases_F.Polykpo );
	  L2Projection<DDRCore::PolyBasisFaceType>
	    l2_Pkpo_F(*bases_F.Polykpo, qr_2kpo_F, evaluate_quad<Function>::compute(*bases_F.Polykpo, qr_2kpo_F));
	  auto l2_Pkpo_q = l2_Pkpo_F.compute(q);
	  errors_faces(0, iF) += l2_Pkpo_F.squared_error(q, l2_Pkpo_q);

	  // Check the approximation properties of Rk-1(F)
	  if (bases_F.Rolykmo) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Face>::Roly(K - 1) > 0);

	    QuadratureRule qr_2kmo_F = generate_quadrature_rule(F, 2 * (K - 1));
	    L2Projection<DDRCore::RolyBasisFaceType>
	      l2_Rkmo_F(*bases_F.Rolykmo, qr_2kmo_F, evaluate_quad<Function>::compute(*bases_F.Rolykmo, qr_2kmo_F));
	    auto l2_Rkmo_grad_q_cross_nF = l2_Rkmo_F.compute(grad_q_cross_nF);
	    errors_faces(1, iF) += l2_Rkmo_F.squared_error(grad_q_cross_nF, l2_Rkmo_grad_q_cross_nF);
	  }

	  // Check the approximation properties of ROk(F)
	  if (bases_F.RolyComplk) {
	    // Make sure that the space should exist
	    assert( PolynomialSpaceDimension<Face>::RolyCompl(K) > 0);

	    QuadratureRule qr_2k_F = generate_quadrature_rule(F, 2 * K);
	    L2Projection<DDRCore::RolyComplBasisFaceType>
	      l2_ROk_F(*bases_F.RolyComplk, qr_2k_F, evaluate_quad<Function>::compute(*bases_F.RolyComplk, qr_2k_F));
	    auto l2_ROk_grad_q_cross_nF = l2_ROk_F.compute(grad_q_cross_nF);
	    errors_faces(2, iF) += squared_l2_norm(l2_ROk_grad_q_cross_nF, qr_2k_F, l2_ROk_F.basisQuad());
	  }
	}       
      };

  std::cout << "[main] Checking the approximation properties of the DDR spaces on faces" << std::endl;
  parallel_for(mesh_ptr->n_faces(), compute_errors_faces, use_threads);

  // Compute the errors by summing elementary contributions
  double err_l2_Pkpo_Fh = errors_faces.row(0).sum();
  double err_l2_Rkmo_Fh = errors_faces.row(1).sum();
  double err_l2_ROk_Fh  = errors_faces.row(2).sum();
  
  // Print projection errors
  std::cout << FORMAT(25) << "[main] err_l2_Pkpo_Fh" << std::scientific << std::sqrt(err_l2_Pkpo_Fh) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_Rkmo_Fh" << std::scientific << std::sqrt(err_l2_Rkmo_Fh) << std::endl;
  std::cout << FORMAT(25) << "[main] err_l2_ROk_Fh" << std::scientific << std::sqrt(err_l2_ROk_Fh) << std::endl;

  //------------------------------------------------------------------------------
  // Check the approximation properties of the edge spaces
  //------------------------------------------------------------------------------

  Eigen::VectorXd errors_edges(Eigen::VectorXd::Zero(mesh_ptr->n_edges()));

  auto compute_errors_edges
    = [&mesh_ptr, &ddr_core, K, &errors_edges](size_t start, size_t end) {
	for (size_t iE = start; iE < end; iE++) {

	  // Retrieve local face bases
	  const DDRCore::EdgeBases & bases_E = ddr_core.edgeBases(iE);

	  // Create the function v\cdot tE
	  const Edge & E = *mesh_ptr->edge(iE);
	  Eigen::Vector3d tE = E.tangent();
	  auto v_cdot_tE  = [tE](const Eigen::Vector3d & x)->double { return v(x).dot(tE); };
	  
	  // Generate the quadrature rules required for the following tests
	  QuadratureRule qr_2kpo_E = generate_quadrature_rule(E, 2 * (K + 1));

	  // Check the approximation properties of Pk+1(E)
	  assert( bases_E.Polykpo );
	  L2Projection<DDRCore::PolyEdgeBasisType>
	    l2_Pkpo_E(*bases_E.Polykpo, qr_2kpo_E, evaluate_quad<Function>::compute(*bases_E.Polykpo, qr_2kpo_E));
	  auto l2_Pkpo_v_cdot_tE = l2_Pkpo_E.compute(v_cdot_tE);
	  errors_edges(iE) += l2_Pkpo_E.squared_error(v_cdot_tE, l2_Pkpo_v_cdot_tE);
	} // for iE
      };
  
  std::cout << "[main] Checking the approximation properties of the DDR spaces on edges" << std::endl;
  parallel_for(mesh_ptr->n_edges(), compute_errors_edges, use_threads);

  // Compute the errors by summing elementary contributions
  double err_l2_Pkpo_Eh = errors_edges.sum();
  
  // Print projection errors
  std::cout << FORMAT(25) << "[main] err_l2_Pkpo_Eh" << std::scientific << std::sqrt(err_l2_Pkpo_Eh) << std::endl;

  //------------------------------------------------------------------------------
  // Check the face orientations
  //------------------------------------------------------------------------------
  // We check that the normals are ok when counting the orientation
  double check_oriented_normals = 0.0;
  for (Cell* cell : mesh_ptr->get_cells()){
    for (size_t i=0; i < cell->n_faces(); i++){
      Face* face = cell->face(i);
      check_oriented_normals += std::abs( (cell->face_normal(i) - cell->face_orientation(i)*face->normal()).norm() );
    }
  }
  std::cout << "Check orientation normals: " << check_oriented_normals << "\n";

  //------------------------------------------------------------------------------
  // Check the edge normals and orientations
  //------------------------------------------------------------------------------
  double check_norm_edge_normals = 0;
  for (auto& f : mesh_ptr->get_faces()){
    for (size_t i = 0; i < f->n_edges(); i++){
      check_norm_edge_normals += std::abs( (f->edge_normal(i)).norm() - 1.0);
    }
  }
  std::cout << "Check norm of edge normals: " << check_norm_edge_normals << "\n";
  
  // Test orientation: we check that omega_{TF_1}omega_{F_1E}+omega_{TF_2}\omega_{F_2E}=0 for all T and E
  double check_edge_orientation = 0;
  for (auto& T : mesh_ptr->get_cells()){
    std::vector<double> check_edge(T->n_edges(),0.0);
    for (size_t i=0; i < T->n_faces(); i++){
      Face* F = T->face(i);
      double omegaTF = T->face_orientation(i);
      for (size_t ilE=0; ilE < F->n_edges(); ilE++){
        size_t iE = F->edge(ilE)->global_index();
        // where is the edge E in T?
        size_t k = 0;
        while (k < T->n_edges() && iE != T->edge(k)->global_index()){
          k++;
        }
        if (iE != T->edge(k)->global_index()){
          std::cout << "Ouch, that's bad!\n";
          exit(EXIT_FAILURE);
        }
        check_edge[k] += omegaTF * F->edge_orientation(ilE);
      }
    }
    for (size_t k = 0; k<T->n_edges(); k++){
      check_edge_orientation += std::abs(check_edge[k]);
    }
  }
  std::cout << "Check edge orientation: " << check_edge_orientation << "\n";

  std::cout << "[main] Done" << std::endl;
  exit(0);
}


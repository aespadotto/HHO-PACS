// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/math/constants/constants.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <xgrad.hpp>
#include <parallel_for.hpp>

#include "xgrad-test.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "../../meshes/";
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
    ("pthread,p", boost::program_options::value<bool>()->default_value(true), "Use thread-based parallelism")
    ("function,f", boost::program_options::value<int>()->default_value(0), "Select the function to interpolate");

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

  // Create XGrad
  XGrad x_grad(ddr_core, use_threads);

  // Select and interpolate a scalar function
  int function_to_interpolate = (vm.count("function") ? vm["function"].as<int>() : 0);  
  std::function<double(const Eigen::Vector3d&)> q;
  std::function<Eigen::Vector3d(const Eigen::Vector3d&)> grad_q;
  switch (function_to_interpolate) {
  case 0:
    std::cout << "[main] Interpolating constant function" << std::endl;
    q = constant_scalar;
    grad_q = grad_constant_scalar;
    break;
    
  case 1:
    std::cout << "[main] Interpolating linear function" << std::endl;
    q = linear_scalar;
    grad_q = grad_linear_scalar;
    break;
    
  case 2:
    std::cout << "[main] Interpolating quadratic function" << std::endl;
    q = quadratic_scalar;
    grad_q = grad_quadratic_scalar;
    break;

  case 3:
    std::cout << "[main] Interpolating trigonometric function" << std::endl;
    q = trigonometric_scalar;
    grad_q = grad_trigonometric_scalar;
    break;

  default:
    std::cerr << "ERROR: Unknown function" << std::endl;
    exit(1);
  }
  std::cout << "[main] Interpolating on XGrad" << std::endl;
  auto qh = x_grad.interpolate(q);

  //------------------------------------------------------------------------------
  // Check the consistency of the gradient and potential reconstructions
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------  
  // Edges
  
  Eigen::VectorXd GE_errors_edges = Eigen::VectorXd::Zero(mesh_ptr->n_edges());
  Eigen::VectorXd PE_errors_edges = Eigen::VectorXd::Zero(mesh_ptr->n_edges());
  
  auto compute_errors_edges
    = [&mesh_ptr, &x_grad, &qh, &GE_errors_edges, &PE_errors_edges, q, grad_q](size_t start, size_t end)
      {
	      for (size_t iE = start; iE < end; iE++) {
	        const Edge & E = *mesh_ptr->edge(iE);
	        auto tE = E.tangent();

	        QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (x_grad.degree() + 1));

	        std::function<double(const Eigen::Vector3d &)>
	          grad_q_dot_tE = [&tE, grad_q](const Eigen::Vector3d & x)->double
			          {
			            return grad_q(x).dot(tE);
			          };

	        Eigen::VectorXd IE_q = x_grad.restrictEdge(iE, qh);

	        Eigen::MatrixXd GE_IE_q = x_grad.edgeOperators(iE).gradient * IE_q;
	        auto GE_basis_quad = evaluate_quad<Function>::compute(*x_grad.edgeBases(iE).Polyk, quad_2kpo_E);

	        Eigen::MatrixXd PE_IE_q = x_grad.edgeOperators(iE).potential * IE_q;
	        auto PE_basis_quad = evaluate_quad<Function>::compute(*x_grad.edgeBases(iE).Polykpo, quad_2kpo_E);
	        
	        GE_errors_edges(iE) = squared_l2_error(grad_q_dot_tE, GE_IE_q, GE_basis_quad, quad_2kpo_E);
	        PE_errors_edges(iE) = squared_l2_error(q, PE_IE_q, PE_basis_quad, quad_2kpo_E);
	      } // for iE
      };

  std::cout << "[main] Checking the approximation properties of the edge gradient" << std::endl;
  parallel_for(mesh_ptr->n_edges(), compute_errors_edges, use_threads);

  double GE_error = std::sqrt( GE_errors_edges.sum() );
  double PE_error = std::sqrt( PE_errors_edges.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the gradient at edges " << std::scientific << GE_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at edges " << std::scientific << PE_error << std::endl;

  //------------------------------------------------------------------------------  
  // Faces
  
  Eigen::VectorXd GF_errors_faces = Eigen::VectorXd::Zero(mesh_ptr->n_faces());
  Eigen::VectorXd PF_errors_faces = Eigen::VectorXd::Zero(mesh_ptr->n_faces());
  
  auto compute_errors_faces
    = [&mesh_ptr, &x_grad, &qh, &GF_errors_faces, &PF_errors_faces, q, grad_q](size_t start, size_t end)
      {
	      for (size_t iF = start; iF < end; iF++) {
	        const Face & F = *mesh_ptr->face(iF);
	        auto nF = F.normal();

	        QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * (x_grad.degree() + 1));

	        std::function<Eigen::Vector3d(const Eigen::Vector3d &)>
	          nF_cross_grad_q_cross_nF = [&nF, grad_q](const Eigen::Vector3d & x)->Eigen::Vector3d
				             {
					       return nF.cross(grad_q(x)).cross(nF);
				             };

	        Eigen::VectorXd IF_q = x_grad.restrictFace(iF, qh);
	        
	        Eigen::MatrixXd GF_IF_q = x_grad.faceOperators(iF).gradient * IF_q;
	        auto GF_basis_quad = evaluate_quad<Function>::compute(*x_grad.faceBases(iF).Polyk2, quad_2kpo_F);

	        Eigen::MatrixXd PF_IF_q = x_grad.faceOperators(iF).potential * IF_q;
	        auto PF_basis_quad = evaluate_quad<Function>::compute(*x_grad.faceBases(iF).Polykpo, quad_2kpo_F);

	        GF_errors_faces(iF) = squared_l2_error(nF_cross_grad_q_cross_nF, GF_IF_q, GF_basis_quad, quad_2kpo_F);
	        PF_errors_faces(iF) = squared_l2_error(q, PF_IF_q, PF_basis_quad, quad_2kpo_F);
	      } // for iF
      };

  std::cout << "[main] Checking the approximation properties of the face gradient" << std::endl;
  parallel_for(mesh_ptr->n_faces(), compute_errors_faces, use_threads);

  double GF_error = std::sqrt( GF_errors_faces.sum() );
  double PF_error = std::sqrt( PF_errors_faces.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the gradient at faces " << std::scientific << GF_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at faces " << std::scientific << PF_error << std::endl;

  //------------------------------------------------------------------------------
  // Cells
  
  Eigen::VectorXd GT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());
  Eigen::VectorXd PT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());
  Eigen::VectorXd L2norm_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());  
  
  auto compute_errors_cells
    = [&mesh_ptr, &x_grad, &qh, &GT_errors_cells, &PT_errors_cells, &L2norm_errors_cells, q, grad_q](size_t start, size_t end)
      {
	      for (size_t iT = start; iT < end; iT++) {
	        const Cell & T = *mesh_ptr->cell(iT);

	        QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (x_grad.degree()+1));

	        Eigen::VectorXd IT_q = x_grad.restrictCell(iT, qh);
	        Eigen::MatrixXd GT_IT_q = x_grad.cellOperators(iT).gradient * IT_q;
	        auto GT_basis_quad = evaluate_quad<Function>::compute(*x_grad.cellBases(iT).Polyk3, quad_2kpo_T);
	        
	        Eigen::MatrixXd PT_IT_q = x_grad.cellOperators(iT).potential * IT_q;
	        auto PT_basis_quad = evaluate_quad<Function>::compute(*x_grad.cellBases(iT).Polykpo, quad_2kpo_T);

	        GT_errors_cells(iT) = squared_l2_error(grad_q, GT_IT_q, GT_basis_quad, quad_2kpo_T);
	        PT_errors_cells(iT) = squared_l2_error(q, PT_IT_q, PT_basis_quad, quad_2kpo_T);
	        
	        // Compute int_T q^2
          double int_q2 = 0.;
          for (size_t iqn=0; iqn < quad_2kpo_T.size(); iqn++){
            int_q2 += quad_2kpo_T[iqn].w * std::pow(q(quad_2kpo_T[iqn].vector()), 2);
          }
          Eigen::MatrixXd L2T = x_grad.computeL2Product(iT);
          L2norm_errors_cells(iT) = std::abs(IT_q.transpose() * L2T * IT_q - int_q2);

          // Check invertibility L2T
//          std::cout << L2T.fullPivLu().rank() - L2T.rows() << std::endl;

	      } // for iT
      };
 
  std::cout << "[main] Checking the approximation properties of the cell gradient" << std::endl;
  parallel_for(mesh_ptr->n_cells(), compute_errors_cells, use_threads);

  double GT_error = std::sqrt( GT_errors_cells.sum() );
  double PT_error = std::sqrt( PT_errors_cells.sum() );
  double L2norm_error = std::sqrt( L2norm_errors_cells.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the gradient at cells " << std::scientific << GT_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at cells " << std::scientific << PT_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the Xgrad L2 norm " << std::scientific << L2norm_error << std::endl;  

  std::cout << "[main] Done" << std::endl;
  exit(0);
}


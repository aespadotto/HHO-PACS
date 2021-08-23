// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/math/constants/constants.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <xcurl.hpp>
#include <parallel_for.hpp>

#include "xcurl-test.hpp"

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

  // Create XCurl
  XCurl x_curl(ddr_core, use_threads);

  // Interpolate a vector function
  int function_to_interpolate = (vm.count("function") ? vm["function"].as<int>() : 0);    
  std::function<Eigen::Vector3d(const Eigen::Vector3d &)> v;
  std::function<Eigen::Vector3d(const Eigen::Vector3d &)> curl_v;
  switch (function_to_interpolate) {
  case 0:
    std::cout << "[main] Interpolating constant function" << std::endl;    
    v = constant_vector;
    curl_v = curl_constant_vector;
    break;

  case 1:
    std::cout << "[main] Interpolating linear function" << std::endl;    
    v = linear_vector;
    curl_v = curl_linear_vector;
    break;

  case 4:
    std::cout << "[main] Interpolating constant function" << std::endl;    
    v = trigonometric_vector;
    curl_v = curl_trigonometric_vector;
    break;

  default:
    std::cerr << "ERROR: Unknown function" << std::endl;
    exit(1);    
  }
  std::cout << "[main] Interpolating on XCurl" << std::endl;
  auto vh = x_curl.interpolate(v);

  //------------------------------------------------------------------------------
  // Check the consistency of the curl and potential reconstructions
  //------------------------------------------------------------------------------

  Eigen::VectorXd CF_errors_faces = Eigen::VectorXd::Zero(mesh_ptr->n_faces());
  Eigen::VectorXd PF_errors_faces = Eigen::VectorXd::Zero(mesh_ptr->n_faces());  

  auto compute_errors_faces
    = [&mesh_ptr, &x_curl, &vh, &CF_errors_faces, &PF_errors_faces, v, curl_v](size_t start, size_t end)
      {
	      for (size_t iF = start; iF < end; iF++) {
	        const Face & F = *mesh_ptr->face(iF);
	        auto nF = F.normal();

	        QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * x_curl.degree());

	        std::function<double(const Eigen::Vector3d &)>
	          rotF_v = [&nF, curl_v](const Eigen::Vector3d & x)->double
		           {
		             return curl_v(x).dot(nF);
		           };

	        std::function<Eigen::Vector3d(const Eigen::Vector3d &)>
	          nF_cross_v_cross_nF = [&nF, v](const Eigen::Vector3d & x)->Eigen::Vector3d
				        {
				          return (nF.cross(v(x))).cross(nF);
				        };
	        
	        Eigen::VectorXd IF_v = x_curl.restrictFace(iF, vh);

	        Eigen::VectorXd CF_IF_v = x_curl.faceOperators(iF).curl * IF_v;
	        auto CF_basis_quad = evaluate_quad<Function>::compute(*x_curl.faceBases(iF).Polyk, quad_2k_F);

	        Eigen::VectorXd PF_IF_v = x_curl.faceOperators(iF).potential * IF_v;
	        auto PF_basis_quad = evaluate_quad<Function>::compute(*x_curl.faceBases(iF).Polyk2, quad_2k_F);

	        CF_errors_faces(iF) = squared_l2_error(rotF_v, CF_IF_v, CF_basis_quad, quad_2k_F);
	        PF_errors_faces(iF) = squared_l2_error(nF_cross_v_cross_nF, PF_IF_v, PF_basis_quad, quad_2k_F);
	      } // for iF
      };

  std::cout << "[main] Checking the approximation properties of the face curl and potential" << std::endl;
  parallel_for(mesh_ptr->n_faces(), compute_errors_faces, use_threads);

  double CF_error = std::sqrt( CF_errors_faces.sum() );
  double PF_error = std::sqrt( PF_errors_faces.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the curl at faces " << std::scientific << CF_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at faces " << std::scientific << PF_error << std::endl;

  //------------------------------------------------------------------------------
  
  Eigen::VectorXd CT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());
  Eigen::VectorXd PT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());  
  Eigen::VectorXd L2norm_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());  

  auto compute_errors_cells
    = [&mesh_ptr, &x_curl, &vh, &CT_errors_cells, &PT_errors_cells, &L2norm_errors_cells, v, curl_v](size_t start, size_t end)
      {
	      for (size_t iT = start; iT < end; iT++) {
	        const Cell & T = *mesh_ptr->cell(iT);

	        QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * x_curl.degree());

	        Eigen::VectorXd IT_v = x_curl.restrictCell(iT, vh);

	        Eigen::VectorXd CT_IT_v = x_curl.cellOperators(iT).curl * IT_v;
	        Eigen::VectorXd PT_IT_v = x_curl.cellOperators(iT).potential * IT_v;
	        auto CT_PT_basis_quad = evaluate_quad<Function>::compute(*x_curl.cellBases(iT).Polyk3, quad_2k_T);
	      
	        CT_errors_cells(iT) = squared_l2_error(curl_v, CT_IT_v, CT_PT_basis_quad, quad_2k_T);
	        PT_errors_cells(iT) = squared_l2_error(v, PT_IT_v, CT_PT_basis_quad, quad_2k_T);

          // Compute int_T v^2
          double int_v2 = 0.;
          for (size_t iqn=0; iqn < quad_2k_T.size(); iqn++){
            int_v2 += quad_2k_T[iqn].w * v(quad_2k_T[iqn].vector()).squaredNorm();
          }
          Eigen::MatrixXd L2T = x_curl.computeL2Product(iT);
          L2norm_errors_cells(iT) = std::abs(IT_v.transpose() * L2T * IT_v - int_v2);
	      } // for iT
      };
  
  std::cout << "[main] Checking the approximation properties of the cell curl and potential" << std::endl;
  parallel_for(mesh_ptr->n_cells(), compute_errors_cells, use_threads);

  double CT_error = std::sqrt( CT_errors_cells.sum() );
  double PT_error = std::sqrt( PT_errors_cells.sum() );
  double L2norm_error = std::sqrt( L2norm_errors_cells.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the curl at cells " << std::scientific << CT_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at cells " << std::scientific << PT_error << std::endl;  
  std::cout << FORMAT(25) << "[main] L2-error for the Xcurl L2 norm " << std::scientific << L2norm_error << std::endl;  
  
  std::cout << "[main] Done" << std::endl;
  exit(0);
}


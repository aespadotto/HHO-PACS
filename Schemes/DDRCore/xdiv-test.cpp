// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/math/constants/constants.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <xdiv.hpp>
#include <parallel_for.hpp>

#include "xdiv-test.hpp"

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

  // Create XDiv
  XDiv x_div(ddr_core, use_threads);

  // Interpolate a vector function
  int function_to_interpolate = (vm.count("function") ? vm["function"].as<int>() : 0);    
  std::function<Eigen::Vector3d(const Eigen::Vector3d &)> v;
  std::function<double(const Eigen::Vector3d &)> div_v;
  switch (function_to_interpolate) {
  case 0:
    std::cout << "[main] Interpolating constant function" << std::endl;    
    v = constant_vector;
    div_v = div_constant_vector;
    break;

  case 4:
    std::cout << "[main] Interpolating trigonometric function" << std::endl;    
    v = trigonometric_vector;
    div_v = div_trigonometric_vector;
    break;

  default:
    std::cerr << "ERROR: Unknown function" << std::endl;
    exit(1);    
  }
  std::cout << "[main] Interpolating on XDiv" << std::endl;
  auto vh = x_div.interpolate(v);

  //------------------------------------------------------------------------------
  // Check the consistency of the divergence and potential reconstructions
  //------------------------------------------------------------------------------
  
  Eigen::VectorXd DT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());
  Eigen::VectorXd PT_errors_cells = Eigen::VectorXd::Zero(mesh_ptr->n_cells());

  auto compute_errors_cells
    = [&mesh_ptr, &x_div, &vh, &DT_errors_cells, &PT_errors_cells, v, div_v](size_t start, size_t end)
      {
	for (size_t iT = start; iT < end; iT++) {
	  const Cell & T = *mesh_ptr->cell(iT);

	  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * x_div.degree());

	  Eigen::VectorXd IT_v = x_div.restrictCell(iT, vh);

	  Eigen::MatrixXd DT_IT_v = x_div.cellOperators(iT).divergence * IT_v;
	  auto DT_basis_quad = evaluate_quad<Function>::compute(*x_div.cellBases(iT).Polyk, quad_2k_T);
	  DT_errors_cells(iT) = squared_l2_error(div_v, DT_IT_v, DT_basis_quad, quad_2k_T);
	  
	  Eigen::MatrixXd PT_IT_v = x_div.cellOperators(iT).potential * IT_v;
	  auto PT_basis_quad = evaluate_quad<Function>::compute(*x_div.cellBases(iT).Polyk3, quad_2k_T);
	  PT_errors_cells(iT) = squared_l2_error(v, PT_IT_v, PT_basis_quad, quad_2k_T);
	} // for iT
      };

  std::cout << "[main] Checking the approximation properties of the cell gradient" << std::endl;
  parallel_for(mesh_ptr->n_cells(), compute_errors_cells, use_threads);

  double DT_error = std::sqrt( DT_errors_cells.sum() );
  double PT_error = std::sqrt( PT_errors_cells.sum() );
  std::cout << FORMAT(25) << "[main] L2-error for the divergence at cells " << std::scientific << DT_error << std::endl;
  std::cout << FORMAT(25) << "[main] L2-error for the potential at cells " << std::scientific << PT_error << std::endl;

  std::cout << "[main] Done" << std::endl;
  exit(0);
}


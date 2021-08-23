// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/math/constants/constants.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>
#include <vertex.hpp>

#include <basis.hpp>

#include "l2projection.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;
using namespace HArDCore3D::Tests;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "../../meshes/";
std::string default_mesh = mesh_dir + "Voro-small-0/RF_fmt/voro-2";
std::string default_meshtype = "RF";

//------------------------------------------------------------------------------
// Typedefs
//------------------------------------------------------------------------------

typedef Family<MonomialScalarBasisCell> PkTBasisType;
typedef TensorizedVectorFamily<PkTBasisType, 3> Pk3TBasisType;
typedef Family<GradientBasis<ShiftedBasis<MonomialScalarBasisCell> > > GkTBasisType;
typedef Family<TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> > GOkTBasisType;
typedef CurlBasis<Family<TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> > > RkTBasisType;
typedef Family<TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> > ROkTBasisType;

typedef RolyComplBasisCell RckTBasisType;
typedef GolyComplBasisCell GckTBasisType;
typedef RolyComplBasisFace RckFBasisType;
typedef GolyComplBasisFace GckFBasisType;

typedef Family<MonomialScalarBasisFace> PkFBasisType;
typedef TangentFamily<PkFBasisType> Pk2FBasisType;
typedef Family<CurlBasis<ShiftedBasis<Family<MonomialScalarBasisFace> > > > RkFBasisType;
typedef Family<GradientBasis<ShiftedBasis<Family<MonomialScalarBasisFace> > > > GkFBasisType;
typedef Family<Pk2FBasisType> ROkFBasisType;

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

static const double PI = boost::math::constants::pi<double>();
using std::sin;

static auto q = [](const Eigen::Vector3d & x) -> double {
                  return sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2));
                };

static auto grad_q = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                       return PI * Eigen::Vector3d(
					      cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
					      sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2)),
					      sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2))
					      );
                     };

static auto v = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
		  return Eigen::Vector3d(
					 sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
					 sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
					 sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2))
					 );
		};

static auto curl_v
= [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
    return PI * Eigen::Vector3d(
				sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2)) - sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2)),
				sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2)) - cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
				cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)) - sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2))
				);
  };

//------------------------------------------------------------------------------

int main(int argc, const char* argv[]) {

  // Program options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<std::string>(), "Set the mesh")
    ("meshtype,t", boost::program_options::value<std::string>(), "Set the mesh type (TG,MSH,RF)")
    ("degree,k", boost::program_options::value<size_t>()->default_value(1), "The polynomial degree on the faces")
    ("eps,e", boost::program_options::value<double>()->default_value(1.e-12), "The tolerance in numerical comparisons");

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

  // Select the degree
  size_t K = vm["degree"].as<size_t>();

  // Select the tolerance for numerical comparisons
  double tol = vm["eps"].as<double>();

  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  //------------------------------------------------------------------------------
  // Generate and store the bases over the elements
  //------------------------------------------------------------------------------
 
  // Initialize vectors to store the basis for the DDR spaces on elements
  std::vector<PkTBasisType> basis_Pk_Th; basis_Pk_Th.reserve(mesh_ptr->n_cells());
  std::vector<Pk3TBasisType> basis_Pk3_Th; basis_Pk3_Th.reserve(mesh_ptr->n_cells());
  std::vector<GkTBasisType> basis_Gk_Th; basis_Gk_Th.reserve(mesh_ptr->n_cells());
  std::vector<GOkTBasisType> basis_GOk_Th; basis_GOk_Th.reserve(mesh_ptr->n_cells());
  std::vector<RkTBasisType> basis_Rk_Th; basis_Rk_Th.reserve(mesh_ptr->n_cells());
  std::vector<ROkTBasisType> basis_ROk_Th; basis_ROk_Th.reserve(mesh_ptr->n_cells());
  
  std::vector<RckTBasisType> basis_Rck_Th; basis_Rck_Th.reserve(mesh_ptr->n_cells());
  std::vector<GckTBasisType> basis_Gck_Th; basis_Gck_Th.reserve(mesh_ptr->n_cells());

  // Initialize the projection errors
  double err_l2_Pk_Th = 0.;
  double err_l2_Gk_Th = 0.;
  double err_l2_GOk_Th = 0.;
  double err_l2_Rk_Th = 0.;
  double err_l2_ROk_Th = 0.;
  
 
  for (size_t iT = 0; iT < mesh_ptr->n_cells(); iT++) {
//  for (size_t iT = 0; iT < 0; iT++) {
    std::cout << "** Element " << iT << std::endl;
    
    const Cell & T = *mesh_ptr->cell(iT);

    //------------------------------------------------------------------------------
    // Bases for Pk(T) and Pk(T)^3
    //------------------------------------------------------------------------------
    
    // Generate a monomial scalar basis for Pk(T)
    MonomialScalarBasisCell basis_Pk_T(T, K);
    std::cout << FORMAT(45) << "Basis dimension for Pk(T) " << basis_Pk_T.dimension() << std::endl;

    // Generate a quadrature rule with degree of exactness 2k
    QuadratureRule qr_2k_T = generate_quadrature_rule(T, 2 * K);

    // Evaluate the scalar monomial basis for Pk(T) at quadrature nodes
    auto basis_Pk_T_quad = evaluate_quad<Function>::compute(basis_Pk_T, qr_2k_T);
    

    // Generate an L2-orthonormal basis for Pk(T).
    // After calling orthonormalization, basis_Pk_T will contain the evaluation of the
    // orthonormal basis at quadrature nodes
    PkTBasisType on_basis_Pk_T = l2_orthonormalize(basis_Pk_T, qr_2k_T, basis_Pk_T_quad);    
    
    // Evaluate orthonormal basis for Pk(T) at quadrature nodes
    auto on_basis_Pk_T_quad = evaluate_quad<Function>::compute(on_basis_Pk_T, qr_2k_T);

    // Check that basis_Pk_T_quad after orthonormalization and on_basis_Pk_T_quad
    // coincide
    {
      double diff = 0.;
      for (size_t i = 0; i < basis_Pk_T.dimension(); i++) {
	for (size_t iqn = 0; iqn < qr_2k_T.size(); iqn++) {
	  diff += std::pow(basis_Pk_T_quad[i][iqn] - on_basis_Pk_T_quad[i][iqn], 2);
	} // for iqn
      } // for i
      diff = std::sqrt(diff);
      if (diff > tol) {
	std::cerr << "ERROR: Inconsistent basis evaluation after orthonormalization Expected err < "
		  << tol << " got " << diff << std::endl;
	exit(1);
      }
    }

    // Compute the L2-projection of a scalar function and the corresponding error
    // When the elements are not split, one should use a more precise quadrature
    // to compute the error. This is usually not necessary when elements are split.
    // For performance reasons, we do not increase the degree of exactness of the
    // quadrature here
    L2Projection<PkTBasisType> l2_Pk_T(on_basis_Pk_T, qr_2k_T, on_basis_Pk_T_quad);
    auto l2_Pk_q = l2_Pk_T.compute(q);
    err_l2_Pk_Th += l2_Pk_T.squared_error(q, l2_Pk_q);

    // Generate a vector basis for Pk(T)^3 by tensorization    
    Pk3TBasisType on_basis_Pk3_T(on_basis_Pk_T);
    std::cout << FORMAT(45) << "Basis dimension for Pk(T)^3" << on_basis_Pk3_T.dimension() << std::endl;
    
    // Evaluate the vector basis for Pk(T)^3 at quadrature points
    // This should be avoided if possible, as it is an expensive operation
    auto on_basis_Pk3_T_quad = evaluate_quad<Function>::compute(on_basis_Pk3_T, qr_2k_T);

    //------------------------------------------------------------------------------
    // Bases for Gk(T) and GOk(T)
    //------------------------------------------------------------------------------

    // Generate a basis for Gk(T)
    GradientBasis<ShiftedBasis<MonomialScalarBasisCell> >
      basis_Gk_T(ShiftedBasis<MonomialScalarBasisCell>(MonomialScalarBasisCell(T, K+1), 1));
    std::cout << FORMAT(45) << "Basis dimension for Gk(T)" << basis_Gk_T.dimension() << std::endl;
 
    // Evaluate the bases for Gk(T) at quadrature nodes
    auto basis_Gk_T_quad = evaluate_quad<Function>::compute(basis_Gk_T, qr_2k_T);

    // Generate an L2-orthonormal basis for Gk(T)
    GkTBasisType on_basis_Gk_T = l2_orthonormalize(basis_Gk_T, qr_2k_T, basis_Gk_T_quad);

    // Compute the L2-projection of a gradient function and the corresponding error
    L2Projection<GkTBasisType> l2_Gk_T(on_basis_Gk_T, qr_2k_T, basis_Gk_T_quad);
    auto l2_Gk_grad_q = l2_Gk_T.compute(grad_q);
    err_l2_Gk_Th += l2_Gk_T.squared_error(grad_q, l2_Gk_grad_q);
    
    // Compute the Gram matrix of Gk(T) and Pk(T)^3
    Eigen::MatrixXd MG_tensorized = compute_gram_matrix(basis_Gk_T_quad, on_basis_Pk_T_quad, qr_2k_T);
    Eigen::MatrixXd MG = compute_gram_matrix(basis_Gk_T_quad, on_basis_Pk3_T_quad, qr_2k_T);    
    if ( (MG-MG_tensorized).norm()  > tol ) {
      std::cerr << "ERROR: Inconsistent standard and tensorized Gram matrices. Expected err < "<< tol << " got "
                << (MG-MG_tensorized).norm()
                << std::endl;
      exit(1);
    }
        
    // Compute a basis for GOk(T)
    if (K>=1){
      GOkTBasisType basis_GOk_T(on_basis_Pk3_T, MG_tensorized.fullPivLu().kernel().transpose());
      std::cout << FORMAT(45) << "Basis dimension for GOk(T)" << basis_GOk_T.dimension() << std::endl;

      // Check that we got the dimension right
      if (basis_GOk_T.dimension() != on_basis_Pk3_T.dimension() - on_basis_Gk_T.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for GOk(T). Expected "
                  << ( on_basis_Pk3_T.dimension() - on_basis_Gk_T.dimension() )
                  << " got " << basis_GOk_T.dimension() << std::endl;
        exit(1);
      }

      // Compute the L2-projection of a gradient function. By a triangle inequality,
      // we expect that the L2-norm of the projection goes to zero as h^{k+1}
      auto basis_GOk_T_quad = evaluate_quad<Function>::compute(basis_GOk_T, qr_2k_T);
      L2Projection<GOkTBasisType> l2_GOk_T(basis_GOk_T, qr_2k_T, basis_GOk_T_quad);
      auto l2_GOk_grad_q = l2_GOk_T.compute(grad_q);
      err_l2_GOk_Th += squared_l2_norm(l2_GOk_grad_q, qr_2k_T, basis_GOk_T_quad);
      
      // Store
      basis_GOk_Th.push_back(basis_GOk_T);

    }
    
    //------------------------------------------------------------------------------
    // Bases for Rk(T) and ROk(T)
    //------------------------------------------------------------------------------

    // Orthonormal basis for Pk+1(T)
    MonomialScalarBasisCell basis_Pkpo_T(T, K + 1);
    QuadratureRule qr_2kpo_T = generate_quadrature_rule(T, 2 * (K + 1));
    auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(basis_Pkpo_T, qr_2kpo_T);
    Family<MonomialScalarBasisCell> on_basis_Pkpo_T = l2_orthonormalize(basis_Pkpo_T, qr_2kpo_T, basis_Pkpo_T_quad);

    // Tensorized basis for Pk+1(T)^3
    TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> on_basis_Pkpod_T(on_basis_Pkpo_T);

    // Basis for Gk+1(T)
    GradientBasis<ShiftedBasis<MonomialScalarBasisCell> >
      basis_Gkpo_T(ShiftedBasis<MonomialScalarBasisCell>(MonomialScalarBasisCell(T, K + 2), 1));
    auto basis_Gkpo_T_quad = evaluate_quad<Function>::compute(basis_Gkpo_T, qr_2kpo_T);
    GkTBasisType on_basis_Gkpo_T = l2_orthonormalize(basis_Gkpo_T, qr_2kpo_T, basis_Gkpo_T_quad);

    // Basis for GOk+1(T)
    auto MGpo = compute_gram_matrix(basis_Gkpo_T_quad, basis_Pkpo_T_quad, qr_2kpo_T);
    Family<TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> > basis_GOkpo_T(on_basis_Pkpod_T, MGpo.fullPivLu().kernel().transpose());

    // Basis for Rk(T)
    RkTBasisType basis_Rk_T(basis_GOkpo_T);
    std::cout << FORMAT(45) << "Basis dimension for Rk(T)" << basis_Rk_T.dimension() << std::endl;

    // Compute the L2-orthogonal projection
    L2Projection<RkTBasisType> l2_Rk_T(basis_Rk_T, qr_2k_T, evaluate_quad<Function>::compute(basis_Rk_T, qr_2k_T));
    auto l2_curl_v = l2_Rk_T.compute(curl_v);
    err_l2_Rk_Th += l2_Rk_T.squared_error(curl_v, l2_curl_v);
    
    // Basis for ROk(T)
    if (K>=1){
      auto basis_Rk_T_quad = evaluate_quad<Function>::compute(basis_Rk_T, qr_2k_T);
      Eigen::MatrixXd MR = compute_gram_matrix(basis_Rk_T_quad, on_basis_Pk_T_quad, qr_2k_T);
      ROkTBasisType basis_ROk_T(on_basis_Pk3_T, MR.fullPivLu().kernel().transpose());
      std::cout << FORMAT(45) << "Basis dimension for ROk(T)" << basis_ROk_T.dimension() << std::endl;

      // Check that we got the dimension right
      if (basis_ROk_T.dimension() != on_basis_Pk3_T.dimension() - basis_GOkpo_T.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for ROk(T). Expected "
                  << (basis_ROk_T.dimension() != on_basis_Pk3_T.dimension() - basis_GOkpo_T.dimension())
                  << " got " << basis_ROk_T.dimension();
        exit(1);
      }

      // Compute the L2-projection of a curl function. By a triangle inequality,
      // we expect that the L2-norm of the projection goes to zero as h^{k+1}
      auto basis_ROk_T_quad = evaluate_quad<Function>::compute(basis_ROk_T, qr_2k_T);
      L2Projection<ROkTBasisType> l2_ROk_T(basis_ROk_T, qr_2k_T, basis_ROk_T_quad);
      auto l2_ROk_curl_v = l2_ROk_T.compute(curl_v);
      err_l2_ROk_Th += squared_l2_norm(l2_ROk_curl_v, qr_2k_T, basis_ROk_T_quad);
      
      // Store
      basis_ROk_Th.push_back(basis_ROk_T);

    }
        
    //------------------------------------------------------------------------------
    // Bases for Rck(T)
    //------------------------------------------------------------------------------
    if (K>=1){
      RckTBasisType basis_Rck_T(T, K);
      std::cout << FORMAT(45) << "Basis dimension for Rck(T)" << basis_Rck_T.dimension() << std::endl;
      if (basis_Rck_T.dimension() != on_basis_Pk3_T.dimension() - basis_GOkpo_T.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for Rck(T). Expected "
                  << (basis_Rck_T.dimension() != on_basis_Pk3_T.dimension() - basis_GOkpo_T.dimension())
                  << " got " << basis_Rck_T.dimension();
        exit(1);
      }
      // Test colinearity at a few points
      double error_colinearity = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        VectorRd x = VectorRd::Random();
        for (size_t i=0; i < basis_Rck_T.dimension(); i++){
          error_colinearity += std::abs( (basis_Rck_T.function(i, x).cross(x-T.center_mass())).norm() / basis_Rck_T.function(i, x).norm() );
        }
      }
      if (error_colinearity > tol){
        std::cout << "Basis of RckT not colinear with x, error=" << error_colinearity << std::endl;
        exit(1);
      }
      
      // Compute values on quadrature points
      auto basis_Rck_T_quad = evaluate_quad<Function>::compute(basis_Rck_T, qr_2k_T);
      Eigen::MatrixXd MassRck_T = compute_gram_matrix(basis_Rck_T_quad, basis_Rck_T_quad, qr_2k_T);
  //    std::cout << "Mass matrix:\n " << MassRck_T << std::endl;
      
      // Orthonormalise and re-test
      Family<RckTBasisType> on_basis_Rck_T = l2_orthonormalize(basis_Rck_T, qr_2k_T, basis_Rck_T_quad);
      error_colinearity = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        VectorRd x = T.center_mass() + T.diam()*VectorRd::Random();
        for (size_t i=0; i < basis_Rck_T.dimension(); i++){
          error_colinearity += std::abs( (on_basis_Rck_T.function(i, x).cross(x-T.center_mass())).norm() / on_basis_Rck_T.function(i, x).norm() );
        }
      }
      if (error_colinearity > tol){
        std::cout << "ON basis of RckT not colinear with x-xF, error=" << error_colinearity << std::endl;
        exit(1);
      }

      Eigen::MatrixXd on_MassRck_T = compute_gram_matrix(basis_Rck_T_quad, basis_Rck_T_quad, qr_2k_T);
  //    std::cout << "ON Mass matrix:\n " << on_MassRck_T << std::endl;
      
      // Store
      basis_Rck_Th.push_back(basis_Rck_T);
    }
    
    //------------------------------------------------------------------------------
    // Bases for Gck(T)
    //------------------------------------------------------------------------------
    if (K>=1){
      GckTBasisType basis_Gck_T(T, K);
      std::cout << FORMAT(45) << "Basis dimension for Gck(T)" << basis_Gck_T.dimension() << std::endl;
      if (basis_Gck_T.dimension() != on_basis_Pk3_T.dimension() - on_basis_Gk_T.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for Gck(T). Expected "
                  << ( on_basis_Pk3_T.dimension() - on_basis_Gk_T.dimension() )
                  << " got " << basis_Gck_T.dimension() << std::endl;
        exit(1);
      }

      // Test orthogonality at a few points
      double error_orthogonality = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        VectorRd x = T.center_mass() + T.diam()*VectorRd::Random();
        for (size_t i=0; i < basis_Gck_T.dimension(); i++){
          error_orthogonality += std::abs( basis_Gck_T.function(i, x).dot(x-T.center_mass()) / (basis_Gck_T.function(i, x).norm() * (x-T.center_mass()).norm()) );
        }
      }
      if (error_orthogonality>tol){
        std::cout << "Basis of GckT bis not orthogonal with x-xT, error=" << error_orthogonality << std::endl;
        exit(1);
      }

      // Compute values on quadrature points
      auto basis_Gck_T_quad = evaluate_quad<Function>::compute(basis_Gck_T, qr_2k_T);
      Eigen::MatrixXd MassGck_T = compute_gram_matrix(basis_Gck_T_quad, basis_Gck_T_quad, qr_2k_T);
  //    std::cout << "Mass matrix:\n " << MassGck_T << std::endl;
      
      // Orthonormalise and re-test
      Family<GckTBasisType> on_basis_Gck_T = l2_orthonormalize(basis_Gck_T, qr_2k_T, basis_Gck_T_quad);
      error_orthogonality = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        VectorRd x = T.center_mass() + T.diam()*VectorRd::Random();
        for (size_t i=0; i < basis_Gck_T.dimension(); i++){
          error_orthogonality += std::abs( on_basis_Gck_T.function(i, x).dot(x-T.center_mass()) / (on_basis_Gck_T.function(i, x).norm() * (x-T.center_mass()).norm()) );
        }
      }
      if (error_orthogonality>tol){
        std::cout << "ON basis of GckT not orthogonal with x-xT, error=" << error_orthogonality << std::endl;
        exit(1);
      }
      Eigen::MatrixXd on_MassGck_T = compute_gram_matrix(basis_Gck_T_quad, basis_Gck_T_quad, qr_2k_T);
  //    std::cout << "ON Mass matrix:\n " << on_MassGck_T << std::endl;

      // Store
      basis_Gck_Th.push_back(basis_Gck_T);
      
      
      // Basis for Rck-1
      CurlBasis<GolyComplBasisCell> basis_Rckmo_T(basis_Gck_T);
      auto basis_Rckmo_T_quad = evaluate_quad<Function>::compute(basis_Rckmo_T, qr_2k_T);
      Eigen::MatrixXd MassRckmo_T = compute_gram_matrix(basis_Rckmo_T_quad, basis_Rckmo_T_quad, qr_2k_T);
      size_t rk = MassRckmo_T.fullPivLu().rank();
      std::cout << "dimension of Rckmo " << rk << ", expected " << PolynomialSpaceDimension<Cell>::Roly(K-1) << std::endl;
//      std::cout << std::endl << MassRckmo_T << std::endl;
    }
        
    // Store computed bases
    basis_Pk_Th.push_back(on_basis_Pk_T);
    basis_Pk3_Th.push_back(on_basis_Pk3_T);
    basis_Gk_Th.push_back(on_basis_Gk_T);
    basis_Rk_Th.push_back(basis_Rk_T);
  } // for iT

  // Print projection errors
  std::cout << FORMAT(15) << "err_l2_Pk_Th = " << std::scientific << std::sqrt(err_l2_Pk_Th) << std::endl;
  std::cout << FORMAT(15) << "err_l2_Gk_Th = " << std::scientific << std::sqrt(err_l2_Gk_Th) << std::endl;
  std::cout << FORMAT(15) << "err_l2_GOk_Th = " << std::scientific << std::sqrt(err_l2_GOk_Th) << std::endl;
  std::cout << FORMAT(15) << "err_l2_Rk_Th = " << std::scientific << std::sqrt(err_l2_Rk_Th) << std::endl;
  std::cout << FORMAT(15) << "err_l2_ROk_Th = " << std::scientific << std::sqrt(err_l2_ROk_Th) << std::endl;


  //------------------------------------------------------------------------------
  // Generate and store the bases over the faces
  //------------------------------------------------------------------------------

  // Initialize the projection errors
  double err_l2_Pk_Fh = 0.;
  double err_l2_Rk_Fh = 0.;
  double err_l2_ROk_Fh = 0.;

  // Initialize vectors to store the basis for the DDR spaces on faces
  std::vector<PkFBasisType> basis_Pk_Fh; basis_Pk_Fh.reserve(mesh_ptr->n_faces());
  std::vector<RkFBasisType> basis_Rk_Fh; basis_Rk_Fh.reserve(mesh_ptr->n_faces());
  std::vector<ROkFBasisType> basis_ROk_Fh; basis_ROk_Fh.reserve(mesh_ptr->n_faces());

  std::vector<RckFBasisType> basis_Rck_Fh; basis_Rck_Fh.reserve(mesh_ptr->n_faces());
  std::vector<GckFBasisType> basis_Gck_Fh; basis_Gck_Fh.reserve(mesh_ptr->n_faces());
  
//  for (size_t iF = 0; iF < mesh_ptr->n_faces(); iF++) {
  for (size_t iF = 0; iF < 0; iF++) {
    std::cout << "** Face " << iF << std::endl;

    const Face & F = *mesh_ptr->face(iF);

    //------------------------------------------------------------------------------
    // Bases for Pk(F) and Pk(F)^2
    //------------------------------------------------------------------------------
    
    MonomialScalarBasisFace basis_Pk_F(F, K);
    std::cout << FORMAT(45) << "Basis dimension for Pk(F) " << basis_Pk_F.dimension() << std::endl;
    
    // Generate a quadrature rule with degree of exactness 2k
    QuadratureRule qr_2k_F = generate_quadrature_rule(F, 2 * K);

    // Evaluate the scalar monomial basis for Pk(F) at quadrature nodes
    auto basis_Pk_F_quad = evaluate_quad<Function>::compute(basis_Pk_F, qr_2k_F);

    // Generate an L2-orthonormal basis for Pk(F)
    PkFBasisType on_basis_Pk_F = l2_orthonormalize(basis_Pk_F, qr_2k_F, basis_Pk_F_quad);

    // Generate a basis for Pk(F)^2
    Pk2FBasisType basis_Pk2_F(on_basis_Pk_F, basis_Pk_F.jacobian());
    auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(basis_Pk2_F, qr_2k_F);

    // Compute the L2-projection of a scalar function and the corresponding error
    L2Projection<PkFBasisType> l2_Pk_F(on_basis_Pk_F, qr_2k_F, basis_Pk_F_quad);
    auto l2_Pk_q = l2_Pk_F.compute(q);
    err_l2_Pk_Fh += l2_Pk_F.squared_error(q, l2_Pk_q);
    
    //------------------------------------------------------------------------------
    // Basis for Rk(F) and ROk(F)
    //------------------------------------------------------------------------------
    
    // Generate a basis for Rk(F)
    MonomialScalarBasisFace basis_Pkpo_F(F, K + 1);
    QuadratureRule qr_2kpo_F = generate_quadrature_rule(F, 2 * (K + 1));
    auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(basis_Pkpo_F, qr_2kpo_F);
    PkFBasisType on_basis_Pkpo_F = l2_orthonormalize(basis_Pkpo_F, qr_2kpo_F, basis_Pkpo_F_quad);
    
    CurlBasis<ShiftedBasis<Family<MonomialScalarBasisFace> > > basis_Rk_F(ShiftedBasis<PkFBasisType>(on_basis_Pkpo_F, 1));
    auto basis_Rk_F_quad = evaluate_quad<Function>::compute(basis_Rk_F, qr_2k_F);
    RkFBasisType on_basis_Rk_F = l2_orthonormalize(basis_Rk_F, qr_2k_F, basis_Rk_F_quad);
    std::cout << FORMAT(45) << "Basis dimension for Rk(F) " << on_basis_Rk_F.dimension() << std::endl;

    // Compute the L2-projection of a normal curl function and the correspoding error
    Eigen::Vector3d nF = basis_Pk_F.normal();
    auto grad_q_cross_nF  = [nF](const Eigen::Vector3d & x)->Eigen::Vector3d { return grad_q(x).cross(nF); };
    L2Projection<RkFBasisType> l2_Rk_F(on_basis_Rk_F, qr_2k_F, basis_Rk_F_quad);
    auto l2_Rk_grad_q_cross_nF = l2_Rk_F.compute(grad_q_cross_nF);
    err_l2_Rk_Fh += l2_Rk_F.squared_error(grad_q_cross_nF, l2_Rk_grad_q_cross_nF);

    // Generate a basis for ROk(F)
    if (K>=1){
      Eigen::MatrixXd MR = compute_gram_matrix(basis_Rk_F_quad, basis_Pk2_F_quad, qr_2k_F);
      ROkFBasisType basis_ROk_F(basis_Pk2_F, MR.fullPivLu().kernel().transpose());
      std::cout << FORMAT(45) << "Basis dimension for ROk(F) " << basis_ROk_F.dimension() << std::endl;

      // Check that we got the dimension right
      if (basis_ROk_F.dimension() != basis_Pk2_F.dimension() - on_basis_Rk_F.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for ROk(F). Expected "
                  << ( basis_Pk2_F.dimension() - on_basis_Rk_F.dimension() )
                  << " got " << basis_ROk_F.dimension() << std::endl;
        exit(1);
      }

      // Compute the L2-projection of a normal curl function and the correspoding L2-norm
      auto basis_ROk_F_quad = evaluate_quad<Function>::compute(basis_ROk_F, qr_2k_F);
      L2Projection<ROkFBasisType> l2_ROk_F(basis_ROk_F, qr_2k_F, basis_ROk_F_quad);
      auto l2_ROk_grad_q_cross_nF = l2_ROk_F.compute(grad_q_cross_nF);
      err_l2_ROk_Fh += squared_l2_norm(l2_ROk_grad_q_cross_nF, qr_2k_F, basis_ROk_F_quad);
      
      // Store
      basis_ROk_Fh.push_back(basis_ROk_F);
    }    
    
    //------------------------------------------------------------------------------
    // Basis for Gk(F)
    //------------------------------------------------------------------------------
    
    // Generate a basis for Gk(F)
    GradientBasis<ShiftedBasis<Family<MonomialScalarBasisFace> > > basis_Gk_F(ShiftedBasis<PkFBasisType>(on_basis_Pkpo_F, 1));
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(basis_Gk_F, qr_2k_F);
    GkFBasisType on_basis_Gk_F = l2_orthonormalize(basis_Gk_F, qr_2k_F, basis_Gk_F_quad);
    std::cout << FORMAT(45) << "Basis dimension for Gk(F) " << on_basis_Gk_F.dimension() << std::endl;


    //------------------------------------------------------------------------------
    // Basis for Rck(F)
    //------------------------------------------------------------------------------
    if (K>=1){
      RckFBasisType basis_Rck_F(F, K);
      std::cout << FORMAT(45) << "Basis dimension for Rck(F)" << basis_Rck_F.dimension() << std::endl;
      if (basis_Rck_F.dimension() != basis_Pk2_F.dimension() - on_basis_Rk_F.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for Rck(F). Expected "
                  << (basis_Pk2_F.dimension() - on_basis_Rk_F.dimension())
                  << " got " << basis_Rck_F.dimension();
        exit(1);
      }
      // Test colinearity at a few points
      double error_colinearity = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        Eigen::Vector2d coef = Eigen::Vector2d::Random();
        VectorRd x = F.center_mass() + F.edge_normal(0)*coef(0) + F.edge(0)->tangent()*coef(0);
        for (size_t i=0; i < basis_Rck_F.dimension(); i++){
          error_colinearity += std::abs( (basis_Rck_F.function(i, x).cross(x-F.center_mass())).norm() / basis_Rck_F.function(i, x).norm() );
        }
      }
      if (error_colinearity>tol){
        std::cout << "Basis of RckF not colinear with x, error=" << error_colinearity << std::endl;
        exit(1);
      }
      
      // Compute values on quadrature points
      auto basis_Rck_F_quad = evaluate_quad<Function>::compute(basis_Rck_F, qr_2k_F);
      Eigen::MatrixXd MassRck_F = compute_gram_matrix(basis_Rck_F_quad, basis_Rck_F_quad, qr_2k_F);
  //    std::cout << "Mass matrix:\n " << MassRck_F << std::endl;

      // Orthonormalise and re-test
      Family<RckFBasisType> on_basis_Rck_F = l2_orthonormalize(basis_Rck_F, qr_2k_F, basis_Rck_F_quad);
      error_colinearity = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        Eigen::Vector2d coef = Eigen::Vector2d::Random();
        VectorRd x = F.center_mass() + F.edge_normal(0)*coef(0) + F.edge(0)->tangent()*coef(0);
        for (size_t i=0; i < basis_Rck_F.dimension(); i++){
          error_colinearity += std::abs( (on_basis_Rck_F.function(i, x).cross(x-F.center_mass())).norm() / on_basis_Rck_F.function(i, x).norm());
        }
      }
      if (error_colinearity>tol){
        std::cout << "ON basis of RckF not colinear with x, error=" << error_colinearity << std::endl;
        exit(1);
      }

      Eigen::MatrixXd on_MassRck_F = compute_gram_matrix(basis_Rck_F_quad, basis_Rck_F_quad, qr_2k_F);
  //    std::cout << "ON Mass matrix:\n " << on_MassRck_F << std::endl;
      
      // Store
      basis_Rck_Fh.push_back(basis_Rck_F);

    }
    
    //------------------------------------------------------------------------------
    // Basis for Gck(F)
    //------------------------------------------------------------------------------
    if (K>=1){
      GckFBasisType basis_Gck_F(F, K);
      std::cout << FORMAT(45) << "Basis dimension for Gck(F)" << basis_Gck_F.dimension() << std::endl;
      if (basis_Gck_F.dimension() != basis_Pk2_F.dimension() - on_basis_Gk_F.dimension()) {
        std::cerr << "ERROR: Inconsistent basis dimension for Gck(F). Expected "
                  << (basis_Pk2_F.dimension() - on_basis_Rk_F.dimension())
                  << " got " << basis_Gck_F.dimension();
        exit(1);
      }
      // Test orthogonality at a few points
      double error_orthogonality = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        Eigen::Vector2d coef = Eigen::Vector2d::Random();
        VectorRd x = F.center_mass() + F.edge_normal(0)*coef(0) + F.edge(0)->tangent()*coef(0);
        for (size_t i=0; i < basis_Gck_F.dimension(); i++){
          error_orthogonality += std::abs( basis_Gck_F.function(i, x).dot(x-F.center_mass()) / (basis_Gck_F.function(i, x).norm() * (x-F.center_mass()).norm()) );
        }
      }
      if (error_orthogonality>tol){
        std::cout << "Basis of GckF not orthogonal with x-xF, error=" << error_orthogonality << std::endl;
        exit(1);
      }

      // Compute values on quadrature points
      auto basis_Gck_F_quad = evaluate_quad<Function>::compute(basis_Gck_F, qr_2k_F);
      Eigen::MatrixXd MassGck_F = compute_gram_matrix(basis_Gck_F_quad, basis_Gck_F_quad, qr_2k_F);
  //    std::cout << "Mass matrix:\n " << MassGck_F << std::endl;
      
      // Orthonormalise and re-test
      Family<GckFBasisType> on_basis_Gck_F = l2_orthonormalize(basis_Gck_F, qr_2k_F, basis_Gck_F_quad);
      error_orthogonality = 0;
      for (size_t dummy=0; dummy < 10; dummy++){
        Eigen::Vector2d coef = Eigen::Vector2d::Random();
        VectorRd x = F.center_mass() + F.edge_normal(0)*coef(0) + F.edge(0)->tangent()*coef(0);
        for (size_t i=0; i < basis_Gck_F.dimension(); i++){
          error_orthogonality += std::abs( on_basis_Gck_F.function(i, x).dot(x-F.center_mass()) / (on_basis_Gck_F.function(i, x).norm() * (x-F.center_mass()).norm()) );
        }
      }
      if (error_orthogonality>tol){
        std::cout << "ON basis of GckF not orthogonal with x-xF, error=" << error_orthogonality << std::endl;
        exit(1);
      }
      Eigen::MatrixXd on_MassGck_F = compute_gram_matrix(basis_Gck_F_quad, basis_Gck_F_quad, qr_2k_F);
  //    std::cout << "ON Mass matrix:\n " << on_MassGck_F << std::endl;

      // Store
      basis_Gck_Fh.push_back(basis_Gck_F);
    
    }
        
    // Store computed bases
    basis_Pk_Fh.push_back(on_basis_Pk_F);
    basis_Rk_Fh.push_back(on_basis_Rk_F);
  } // for iF

  // Print projection errors
  std::cout << FORMAT(15) << "err_l2_Pk_Fh = " << std::scientific << std::sqrt(err_l2_Pk_Fh) << std::endl;
  std::cout << FORMAT(15) << "err_l2_Rk_Fh = " << std::scientific << std::sqrt(err_l2_Rk_Fh) << std::endl;
  std::cout << FORMAT(15) << "err_l2_ROk_Fh = " << std::scientific << std::sqrt(err_l2_ROk_Fh) << std::endl;

  std::cout << "Done" << std::endl;
  
  return 0;
}

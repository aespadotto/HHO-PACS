// Authors: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr), Jerome Droniou (jerome.droniou@monash.edu)
#include <fstream>
#include <iomanip>
#include <thread>

#include "ddr-stokes.hpp"
#include <parallel_for.hpp>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#ifdef WITH_UMFPACK
#include <Eigen/UmfPackSupport>
#endif

#ifdef WITH_MKL
#include <Eigen/PardisoSupport>
#endif

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
    ("solution,s", boost::program_options::value<int>()->default_value(0), "Select the solution")
    ("pressure_scaling", boost::program_options::value<double>()->default_value(.1), "Select the pressure scaling")
    ("export-matrix,e", "Export matrix to Matrix Market format")
    ("iterative-solver,i", "Use iterative linear solver")
    ("stabilization-parameter,x", boost::program_options::value<double>(), "Set the stabilization parameter");

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

  std::cout << "[main] Mesh file: " << mesh_file << std::endl;
  
  // Select the degree 
  size_t K = vm["degree"].as<size_t>();
  std::cout << FORMAT(25) << "[main] Degree" << K << std::endl;

  // Select the solution
  int solution = (vm.count("solution") ? vm["solution"].as<int>() : 0);
  Stokes::ForcingTermType f;
  Stokes::VelocityType u;
  Stokes::VorticityType omega;
  Stokes::PressureType p;
  Stokes::PressureGradientType grad_p;
  Stokes::ViscosityType nu(1.);
  
  pressure_scaling = vm["pressure_scaling"].as<double>();

  switch (solution) {
  case 0:
    std::cout << "[main] Trigonometric solution" << std::endl;
    f = trigonometric_f;
    u = trigonometric_u;
    omega = trigonometric_curl_u;
    p = trigonometric_p;
    grad_p = trigonometric_grad_p;
    nu = trigonometric_nu;
    break;

  case 1:
    std::cout << "[main] Linear solution" << std::endl;
    f = linear_f;
    u = linear_u;
    omega = linear_curl_u;
    p = linear_p;
    grad_p = linear_grad_p;
    nu = linear_nu;
    break;

  case 2:
    std::cout << "[main] Linear velocity, trigonometric pressure" << std::endl;
    f = trigonometric_grad_p;
    u = linear_u;
    omega = linear_curl_u;
    p = trigonometric_p;
    grad_p = trigonometric_grad_p;
    nu = linear_nu;
    break;

  case 3:
    std::cout << "[main] Field solution" << std::endl;
    f = field_f;
    u = field_u;
    omega = field_curl_u;
    p = field_p;
    grad_p = field_grad_p;
    nu = field_nu;
    break;

  default:
    std::cerr << "[main] ERROR: Unknown exact solution" << std::endl;
    exit(1);    
  }


  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  boost::timer::cpu_timer timer;
  // Create DDR core
  timer.start();
  bool use_threads = (vm.count("pthread") ? vm["pthread"].as<bool>() : true);
  std::cout << "[main] " << (use_threads ? "Parallel execution" : "Sequential execution") << std:: endl;
  DDRCore ddr_core(*mesh_ptr, K, use_threads);
  timer.stop();
  double t_wall_ddrcore = double(timer.elapsed().wall) * pow(10, -9);
  double t_proc_ddrcore = double(timer.elapsed().user + timer.elapsed().system) * pow(10, -9);
  std::cout << "[main] Time DDRCore (wall/proc) " << t_wall_ddrcore << "/" << t_proc_ddrcore << std::endl;

  // Assemble the problem
  timer.start();
  Stokes st(ddr_core, use_threads);  
  if(vm.count("stabilization-parameter")) {
    st.stabilizationParameter() = vm["stabilization-parameter"].as<double>();
  }
  st.assembleLinearSystem(f, u, omega, nu);
  timer.stop();
  double t_wall_model = double(timer.elapsed().wall) * pow(10, -9);
  double t_proc_model = double(timer.elapsed().user + timer.elapsed().system) * pow(10, -9);
  std::cout << "[main] Time model (wall/proc) " << t_wall_model << "/" << t_proc_model << std::endl;

  // Export matrix if requested  
  if (vm.count("export-matrix")) {
    std::cout << "[main] Exporting matrix to Matrix Market format" << std::endl;
    saveMarket(st.systemMatrix(), "A_stokes.mtx");
    saveMarket(st.systemVector(), "b_stokes.mtx");
  }


  // Solve the problem
  timer.start();
  Eigen::VectorXd uph;
  if (vm.count("iterative-solver")) {
    std::cout << "[main] Solving the linear system using BiCGSTAB" << std::endl;
    
    Eigen::BiCGSTAB<Stokes::SystemMatrixType, Eigen::IncompleteLUT<double> > solver;
    // solver.preconditioner().setFillfactor(2);
    solver.compute(st.systemMatrix());
    if (solver.info() != Eigen::Success) {
      std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
      exit(1);
    }
    uph = solver.solve(st.systemVector()).head(st.dimension());
    if (solver.info() != Eigen::Success) {
      std::cerr << "[main] ERROR: Could not solve direct system" << std::endl;
      exit(1);
    }
  } else { 
#ifdef WITH_MKL
    std::cout << "[main] Solving the linear system using Pardiso" << std::endl;    
    Eigen::PardisoLU<Stokes::SystemMatrixType> solver;
#elif WITH_UMFPACK
    std::cout << "[main] Solving the linear system using Umfpack" << std::endl;    
    Eigen::UmfPackLU<Stokes::SystemMatrixType> solver;
#else
    std::cout << "[main] Solving the linear system using direct solver" << std::endl;    
    Eigen::SparseLU<Stokes::SystemMatrixType> solver;
#endif
    solver.compute(st.systemMatrix());
    if (solver.info() != Eigen::Success) {
      std::cerr << "[main] ERROR: Could not factorize matrix" << std::endl;
    }
    uph = solver.solve(st.systemVector()).head(st.dimension());
    if (solver.info() != Eigen::Success) {
      std::cerr << "[main] ERROR: Could not solve linear system" << std::endl;
    }
  }
  timer.stop();
  double t_wall_solve = double(timer.elapsed().wall) * pow(10, -9);
  double t_proc_solve = double(timer.elapsed().user + timer.elapsed().system) * pow(10, -9);
  std::cout << "[main] Time solve (wall/proc) " << t_wall_solve << "/" << t_proc_solve << std::endl;
  
  // Interpolate of exact solution and error vector
  Eigen::VectorXd upI = Eigen::VectorXd::Zero(st.dimension());  
  upI.head(st.xCurl().dimension()) = st.xCurl().interpolate(u);
  upI.tail(st.xGrad().dimension()) = st.xGrad().interpolate(p);
  Eigen::VectorXd eph = uph - upI;
  // Errors in Hcurl and Hgrad norms
  StokesNorms norms = st.computeStokesNorms(upI);
  StokesNorms errors = st.computeStokesNorms(eph);
  std::cout << "[main] Hcurl norm u= " << norms.hcurl_u << "; Hgrad norm p= " << norms.hgrad_p <<std::endl;
  double error_hcurl_u = errors.hcurl_u / (norms.hcurl_u + pow(10, -12));
  double error_hgrad_p = errors.hgrad_p / (norms.hgrad_p + pow(10, -12));
  std::cout << "[main] Hcurl error u= " << error_hcurl_u << "; Hgrad error p= " << error_hgrad_p << std::endl;
  std::cout << "[main] Mesh diameter " << mesh_ptr->h_max() << std::endl;
 
  // Comment out to avoid evaluation of defect of commutation with the chosen quadrature degree (which should match the degree used to interpolate f in assembleLinearSystem
//  std::cout << "[main | DEBUG] Defect commutation= " << st.evaluateDefectCommutationGradInterp(p, grad_p, 2*st.xCurl().degree()+8) << std::endl;
   
  // Write results to file
  double eps = std::pow(10, -10);
  std::ofstream out("results.txt");
  out << "Solution: " << solution << std::endl;
  out << "Mesh: " << mesh_file << std::endl;
  out << "Degree: " << K << std::endl;
  out << "MeshSize: " << mesh_ptr->h_max() << std::endl;
  out << "NbCells: " << mesh_ptr->n_cells() << std::endl;
  out << "NbFaces: " << mesh_ptr->n_faces() << std::endl;
  out << "NbEdges: " << mesh_ptr->n_edges() << std::endl;
  out << "DimXCurl: " << st.xCurl().dimension() << std::endl;
  out << "DimXGrad: " << st.xGrad().dimension() << std::endl;
  out << "E_HcurlVel: " << error_hcurl_u << std::endl;
  out << "E_HgradPre: " << error_hgrad_p << std::endl;
  out << "E_L2Vel: " << errors.u/(norms.u+eps) << std::endl;
  out << "E_L2CurlVel: " << errors.curl_u/(norms.curl_u+eps) << std::endl;
  out << "E_L2Pre: " << errors.p/(norms.p+eps) << std::endl;
  out << "E_L2GradPre: " << errors.grad_p/(norms.grad_p+eps) << std::endl;
  out << "TwallDDRCore: " << t_wall_ddrcore << std::endl;  
  out << "TprocDDRCore: " << t_proc_ddrcore << std::endl;  
  out << "TwallModel: " << t_wall_model << std::endl;  
  out << "TprocModel: " << t_proc_model << std::endl;  
  out << "TwallSolve: " << t_wall_solve << std::endl;  
  out << "TprocSolve: " << t_proc_solve << std::endl;  
  out << std::flush;
  out.close();

  std::cout << "[main] Done" << std::endl;
  return 0;
}

//------------------------------------------------------------------------------
// Stokes
//------------------------------------------------------------------------------

Stokes::Stokes(
               const DDRCore & ddrcore,
               bool use_threads,
               std::ostream & output
               )
  : m_ddrcore(ddrcore),
    m_use_threads(use_threads),
    m_output(output),
    m_xgrad(ddrcore, use_threads),
    m_xcurl(ddrcore, use_threads),
    m_xdiv(ddrcore, use_threads),
    m_A(dimension()+1, dimension()+1),  // System size is dimension+1 because of Lagrange multiplier for zero-average constraint on pressure
    m_b(Eigen::VectorXd::Zero(dimension()+1)),
    m_stab_par(1.)    
{
  m_output << "[Stokes] Initializing" << std::endl;
}

//------------------------------------------------------------------------------

void Stokes::assembleLinearSystem(
                                  const ForcingTermType & f,
                                  const VelocityType & u,
                                  const VorticityType & omega,
                                  const ViscosityType & nu
                                  )
{
  // Interpolate of forcing term in XCurl
  Eigen::VectorXd interp_f = m_xcurl.interpolate(f, 2*m_xcurl.degree()+8);
    
  // Assemble all local contributions
  auto assemble_all = [this, interp_f, u, nu](
                                       size_t start,
                                       size_t end,
                                       std::list<Eigen::Triplet<double> > * my_triplets,
                                       Eigen::VectorXd * my_rhs
                                       )->void
                      {
                        for (size_t iT = start; iT < end; iT++) {
                          this->_assemble_local_contribution(
                                                         iT,
                                                         this->_compute_local_contribution(iT, interp_f, nu),
                                                         *my_triplets,
                                                         *my_rhs
                                                         );
                        } // for iT
                      };
                      
  // Assemble the matrix and rhs
  if (m_use_threads) {
    m_output << "[Stokes] Parallel assembly" << std::endl;
  }else{
    m_output << "[Stokes] Sequential assembly" << std::endl;
  }
  std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> 
      system = parallel_assembly_system(m_ddrcore.mesh().n_cells(), this->dimension()+1, assemble_all, m_use_threads);
  m_A = system.first;
  m_b = system.second;

  // Assemble boundary conditions
  for (auto iF : m_ddrcore.mesh().get_b_faces()) {
    const Face & F = *iF;
    
    // Unit normal vector to F pointing out of the domain
    const Cell & TF = *F.cell(0);
    Eigen::Vector3d nF = TF.face_normal(TF.index_face(&F));

    // Degree of quadratures to compute boundary conditions
    const size_t dqrbc = 2 * m_ddrcore.degree() + 3;

    // Boundary condition on the tangential component of the vorticity
    {
      FType<Eigen::Vector3d> omega_cross_nF = [&omega, &nF](const Eigen::Vector3d & x) {
                                                        return omega(x).cross(nF);
                                                        };

      QuadratureRule quad_dqrbc_F = generate_quadrature_rule(F, dqrbc);
      Eigen::VectorXd bF = integrate(omega_cross_nF, evaluate_quad<Function>::compute(*m_ddrcore.faceBases(F.global_index()).Polyk2, quad_dqrbc_F), quad_dqrbc_F).transpose() * m_xcurl.faceOperators(F.global_index()).potential;
      auto I_F = m_xcurl.globalDOFIndices(F);
      for (size_t i = 0; i < I_F.size(); i++) {
        m_b(I_F[i]) += bF(i);
      } // for i
    }

    // Boundary condition on the normal component of the velocity    
    {
      FType<double> u_dot_nF = [&u, &nF](const Eigen::Vector3d & x) {
                                               return u(x).dot(nF);
                                               };

      QuadratureRule quad_dqrbc_F = generate_quadrature_rule(F, dqrbc);
      Eigen::VectorXd bF = - integrate(u_dot_nF, evaluate_quad<Function>::compute(*m_ddrcore.faceBases(F.global_index()).Polykpo, quad_dqrbc_F), quad_dqrbc_F).transpose() * m_xgrad.faceOperators(F.global_index()).potential;
      auto I_F = m_xgrad.globalDOFIndices(F);
      size_t dim_xcurl = m_xcurl.dimension();
      for (size_t i = 0; i < I_F.size(); i++) {
        m_b(dim_xcurl + I_F[i]) += bF(i);
      } // for i
    }
  } // for iF
}

//------------------------------------------------------------------------------
 
std::pair<Eigen::MatrixXd, Eigen::VectorXd>
    Stokes::_compute_local_contribution(
                                        size_t iT, 
                                        const Eigen::VectorXd & interp_f,
                                        const ViscosityType & nu
                                        )
{
  const Cell & T = *m_ddrcore.mesh().cell(iT);

  size_t dim_xcurl_T = m_xcurl.dimensionCell(iT);
  size_t dim_xgrad_T = m_xgrad.dimensionCell(iT);
  size_t dim_T = dim_xcurl_T + dim_xgrad_T;   
  
  Eigen::MatrixXd AT = Eigen::MatrixXd::Zero(dim_T+1, dim_T+1);  // +1 for Lagrange multiplier
  Eigen::VectorXd lT = Eigen::VectorXd::Zero(dim_T+1);

  //------------------------------------------------------------------------------
  // Local matrix
  //------------------------------------------------------------------------------

  // Mass matrix for (P^k(T))^3  
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * m_ddrcore.degree());
  Eigen::MatrixXd mass_Pk3_T = compute_gram_matrix(evaluate_quad<Function>::compute(*m_ddrcore.cellBases(iT).Polyk3, quad_2k_T), quad_2k_T);

  // aT
  AT.topLeftCorner(dim_xcurl_T, dim_xcurl_T) = m_xdiv.computeL2ProductCurl(iT, m_xcurl, "both", m_stab_par, mass_Pk3_T, nu);
  
  // bT
  Eigen::MatrixXd BT = m_xcurl.computeL2ProductGradient(iT, m_xgrad, "right", m_stab_par, mass_Pk3_T); 
  AT.block(0, dim_xcurl_T, dim_xcurl_T, dim_xgrad_T ) = BT;
  AT.block(dim_xcurl_T, 0, dim_xgrad_T, dim_xcurl_T ) = -BT.transpose();

  // Lagrange multiplier: enforce \int P^{k+1}p = 0
  FType<double> cst_fct_one = [](const Eigen::Vector3d &x) -> double { return 1.0; };
  QuadratureRule quad_kpo_T = generate_quadrature_rule(T, m_ddrcore.degree()+1 );
  // intPko is line vector representing \int_T P^{k+1}
  Eigen::RowVectorXd intPkpo = integrate(cst_fct_one, evaluate_quad<Function>::compute(*m_ddrcore.cellBases(iT).Polykpo, quad_kpo_T), quad_kpo_T).transpose()
    * m_xgrad.cellOperators(iT).potential;

  AT.block(dim_T, dim_xcurl_T, 1, dim_xgrad_T) = intPkpo;
  AT.block(dim_xcurl_T, dim_T, dim_xgrad_T, 1) = -intPkpo.transpose();
    
  //------------------------------------------------------------------------------
  // Local source vector
  //------------------------------------------------------------------------------
  lT.head(dim_xcurl_T) = m_xcurl.computeL2Product(iT, m_stab_par, mass_Pk3_T) * m_xcurl.restrictCell(iT, interp_f);

  return std::make_pair(AT, lT);
}

//------------------------------------------------------------------------------

void Stokes::_assemble_local_contribution(
                                          size_t iT,
                                          const std::pair<Eigen::MatrixXd, Eigen::VectorXd> & lsT,
                                          std::list<Eigen::Triplet<double> > & my_triplets,
                                          Eigen::VectorXd & my_rhs
                                          )
{
  const Cell & T = *m_ddrcore.mesh().cell(iT);

  size_t dim_T = m_xcurl.dimensionCell(iT) + m_xgrad.dimensionCell(iT);
  size_t dim_xcurl = m_xcurl.dimension();

  // Create the vector of DOF indices
  auto I_xcurl_T = m_xcurl.globalDOFIndices(T);
  auto I_xgrad_T = m_xgrad.globalDOFIndices(T);
  std::vector<size_t> I_T(dim_T+1);
  auto it_I_T = std::copy(I_xcurl_T.begin(), I_xcurl_T.end(), I_T.begin());
  std::transform(I_xgrad_T.begin(), I_xgrad_T.end(), it_I_T, [&dim_xcurl](const size_t & index) { return index + dim_xcurl; });
  I_T[dim_T] = dimension();

  // Assemble
  const Eigen::MatrixXd & AT = lsT.first;
  const Eigen::VectorXd & bT = lsT.second;
  for (size_t i = 0; i < dim_T+1; i++) {
    my_rhs(I_T[i]) += bT(i);
    for (size_t j = 0; j < dim_T+1; j++) {
      my_triplets.push_back( Eigen::Triplet<double>(I_T[i], I_T[j], AT(i,j)) );
    } // for j
  } // for i
}


//------------------------------------------------------------------------------

StokesNorms Stokes::computeStokesNorms( const Eigen::VectorXd & v ) const
{
  const size_t ncells = m_ddrcore.mesh().n_cells();
  Eigen::VectorXd local_sqnorm_u = Eigen::VectorXd::Zero(ncells);
  Eigen::VectorXd local_sqnorm_curl_u = Eigen::VectorXd::Zero(ncells);
  Eigen::VectorXd local_sqnorm_p = Eigen::VectorXd::Zero(ncells);
  Eigen::VectorXd local_sqnorm_grad_p = Eigen::VectorXd::Zero(ncells);

  // Xcurl correspond to the first components of v, Xgrad to the last ones
  Eigen::VectorXd v_curl = v.head(m_xcurl.dimension());
  Eigen::VectorXd v_grad = v.tail(m_xgrad.dimension());
  
  std::function<void(size_t, size_t)> compute_local_squarednorms
    = [this, &v_curl, &v_grad, &local_sqnorm_u, &local_sqnorm_curl_u, &local_sqnorm_p, &local_sqnorm_grad_p](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        Cell & T = *m_ddrcore.mesh().cell(iT);
        // Mass matrix for (P^k(T))^3
        QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2*m_ddrcore.degree() );
        Eigen::MatrixXd mass_Pk3_T = compute_gram_matrix(evaluate_quad<Function>::compute(*m_ddrcore.cellBases(iT).Polyk3, quad_2k_T), quad_2k_T);
        Eigen::VectorXd v_curl_T = m_xcurl.restrict(T, v_curl);
        Eigen::VectorXd v_grad_T = m_xgrad.restrict(T, v_grad);

        // Contribution of L2 norms, without any weight (no viscosity)
        local_sqnorm_u(iT) = v_curl_T.transpose() * m_xcurl.computeL2Product(iT, m_stab_par, mass_Pk3_T) * v_curl_T;
        local_sqnorm_p(iT) = v_grad_T.transpose() * m_xgrad.computeL2Product(iT, m_stab_par) * v_grad_T;

        // Contribution of L2 norms of curl and grad
        local_sqnorm_curl_u(iT) = v_curl_T.transpose() * m_xdiv.computeL2ProductCurl(iT, m_xcurl, "both", m_stab_par, mass_Pk3_T) * v_curl_T;
        Eigen::MatrixXd GT = m_xgrad.cellOperators(iT).gradient;
        QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (m_ddrcore.degree() + 1) );
        local_sqnorm_grad_p(iT) = v_grad_T.transpose() * GT.transpose() * mass_Pk3_T * GT * v_grad_T;
      }
    };
  parallel_for(ncells, compute_local_squarednorms, m_use_threads);
  
  double sqnorm_u = local_sqnorm_u.sum();
  double sqnorm_curl_u = local_sqnorm_curl_u.sum();
  double sqnorm_p = local_sqnorm_p.sum();
  double sqnorm_grad_p = local_sqnorm_grad_p.sum();

  return StokesNorms(std::sqrt(std::abs(sqnorm_u)), std::sqrt(std::abs(sqnorm_curl_u)), std::sqrt(std::abs(sqnorm_p)), std::sqrt(std::abs(sqnorm_grad_p)));
}

//------------------------------------------------------------------------------

double Stokes::evaluateDefectCommutationGradInterp(
               const PressureType & p,
               const PressureGradientType & grad_p,
               const size_t deg_quad_interpolate
              ) const
{
  
  // Compute interpolates of pressure and gradient
  Eigen::VectorXd interp_p = m_xgrad.interpolate(p, 14);
  Eigen::VectorXd interp_grad_p = m_xcurl.interpolate(grad_p, deg_quad_interpolate);
  
  // Compute the norms of difference GT (Igrad p) - Icurl (grad p) in each cell
  const size_t ncells = m_ddrcore.mesh().n_cells();
  Eigen::VectorXd local_sqerrors = Eigen::VectorXd::Zero(ncells);
  
  std::function<void(size_t, size_t)> compute_local_sqerrors
    = [this, &interp_p, &interp_grad_p, &local_sqerrors](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        Cell & T = *m_ddrcore.mesh().cell(iT);
        // Mass matrix for (P^k(T))^3
        QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2*m_ddrcore.degree() );
        Eigen::MatrixXd mass_Pk3_T = compute_gram_matrix(evaluate_quad<Function>::compute(*m_ddrcore.cellBases(iT).Polyk3, quad_2k_T), quad_2k_T);
        Eigen::VectorXd interp_p_T = m_xgrad.restrict(T, interp_p);
        Eigen::VectorXd interp_grad_p_T = m_xcurl.restrict(T, interp_grad_p);

        // Contribution of L2 norm is done by developping ||GT p_T - Icurl(grad p)||^2
        local_sqerrors(iT) = interp_p_T.transpose() * m_xcurl.computeL2ProductGradient(iT, m_xgrad, "both", m_stab_par, mass_Pk3_T) * interp_p_T;
        local_sqerrors(iT) += - 2 * interp_p_T.transpose() * m_xcurl.computeL2ProductGradient(iT, m_xgrad, "left", m_stab_par, mass_Pk3_T) * interp_grad_p_T;
        local_sqerrors(iT) += interp_grad_p_T.transpose() * m_xcurl.computeL2Product(iT, m_stab_par, mass_Pk3_T) * interp_grad_p_T;
      } // for
    };
  parallel_for(ncells, compute_local_sqerrors, m_use_threads);
  
  return std::sqrt( std::abs(local_sqerrors.sum()) );
}


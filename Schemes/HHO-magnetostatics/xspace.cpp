#include "xspace.hpp"

#include <parallel_for.hpp>
#include <polynomialspacedimension.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XSpace::XSpace(
               const Mesh & mesh,
               size_t K,
               bool use_threads,
               std::ostream & output
               )
  : HHOSpace(mesh, 2 * PolynomialSpaceDimension<Face>::Poly(K), 3 * PolynomialSpaceDimension<Cell>::Poly(K)),
    m_use_threads(use_threads),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_face_bases(mesh.n_faces()),
    m_curl_reconstruction_bases(mesh.n_cells()),
    m_cell_curls(mesh.n_cells()),
    m_cell_curls_rhs(mesh.n_cells())
{
  m_output << "[XSpace] Initializing" << std::endl;

  // Construct element bases
  std::function<void(size_t, size_t)> construct_all_cell_bases
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *this->mesh().cell(iT);
          
          MonomialScalarBasisCell basis_Pk_T(T, m_K);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * m_K);
          auto basis_Pk_T_quad = evaluate_quad<Function>::compute(basis_Pk_T, quad_2k_T);
        
          this->m_cell_bases[iT].reset( new CellBasisType(l2_orthonormalize(basis_Pk_T, quad_2k_T, basis_Pk_T_quad)) );
          
          MonomialScalarBasisCell basis_Pkmo_T(T, m_K - 1);
          QuadratureRule quad_2kmo_T = generate_quadrature_rule(T, 2 * (m_K - 1));
          auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(basis_Pkmo_T, quad_2kmo_T);
          this->m_curl_reconstruction_bases[iT].reset( new CurlReconstructionBasisType(l2_orthonormalize(basis_Pkmo_T, quad_2kmo_T, basis_Pkmo_T_quad)) );
        } // for iT
      };

  m_output << "[XSpace] Constructing element bases" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_cell_bases, m_use_threads);
  
  // Construct face bases
  std::function<void(size_t, size_t)> construct_all_face_bases
    = [this](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *this->mesh().face(iF);

          MonomialScalarBasisFace basis_Pk_F(F, m_K);
          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * m_K);
          auto basis_Pk_F_quad = evaluate_quad<Function>::compute(basis_Pk_F, quad_2k_F);
          
          this->m_face_bases[iF].reset( new FaceBasisType(l2_orthonormalize(basis_Pk_F, quad_2k_F, basis_Pk_F_quad), basis_Pk_F.jacobian() ) );
        } // for iF
      };
  
  m_output << "[XSpace] Constructing face bases" << std::endl;
  parallel_for(mesh.n_faces(), construct_all_face_bases, m_use_threads);

  // Construct element curls
  std::function<void(size_t, size_t)> construct_all_curls
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *this->mesh().cell(iT);

          //------------------------------------------------------------------------------
          // Left-hand side matrix

          QuadratureRule quad_2kmo_T = generate_quadrature_rule(T, 2 * (degree() - 1));
          Eigen::MatrixXd MCT = compute_gram_matrix(
                                                    evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2kmo_T),
                                                    quad_2kmo_T
                                                    );

          //------------------------------------------------------------------------------
          // Right-hand side matrix

          Eigen::MatrixXd BCT = Eigen::MatrixXd::Zero(3 * PolynomialSpaceDimension<Cell>::Poly(degree() - 1), dimensionCell(iT));

          // Boundary contribution
          for (size_t iF = 0; iF < T.n_faces(); iF++) {
            const Face & F = *T.face(iF);
            
            Eigen::Vector3d nF = F.normal();
            QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
            BCT.block(0, iF * 2 * PolynomialSpaceDimension<Face>::Poly(degree()), 3 * PolynomialSpaceDimension<Cell>::Poly(degree() - 1), 2 * PolynomialSpaceDimension<Face>::Poly(degree()))
              += T.face_orientation(iF) * compute_gram_matrix(
                                                              vector_product(evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2k_F), nF),
                                                              evaluate_quad<Function>::compute(faceBasis(F.global_index()), quad_2k_F),
                                                              quad_2k_F
                                                              );
          } // for iF
          
          // Element contribution
          BCT.bottomRightCorner(3 * PolynomialSpaceDimension<Cell>::Poly(degree() - 1), 3 * PolynomialSpaceDimension<Cell>::Poly(degree()))
            += compute_gram_matrix(
                                   evaluate_quad<Curl>::compute(curlReconstructionBasis(iT), quad_2kmo_T),
                                   evaluate_quad<Function>::compute(cellBasis(iT), quad_2kmo_T),
                                   quad_2kmo_T
                                   );

          m_cell_curls_rhs[iT] = BCT;
          m_cell_curls[iT] = MCT.ldlt().solve(BCT);
        } // for iT
      };

  m_output << "[XSpace] Computing curl reconstructions" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_curls, m_use_threads);
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XSpace::interpolate(const FunctionType & v) const
{
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(dimension());

  interpolate(v, vh);

  return vh;
}

//------------------------------------------------------------------------------

void XSpace::interpolate(const FunctionType & v, Eigen::VectorXd & vh) const
{
  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &vh, v](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *mesh().face(iF);

          Eigen::Vector3d nF = F. normal();
          auto nF_cross_v_cross_nF = [&nF, v](const Eigen::Vector3d & x)->Eigen::Vector3d {
                                       return nF.cross(v(x).cross(nF));
                                     };
          
          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
          vh.segment(globalOffset(F), 2 * PolynomialSpaceDimension<Face>::Poly(degree()))
            = l2_projection(nF_cross_v_cross_nF, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute(faceBasis(iF), quad_2k_F));
        } // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &vh, v](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);

          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * degree());
          vh.segment(globalOffset(T), 3 * PolynomialSpaceDimension<Cell>::Poly(degree()))
            = l2_projection(v, cellBasis(iT), quad_2k_T, evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_T));
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
}

//------------------------------------------------------------------------------

Eigen::VectorXd XSpace::interpolateFaces(const FunctionType & v) const
{
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(mesh().n_faces() * 2 * PolynomialSpaceDimension<Face>::Poly(degree()));

  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &vh, v](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *mesh().face(iF);

          Eigen::Vector3d nF = F. normal();
          auto nF_cross_v_cross_nF = [&nF, v](const Eigen::Vector3d & x)->Eigen::Vector3d {
                                       return nF.cross(v(x).cross(nF));
                                     };
          
          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
          vh.segment(iF * 2 * PolynomialSpaceDimension<Face>::Poly(degree()), 2 * PolynomialSpaceDimension<Face>::Poly(degree()))
            = l2_projection(nF_cross_v_cross_nF, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute(faceBasis(iF), quad_2k_F));
        } // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  return vh;
}

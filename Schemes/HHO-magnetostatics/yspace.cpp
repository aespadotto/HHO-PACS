#include "yspace.hpp"

#include <parallel_for.hpp>
#include <polynomialspacedimension.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------


YSpace::YSpace(
	       const Mesh & mesh,
	       size_t K,
	       bool use_threads,
	       std::ostream & output
	       )
  : HHOSpace(mesh, PolynomialSpaceDimension<Face>::Poly(K), PolynomialSpaceDimension<Cell>::Poly(K-1)),
    m_use_threads(use_threads),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_face_bases(mesh.n_faces()),
    m_gradient_reconstruction_bases(mesh.n_cells()),
    m_cell_gradients(mesh.n_cells()),
    m_cell_gradients_rhs(mesh.n_cells())
{
  m_output << "[YSpace] Initializing" << std::endl;

  // Construct element bases
  std::function<void(size_t, size_t)> construct_all_cell_bases
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *this->mesh().cell(iT);
          
          MonomialScalarBasisCell basis_Pk_T(T, m_K-1);
          MonomialScalarBasisCell basis_Pk1_T(T, m_K);
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * m_K);
          auto basis_Pk_T_quad = evaluate_quad<Function>::compute(basis_Pk_T, quad_2k_T);
          auto basis_Pk1_T_quad = evaluate_quad<Function>::compute(basis_Pk1_T, quad_2k_T);
        
          this->m_cell_bases[iT].reset( new CellBasisType(l2_orthonormalize(basis_Pk_T, quad_2k_T, basis_Pk_T_quad) ) );
          this->m_gradient_reconstruction_bases[iT].reset( new GradientReconstructionBasisType(l2_orthonormalize(basis_Pk1_T, quad_2k_T, basis_Pk1_T_quad) ) );
        } // for iT
      };

  m_output << "[YSpace] Constructing element bases" << std::endl;
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
          
          this->m_face_bases[iF].reset( new FaceBasisType(l2_orthonormalize(basis_Pk_F, quad_2k_F, basis_Pk_F_quad) ) );
        } // for iF
      };
  
  m_output << "[YSpace] Constructing face bases" << std::endl;
  parallel_for(mesh.n_faces(), construct_all_face_bases, m_use_threads);

  // Construct element gradients
  std::function<void(size_t, size_t)> construct_all_gradients
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *this->mesh().cell(iT);

          // Left-hand side matrix
          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * degree());
          Eigen::MatrixXd MGT = compute_gram_matrix(
                                                    evaluate_quad<Function>::compute(gradientReconstructionBasis(iT), quad_2k_T),
                                                    quad_2k_T
                                                    );

          // Right-hand side matrix          
          Eigen::MatrixXd BGT = Eigen::MatrixXd::Zero(3 * PolynomialSpaceDimension<Cell>::Poly(degree()), dimensionCell(iT));
          
          // Boundary contribution
          for (size_t iF = 0 ; iF < T.n_faces(); iF++) {
            const Face & F = *T.face(iF);

            QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
            BGT.block(0, localOffset(T, F), 3 * PolynomialSpaceDimension<Cell>::Poly(degree()), numLocalDofsFace())
              += compute_gram_matrix(
                                     scalar_product(evaluate_quad<Function>::compute(gradientReconstructionBasis(iT), quad_2k_F), T.face_normal(iF)),
                                     evaluate_quad<Function>::compute(faceBasis(F.global_index()), quad_2k_F),
                                     quad_2k_F
                                     );
          } // for iF
          
          // Element contribution
          BGT.block(0, localOffset(T), 3 * PolynomialSpaceDimension<Cell>::Poly(degree()), numLocalDofsCell())
            -= compute_gram_matrix(
                                   evaluate_quad<Divergence>::compute(gradientReconstructionBasis(iT), quad_2k_T),
                                   evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_T),
                                   quad_2k_T
                                   );

          // Solve the local problem
          m_cell_gradients_rhs[iT] = BGT;
          m_cell_gradients[iT] = MGT.ldlt().solve(BGT);
        } // for iT
      };

  m_output << "[YSpace] Computing gradient reconstructions" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_gradients, m_use_threads);
}

//------------------------------------------------------------------------------
// Interpolators
//------------------------------------------------------------------------------

Eigen::VectorXd YSpace::interpolate(const FunctionType & q) const
{
  Eigen::VectorXd qh = Eigen::VectorXd::Zero(dimension());

  interpolate(q, qh);

  return qh;
}

//------------------------------------------------------------------------------

void YSpace::interpolate(const FunctionType & q, Eigen::VectorXd & qh) const
{
  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &qh, q](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *mesh().face(iF);

          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 10);
          qh.segment(globalOffset(F), PolynomialSpaceDimension<Face>::Poly(degree()))
            = l2_projection(q, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute(faceBasis(iF), quad_2k_F));
        } // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &qh, q](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);

          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 10);
          qh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::Poly(degree()-1))
            = l2_projection(q, cellBasis(iT), quad_2k_T, evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_T));
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
}

//------------------------------------------------------------------------------

Eigen::VectorXd YSpace::interpolateFaces(const FunctionType & q) const
{
  Eigen::VectorXd qh = Eigen::VectorXd::Zero(mesh().n_faces() * PolynomialSpaceDimension<Face>::Poly(degree()));

  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &qh, q](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *mesh().face(iF);

          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 10);
          qh.segment(iF * PolynomialSpaceDimension<Face>::Poly(degree()), PolynomialSpaceDimension<Face>::Poly(degree()))
            = l2_projection(q, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute(faceBasis(iF), quad_2k_F));
        } // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  return qh;
}

Eigen::VectorXd YSpace::interpolate_grad (const GradientType & v) const {
    Eigen::VectorXd vh = Eigen::VectorXd::Zero(mesh().n_cells()* gradientReconstructionBasis(0).dimension());
    interpolate_grad(v, vh);
    return vh;
}

void YSpace::interpolate_grad (const GradientType & v, Eigen::VectorXd &vh) const {

    std::function<void(size_t, size_t)> interpolate_cells
            = [this, &vh, v](size_t start, size_t end) -> void {
                for (size_t iT = start; iT < end; iT++) {
                    const Cell &T = *mesh().cell(iT);

                    QuadratureRule quad_2k_T = generate_quadrature_rule(T, 10);
                    vh.segment(iT * gradientReconstructionBasis(iT).dimension(),
                               gradientReconstructionBasis(iT).dimension())
                            = l2_projection <GradientReconstructionBasisType>
                                            (v,
                                            gradientReconstructionBasis(iT),
                                            quad_2k_T,
                                            evaluate_quad<Function>::compute(gradientReconstructionBasis(iT),quad_2k_T)
                                            );
                } // for iT
            };
    parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
}

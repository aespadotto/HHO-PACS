#include "xspace-sum-evol.hpp"

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
  : HHOSpace(mesh, 2 * PolynomialSpaceDimension<Face>::Poly(K-1) + (K+2),
                   3 * PolynomialSpaceDimension<Cell>::Poly(K)),
    m_use_threads(use_threads),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_face_bases(mesh.n_faces()),
    m_curl_reconstruction_bases(mesh.n_cells()),
    m_cell_curls(mesh.n_cells()),
    m_cell_curls_rhs(mesh.n_cells()),
    m_cell_to_face_proj(mesh.n_cells())
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

          typedef CurlBasis<TensorizedVectorFamily<MonomialScalarBasisCell, 3>> temp_basis_type;
            CurlBasis<TensorizedVectorFamily<MonomialScalarBasisCell, 3>> basis_curl =
                    CurlBasis(TensorizedVectorFamily<MonomialScalarBasisCell,3>(MonomialScalarBasisCell(T, m_K)));
            //take out null or repeated elements
            Family<temp_basis_type> filtered = Filter<temp_basis_type> (basis_curl, quad_2k_T);
          auto basis_curl_quad = evaluate_quad<Function>::compute (filtered, quad_2k_T);
          this->m_curl_reconstruction_bases[iT].reset( new CurlReconstructionBasisType (
                  //l2_orthonormalize(filtered, quad_2k_T, basis_curl_quad)
                  Filter<temp_basis_type> (basis_curl, quad_2k_T)
                  //basis_curl
                  ));
         }

      };

  m_output << "[XSpace] Constructing element bases" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_cell_bases, m_use_threads);

  // Construct face bases
  std::function<void(size_t, size_t)> construct_all_face_bases
    = [this](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          const Face & F = *this->mesh().face(iF);

          MonomialScalarBasisFace basis1_Pk_F(F, m_K-1); //ancestors for first part of face basis
          MonomialScalarBasisFace basis2_Pk_F(F, m_K+1); //ancestors for second part of face basis
          auto basis1 = TangentFamily<MonomialScalarBasisFace> (basis1_Pk_F, basis1_Pk_F.jacobian());
          auto basis2 = ShiftedBasis(GradientBasis(basis2_Pk_F),
                                     PolynomialSpaceDimension<Face>::Poly(m_K+1)-(m_K+2));
          BasisDirectSum <TangentFamily<MonomialScalarBasisFace>,
                            ShiftedBasis<GradientBasis<MonomialScalarBasisFace>>>
                            basis_face = BasisDirectSum(basis1, basis2);
          //not l2_orthonormalized (yes)
          QuadratureRule quad_face = generate_quadrature_rule (F, 2*m_K);
          auto basis_face_quad = evaluate_quad<Function>::compute(basis_face, quad_face);
          this->m_face_bases[iF].reset( new FaceBasisType(l2_orthonormalize(basis_face, quad_face, basis_face_quad)));

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
          QuadratureRule quad_2kmo_T = generate_quadrature_rule(T, 2 * (m_K));
          Eigen::MatrixXd MCT = compute_gram_matrix(
                                                    evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2kmo_T),
                                                    quad_2kmo_T
                                                    );
          //if (iT == 0) {std::cout<<"MCT"<<std::endl<<MCT<<std::endl;}
          //------------------------------------------------------------------------------
          // Right-hand side matrix
          size_t no_rows = curlReconstructionBasis(iT).dimension();
          Eigen::MatrixXd BCT = Eigen::MatrixXd::Zero(no_rows, dimensionCell(iT));
          //std::cout<<"Rapid check "<<BCT.rows()<<"*"<<BCT.cols()<<std::endl;

          // Boundary contribution
          for (size_t iF = 0; iF < T.n_faces(); iF++) {
            const Face & F = *T.face(iF);

            Eigen::Vector3d nF = F.normal();
            QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * m_K);
            BCT.block(0, iF * numLocalDofsFace(), no_rows, numLocalDofsFace())
              += T.face_orientation(iF) * compute_gram_matrix(
                                                              vector_product(evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2k_F), nF),
                                                              evaluate_quad<Function>::compute<FaceBasisType>(faceBasis(F.global_index()), quad_2k_F),
                                                              quad_2k_F
                                                              );
            BCT.topRightCorner(no_rows, numLocalDofsCell())
              -= T.face_orientation(iF)*compute_gram_matrix (vector_product(evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2k_F), nF),
                                                             evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_F),
                                                             quad_2k_F);
           
          } // for iF

          // Element contribution
          BCT.topRightCorner(no_rows, numLocalDofsCell())
            += compute_gram_matrix(
                                   evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2kmo_T),
                                   evaluate_quad<Curl>::compute(cellBasis(iT), quad_2kmo_T),
                                   quad_2kmo_T
                                 );


          m_cell_curls_rhs[iT] = BCT;
          m_cell_curls[iT] = MCT.ldlt().solve(BCT);
        } // for iT
      };

  m_output << "[XSpace] Computing curl reconstructions" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_curls, m_use_threads);
  //this->show_info();

    // Construct cell to face projections
    std::function<void(size_t, size_t)> construct_all_projections
            = [this](size_t start, size_t end)->void
            {

                for (size_t iT = start; iT < end; iT++) {
                    const Cell & T = *this->mesh().cell(iT);
                    std::vector <Eigen::MatrixXd> face_matrices(T.n_faces());
                    m_cell_to_face_proj[iT].resize(T.n_faces());
                    for (size_t iF = 0; iF < T.n_faces(); iF++) {
                        const Face & F = *T.face(iF);

                        QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2*degree());
                        auto basis_face_quad = evaluate_quad<Function>::compute(faceBasis(F.global_index()),quad_2k_F);
                        auto basis_cell_quad = evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_F);
                        Eigen::MatrixXd MFF = compute_gram_matrix(
                                basis_face_quad,
                                quad_2k_F
                                );
                        Eigen::MatrixXd MFT = compute_gram_matrix(
                                basis_face_quad,
                                basis_cell_quad,
                                quad_2k_F
                                );
                    Eigen::MatrixXd mat =     MFF.ldlt().solve(MFT);
                    m_cell_to_face_proj[iT][iF]=mat;
                    } // for iF
                } // for iT
            };
    m_output << "[XSpace] Computing cell to face projections" << std::endl;
    parallel_for(mesh.n_cells(), construct_all_projections, m_use_threads);


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

          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 10 );
          vh.segment(globalOffset(F), faceBasis(iF).dimension())
            = l2_projection(nF_cross_v_cross_nF, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute<FaceBasisType>(faceBasis(iF), quad_2k_F));
        } // for iF
      };
   parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &vh, v](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);

          QuadratureRule quad_2k_T = generate_quadrature_rule(T, 10 );
          vh.segment(globalOffset(T), cellBasis(0).dimension())
            = l2_projection(v, cellBasis(iT), quad_2k_T, evaluate_quad<Function>::compute(cellBasis(iT), quad_2k_T));
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
}

//------------------------------------------------------------------------------

Eigen::VectorXd XSpace::interpolateFaces(const FunctionType & v) const
{
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(mesh().n_faces() * faceBasis(0).dimension());

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

          QuadratureRule quad_2k_F = generate_quadrature_rule(F, 10 );
          vh.segment(iF * faceBasis(0).dimension(), faceBasis(0).dimension())
            = l2_projection(nF_cross_v_cross_nF, faceBasis(iF), quad_2k_F, evaluate_quad<Function>::compute(faceBasis(iF), quad_2k_F));
        } // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

  return vh;
}


Eigen::VectorXd XSpace::interpolate_curl (const FunctionType & v) const {
    Eigen::VectorXd vh = Eigen::VectorXd::Zero(mesh().n_cells()* curlReconstructionBasis(0).dimension());
    interpolate_curl(v, vh);
    return vh;
}

void XSpace::interpolate_curl (const FunctionType & v, Eigen::VectorXd &vh) const{

    std::function<void(size_t, size_t)> interpolate_cells
            = [this, &vh, v](size_t start, size_t end)->void
            {
                for (size_t iT = start; iT < end; iT++) {
                    const Cell & T = *mesh().cell(iT);

                    QuadratureRule quad_2k_T = generate_quadrature_rule(T, 10 );
                    vh.segment(iT* curlReconstructionBasis(iT).dimension(), curlReconstructionBasis(iT).dimension())
                            = l2_projection(v,
                                            curlReconstructionBasis(iT), quad_2k_T,
                                            evaluate_quad<Function>::compute(curlReconstructionBasis(iT), quad_2k_T));
                } // for iT
            };
    parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
}

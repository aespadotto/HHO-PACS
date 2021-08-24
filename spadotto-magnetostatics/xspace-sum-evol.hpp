// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr) (original version in Schemes/HHO-magnetostatics)
// Partially adapted by Aurelio Spadotto 

#ifndef XSPACE_SUM_HPP
#define XSPACE_SUM_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>

#include <hhospace.hpp>
#include "add-basis.hpp"

namespace HArDCore3D
{
  class XSpace : public HHOSpace {
  public:
    typedef TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> CellBasisType;
    typedef Family<BasisDirectSum <TangentFamily<MonomialScalarBasisFace>,
                            ShiftedBasis<GradientBasis<MonomialScalarBasisFace>>>>
                            FaceBasisType;

    typedef Family<CurlBasis<TensorizedVectorFamily<MonomialScalarBasisCell, 3>>> CurlReconstructionBasisType;

    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> FunctionType;

    /// Constructor
    XSpace(
	   const Mesh & mesh,
	   size_t K,
	   bool use_threads = true,
	   std::ostream & output = std::cout
	   );

    /// Returns the mesh
    inline const Mesh & mesh() const
    {
      return m_mesh;
    }

     /// Returns the polynomial degree
    size_t degree() const
    {
      return m_K;
    }

    /// Returns cell basis for element iT
    inline const CellBasisType & cellBasis(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_cell_bases[iT] );
      return *m_cell_bases[iT].get();
    }

    /// Returns vector cell basis for element iT
    inline const CurlReconstructionBasisType & curlReconstructionBasis(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_curl_reconstruction_bases[iT] );
      return *m_curl_reconstruction_bases[iT].get();
    }

    /// Returns face bases for face iF
    inline const FaceBasisType & faceBasis(size_t iF) const
    {
      // Make sure that the basis has been created
      assert( m_face_bases[iF] );
      return *m_face_bases[iF].get();
    }

    /// Returns the curl reconstruction
    inline const Eigen::MatrixXd & curlReconstruction(size_t iT) const
    {
      return m_cell_curls[iT];
    }

    /// Returns the right-hand side of the curl reconstruction problem
    inline const Eigen::MatrixXd & curlReconstructionRHS(size_t iT) const
    {
      return m_cell_curls_rhs[iT];
    }

    /// Returns projection matrix from cell space to face space
    inline const Eigen::MatrixXd & cell_to_face_projection(size_t iT, size_t iF) const
    {
        return m_cell_to_face_proj[iT][iF];
    }


    /// Full interpolator
    Eigen::VectorXd interpolate(const FunctionType & v) const;

    /// Full interpolator using a provided vector
    void interpolate(const FunctionType & v, Eigen::VectorXd & vh) const;

    /// Interpolate on curl reconstruction basis (useful to check commutation property)
    Eigen::VectorXd interpolate_curl (const FunctionType & v) const;
    /// using provided vector
    void interpolate_curl (const FunctionType & v, Eigen::VectorXd &vh) const;

    /// Interpolator of face unknowns
    Eigen::VectorXd interpolateFaces(const FunctionType & v) const;

    /// Print Some Info
     inline void show_info (void) const {
        std::cout<<"Dimension of local Cell Space "<<cellBasis(0).dimension()<<" == "<<3*PolynomialSpaceDimension<Cell>::Poly(m_K)<<std::endl;
        std::cout<<"Dimension of local Curl Rec Space "<<curlReconstructionBasis(0).dimension()<<" == "<<PolynomialSpaceDimension<Cell>::GolyCompl(m_K)<<std::endl;
        std::cout<<"Dimension of local Face Space "<<faceBasis(0).dimension()<<" == "
                 <<2*PolynomialSpaceDimension<Face>::Poly(m_K-1)+(m_K+2)<<std::endl;
        std::cout<<"Curl Reconstruction System "<<m_cell_curls[0].rows()<<"*"<<m_cell_curls[0].cols()<<std::endl;
        std::cout<<"Curl Reconstruction Rhs "<<m_cell_curls_rhs[0].rows()<<std::endl;
        std::cout<<"Check that Basis for Curl Rec is independent: "<<std::endl;

        QuadratureRule quad_2kmo_T = generate_quadrature_rule(*this->mesh().cell(0), 2*m_K);
        auto check_matrix_curl = compute_gram_matrix
                            (evaluate_quad<Function>::compute(curlReconstructionBasis(0),quad_2kmo_T),
                             quad_2kmo_T);
        std::cout<<check_matrix_curl<<std::endl;
        //std::cout<<"*********"<<std::endl;

        std::cout<<"evaluation of the basis of curls:"<<std::endl;
        std::cout<<"Check that Cell Basis is independent: "<<std::endl;
        auto check_matrix_cell = compute_gram_matrix(evaluate_quad<Function>::compute(cellBasis(0),quad_2kmo_T),quad_2kmo_T);
        std::cout<<check_matrix_cell.trace()<<std::endl;
        std::cout<<"Check that Face Basis is independent: "<<std::endl;
        QuadratureRule quad_2kmo_F = generate_quadrature_rule(*this->mesh().face(0),2*m_K);
        auto check_matrix_face = compute_gram_matrix(evaluate_quad<Function>::compute(faceBasis(0),quad_2kmo_F),quad_2kmo_F);
        std::cout<<check_matrix_face.trace()<<std::endl;
        std::cout<<"Check a sample of reconstruction matrix"<<std::endl;
        std::cout<<m_cell_curls[0]<<std::endl<<std::endl;
        std::cout<<m_cell_curls_rhs[0].topRightCorner(curlReconstructionBasis(0).dimension(),5)<<"..."<<std::endl;
        return;
    }

  private:
    // Use parallelism
    bool m_use_threads;
    // Degree
    const size_t m_K;
    // Output stream
    std::ostream & m_output;

    // Cell bases
    std::vector<std::unique_ptr<CellBasisType> > m_cell_bases;
    // Face bases
    std::vector<std::unique_ptr<FaceBasisType> > m_face_bases;

    // Curl reconstruction basis
    std::vector<std::unique_ptr<CurlReconstructionBasisType> > m_curl_reconstruction_bases;

    // Container for local gradients
    std::vector<Eigen::MatrixXd> m_cell_curls;
    std::vector<Eigen::MatrixXd> m_cell_curls_rhs;
    std::vector<std::vector<Eigen::MatrixXd>>  m_cell_to_face_proj;
  }; // class XSpace
} // namespace HArDCore3D

#endif // XSPACE_HPP

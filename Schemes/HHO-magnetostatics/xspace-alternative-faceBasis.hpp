#ifndef XSPACE_HPP
#define XSPACE_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>

#include "hhospace.hpp"

//implementation of X space for magnetostatics problem in vector potential formulation using
//as face unknowns polynomials from the direct sum:  P^k-1(F;R^2) + grad_tau (P^k+1(F))
//face dofs are split into two distinct basis generating the two subspaces

namespace HArDCore3D
{
  class XSpace_alternative_faceBasis : public HHOSpace {
  public:
    typedef TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> CellBasisType;
    ////////////////////
    typedef TangentFamily<Family<MonomialScalarBasisFace> > FaceBasisType1;          //P^k-1(F;R^2)
    typedef GradientBasis<ShiftedBasis<MonomialScalarBasisFace> > FaceBasisType2;    //grad (P_hom^k+1(F))
    ////////////////////
    typedef TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> CurlReconstructionBasisType; //--->next to modify

    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> FunctionType;

    /// Constructor
    XSpace_alternative_faceBasis(
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

    /// Returns face bases 1 for face iF
    inline const FaceBasisType1 & faceBasis1(size_t iF) const
    {
      // Make sure that the basis has been created
      assert( m_face_bases1[iF] );
      return *m_face_bases1[iF].get();
    }

    /// Returns face bases 2 for face iF
    inline const FaceBasisType2 & faceBasis2(size_t iF) const
    {
      // Make sure that the basis has been created
      assert( m_face_bases2[iF] );
      return *m_face_bases2[iF].get();
    }

    //return dimension of face basis 1
    inline const size_t face_basis1_dim () const
    {
      return m_face_basis1_dim;
    }

    //return dimension of face basis 2
    inline const size_t face_basis2_dim () const
    {
      return m_face_bases2_dim;
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

    /// Full interpolator
    Eigen::VectorXd interpolate(const FunctionType & v) const;

    /// Full interpolator using a provided vector
    void interpolate(const FunctionType & v, Eigen::VectorXd & vh) const;

    /// Interpolator of face unknowns
    Eigen::VectorXd interpolateFaces(const FunctionType & v) const;

  private:
    // Use parallelism
    bool m_use_threads;
    // Degree
    const size_t m_K;
    // Output stream
    std::ostream & m_output;

    // Cell bases
    std::vector<std::unique_ptr<CellBasisType> > m_cell_bases;
    // Face bases for P^k(Fi; R^2)
    std::vector<std::unique_ptr<FaceBasisType1> > m_face_bases1;
    // Face bases for grad_tau(P_hom^k+1(Fi))
    std::vector<std::unique_ptr<FaceBasisType2> > m_face_bases2;

    // Curl reconstruction basis
    std::vector<std::unique_ptr<CurlReconstructionBasisType> > m_curl_reconstruction_bases;

    // Container for local gradients
    std::vector<Eigen::MatrixXd> m_cell_curls;
    std::vector<Eigen::MatrixXd> m_cell_curls_rhs;

    size_t m_face_basis1_dim;
    size_t m_face_bases2_dim;
  }; // class XSpace
} // namespace HArDCore3D

#endif // XSPACE_HPP

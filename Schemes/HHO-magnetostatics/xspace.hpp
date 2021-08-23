#ifndef XSPACE_HPP
#define XSPACE_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>

#include "hhospace.hpp"

namespace HArDCore3D
{
  class XSpace : public HHOSpace {
  public:
    typedef TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> CellBasisType;
    typedef TangentFamily<Family<MonomialScalarBasisFace> > FaceBasisType;

    typedef TensorizedVectorFamily<Family<MonomialScalarBasisCell>, 3> CurlReconstructionBasisType;
    
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
    // Face bases
    std::vector<std::unique_ptr<FaceBasisType> > m_face_bases;

    // Curl reconstruction basis
    std::vector<std::unique_ptr<CurlReconstructionBasisType> > m_curl_reconstruction_bases;

    // Container for local gradients
    std::vector<Eigen::MatrixXd> m_cell_curls;
    std::vector<Eigen::MatrixXd> m_cell_curls_rhs;
  }; // class XSpace
} // namespace HArDCore3D

#endif // XSPACE_HPP

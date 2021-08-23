#ifndef YSPACE_HPP
#define YSPACE_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>

#include "hhospace.hpp"

namespace HArDCore3D
{
  class YSpace : public HHOSpace {
  public:
    typedef Family<MonomialScalarBasisCell> CellBasisType;
    typedef Family<MonomialScalarBasisFace> FaceBasisType;

    typedef TensorizedVectorFamily<CellBasisType, 3> GradientReconstructionBasisType;

    typedef std::function<double(const Eigen::Vector3d &)> FunctionType;
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> GradientType;
    
    /// Constructor    
    YSpace(const Mesh & mesh,
	   size_t K,
	   bool use_threads = true,
	   std::ostream & output = std::cout);

    /// Returns the mesh
    inline const Mesh & mesh() const
    {
      return m_mesh;
    }

    /// Return the polynomial degree
    size_t degree() const
    {
      return m_K;
    }

    /// Return cell basis for element iT
    inline const CellBasisType & cellBasis(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_cell_bases[iT] );
      return *m_cell_bases[iT].get();
    }

    /// Return vector cell basis for element iT
    inline const GradientReconstructionBasisType & gradientReconstructionBasis(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_gradient_reconstruction_bases[iT] );
      return *m_gradient_reconstruction_bases[iT].get();
    }

    /// Return face bases for face iF
    inline const FaceBasisType & faceBasis(size_t iF) const
    {
      // Make sure that the basis has been created
      assert( m_face_bases[iF] );
      return *m_face_bases[iF].get();
    }
    
    /// Returns the gradient reconstruction
    inline const Eigen::MatrixXd & gradientReconstruction(size_t iT) const
    {
      return m_cell_gradients[iT];
    }

    /// Returns the right-hand side of the gradient reconstruction problem
    inline const Eigen::MatrixXd & gradientReconstructionRHS(size_t iT) const
    {
      return m_cell_gradients_rhs[iT];
    }

    /// Full interpolator
    Eigen::VectorXd interpolate(const FunctionType & q) const;

    /// Full interpolator using a provided vector
    void interpolate(const FunctionType & q, Eigen::VectorXd & qh) const;

    /// Interpolate on curl reconstruction basis (useful to check commutation property)
    Eigen::VectorXd interpolate_grad (const GradientType & v) const;
    /// using provided vector
    void interpolate_grad (const GradientType & v, Eigen::VectorXd &vh) const;


    /// Interpolator of face unknowns
    Eigen::VectorXd interpolateFaces(const FunctionType  & q) const;
    
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

    // Grandient reconstruction basis
    std::vector<std::unique_ptr<GradientReconstructionBasisType> > m_gradient_reconstruction_bases;

    // Container for local gradients
    std::vector<Eigen::MatrixXd> m_cell_gradients;
    std::vector<Eigen::MatrixXd> m_cell_gradients_rhs;
  }; // class YSpace
} // namespace HArDCore3D

#endif // YSPACE_HPP

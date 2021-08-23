#ifndef CARTESIANPRODUCTHHOSPACE_HPP
#define CARTESIANPRODUCTHHOSPACE_HPP

#include <array>

#include "hhospace.hpp"

namespace HArDCore3D
{
  /// Class for Cartesian product HHO spaces
  template<size_t N>
  class CartesianProductHHOSpace : public DOFSpace {
  public:
    typedef std::array<const HHOSpace *, N> CartesianFactorsArray;
    typedef Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> DofIndexVector;
    
    /// Constructor
    CartesianProductHHOSpace(const CartesianFactorsArray & cartesian_factors);

    /// Returns the i-th Cartesian factor
    inline const std::unique_ptr<HHOSpace> cartesianFactor(size_t i) {
      return m_cartesian_factors[i];
    }

    inline size_t numLocalDofsCellBoundary(const Cell & T) const {
      return T.n_faces() * numLocalDofsFace();
    }
    
    /// Returns the local offset of the face F with respect to the cell T for the i-th Cartesian factor
    size_t localOffset(const Cell & T, const Face & F, size_t i) const;

    /// Returns the local offset of the uknowns attached to the cell T for the i-th Cartesian factor
    size_t localOffset(const Cell & T, size_t) const;

    /// Returns the global offset for the unknowns on the face F for the i-th Cartesian factor
    size_t globalOffset(const Face & F, size_t i) const;
    
    /// Returns the global offset for the unknowns on the cell T for the i-th Cartesian factor
    size_t globalOffset(const Cell & T, size_t i) const;

    /// Returns the global offset for the unknowns on the face F
    inline size_t globalOffset(const Face & F) const {
      return F.global_index() * numLocalDofsFace();
    }

    /// Returns the global offset for the unknowns on the cell T
    inline size_t globalOffset(const Cell & T) const {
      return mesh().n_faces() * numLocalDofsFace() + T.global_index() * numLocalDofsCell();
    }

    /// Merge DOF vectors
    Eigen::VectorXd merge(const std::array<Eigen::VectorXd, N> & vhs) const;

    //------------------------------------------------------------------------------
    // Global DOF indices for an element T
    //------------------------------------------------------------------------------

    DofIndexVector globalCellDofIndices(const Cell & T) const;

    DofIndexVector globalCellBoundaryDofIndices(const Cell & T) const;

  private:
    /// Count the total number of face DOFs
    static size_t _total_face_dofs(const CartesianFactorsArray & cartesian_factors);
    
    /// Count the total number of cell DOFs
    static size_t _total_cell_dofs(const CartesianFactorsArray & cartesian_factors);
 
    const CartesianFactorsArray m_cartesian_factors;
    const Mesh & m_mesh;
    std::array<size_t, N> m_component_offset_face;
    std::array<size_t, N> m_component_offset_cell;
  };

  //------------------------------------------------------------------------------
  // Implementation
  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::_total_face_dofs(const CartesianFactorsArray & cartesian_factors)
  {
    size_t total_face_dofs = 0;

    for (size_t i = 0; i < N; i++) {
      total_face_dofs += cartesian_factors[i]->numLocalDofsFace();
    }
    
    return total_face_dofs;
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::_total_cell_dofs(const CartesianFactorsArray & cartesian_factors)
  {
    size_t total_cell_dofs = 0;
    
    for (size_t i = 0; i < N; i++) {
      total_cell_dofs += cartesian_factors[i]->numLocalDofsCell();
    }
    
    return total_cell_dofs;
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  CartesianProductHHOSpace<N>::CartesianProductHHOSpace(const CartesianFactorsArray & cartesian_factors)
    : DOFSpace(cartesian_factors[0]->mesh(), 0, 0, _total_face_dofs(cartesian_factors), _total_cell_dofs(cartesian_factors)),
      m_cartesian_factors(cartesian_factors),
      m_mesh(m_cartesian_factors[0]->mesh())
  {
    // Make sure that the underlying mesh is the same for all the spaces
    for (size_t i = 0; i < N; i++) {
      assert(&m_cartesian_factors[i]->mesh() == &m_mesh);
    } // for i

    // Compute component offsets
    m_component_offset_face.fill(0);
    m_component_offset_cell.fill(0);
    for (size_t i = 0; i < N; i++) { 
      for (size_t j = 0; j < i; j++) {
        m_component_offset_face[i] += m_cartesian_factors[j]->numLocalDofsFace();
        m_component_offset_cell[i] += m_cartesian_factors[j]->numLocalDofsCell();
      } // for j
    } // for i
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::localOffset(const Cell & T, const Face & F, size_t i) const
  {
    return T.index_face(&F) * numLocalDofsFace() + m_component_offset_face[i];
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::localOffset(const Cell & T, size_t i) const
  {
    return T.n_faces() * numLocalDofsFace() + m_component_offset_cell[i];
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::globalOffset(const Face & F, size_t i) const
  {
    return F.global_index() * numLocalDofsFace() + m_component_offset_face[i];
  }
   
  //------------------------------------------------------------------------------

  template<size_t N>
  size_t CartesianProductHHOSpace<N>::globalOffset(const Cell & T, size_t i) const
  {
    return m_mesh.n_faces() * numLocalDofsFace()
      + T.global_index() * numLocalDofsCell()
      + m_component_offset_cell[i];
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  Eigen::VectorXd CartesianProductHHOSpace<N>::merge(const std::array<Eigen::VectorXd, N> & vhs) const
  {
    Eigen::VectorXd vh = Eigen::VectorXd::Zero(dimension());
    
    for (size_t iF = 0; iF < m_mesh.n_faces(); iF++) {
      const Face & F = *m_mesh.face(iF);

      for (size_t i = 0; i < N; i++) {
        vh.segment(globalOffset(F, i), m_cartesian_factors[i]->numLocalDofsFace())
          = vhs[i].segment(m_cartesian_factors[i]->globalOffset(F), m_cartesian_factors[i]->numLocalDofsFace());
      } // for i
    } // for iF

    for (size_t iT = 0; iT < m_mesh.n_cells(); iT++) {
      const Cell & T = *m_mesh.cell(iT);
      
      for (size_t i = 0; i < N; i++) {
        vh.segment(globalOffset(T, i), m_cartesian_factors[i]->numLocalDofsCell())
          = vhs[i].segment(m_cartesian_factors[i]->globalOffset(T), m_cartesian_factors[i]->numLocalDofsCell());
      } // for i
    } // for iT

    return vh;
  } 

  //------------------------------------------------------------------------------

  template<size_t N>
  typename CartesianProductHHOSpace<N>::DofIndexVector
  CartesianProductHHOSpace<N>::globalCellDofIndices(const Cell & T) const
  {
    DofIndexVector I(numLocalDofsCell());

    size_t offset_T = globalOffset(T);
    for (size_t i = 0; i < numLocalDofsCell(); i++) {
      I(i) = offset_T + i;
    } // for i

    return I;
  }

  //------------------------------------------------------------------------------

  template<size_t N>
  typename CartesianProductHHOSpace<N>::DofIndexVector
  CartesianProductHHOSpace<N>::globalCellBoundaryDofIndices(const Cell & T) const
  {
    DofIndexVector I(T.n_faces() * numLocalDofsFace());

    size_t dof_index = 0;
  
    for (size_t iF = 0; iF < T.n_faces(); iF++) {
      size_t offset_F = globalOffset(*T.face(iF));
      for (size_t i = 0; i < numLocalDofsFace(); i++, dof_index++) {
        I(dof_index) = offset_F + i;
      } // for i
    } // for iF

    return I;
  }
  
} // namespace HArDCore3D

#endif

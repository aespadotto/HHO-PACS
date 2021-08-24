// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr) 
#ifndef HHOSPACE_HPP
#define HHOSPACE_HPP
#include <dofspace.hpp>

namespace HArDCore3D
{
  /// Base class for HHO spaces
  class HHOSpace : public DOFSpace {
  public:
    typedef Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> DofIndexVector;
    
    /// Constructor
    HHOSpace(
             const Mesh & mesh,
             size_t n_local_face_dofs,
             size_t n_local_cell_dofs
             );

    //------------------------------------------------------------------------------
    // Global offsets
    //------------------------------------------------------------------------------

    /// Returns the global offset for the unknowns on the face F
    size_t globalOffset(const Face & F) const {
      return F.global_index() * numLocalDofsFace();
    }

    /// Returns the global offset for the unknowns on the cell T
    size_t globalOffset(const Cell & T) const {
      return mesh().n_faces() * numLocalDofsFace() + T.global_index() * numLocalDofsCell();
    }

    //------------------------------------------------------------------------------
    // Restrictions
    //------------------------------------------------------------------------------

    /// Restrict to the face of index iF
    Eigen::VectorXd restrictFace(size_t iF, const Eigen::VectorXd & vh) const
    {
      return vh.segment(globalOffset(*mesh().face(iF)), m_n_local_face_dofs);
    }

    /// Restrict to the cell of index iT
    Eigen::VectorXd restrictCell(size_t iT, const Eigen::VectorXd & vh) const;
    
    /// Restrict to a face
    inline Eigen::VectorXd restrict(const Face & F, const Eigen::VectorXd vh) const
    {
      return restrictFace(F.global_index(), vh);
    }

    /// Restrict to a cell
    inline Eigen::VectorXd restrict(const Cell & T, const Eigen::VectorXd vh) const
    {
      return restrictCell(T.global_index(), vh);
    }

    //------------------------------------------------------------------------------
    // Global DOF indices for an element T
    //------------------------------------------------------------------------------

    DofIndexVector globalCellDofIndices(const Cell & T) const;

    DofIndexVector globalCellBoundaryDofIndices(const Cell & T) const;   
  };
} // namespace HArDCore3D


#endif

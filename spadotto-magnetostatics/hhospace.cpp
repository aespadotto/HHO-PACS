#include "hhospace.hpp"

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

HHOSpace::HHOSpace(
       const Mesh & mesh,
       size_t n_local_face_dofs,
       size_t n_local_cell_dofs
       )
  : DOFSpace(mesh, 0, 0, n_local_face_dofs, n_local_cell_dofs)
{
  // Do nothing
}

//------------------------------------------------------------------------------

HHOSpace::DofIndexVector HHOSpace::globalCellDofIndices(const Cell & T) const
{
  DofIndexVector I(numLocalDofsCell());

  size_t offset_T = globalOffset(T);
  for (size_t i = 0; i < m_n_local_cell_dofs; i++) {
    I(i) = offset_T + i;
  } // for i

  return I;
}

//------------------------------------------------------------------------------

HHOSpace::DofIndexVector HHOSpace::globalCellBoundaryDofIndices(const Cell & T) const
{
  DofIndexVector I(T.n_faces() * numLocalDofsFace());

  size_t dof_index = 0;
  
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    size_t offset_F = globalOffset(*T.face(iF));
    for (size_t i = 0; i < m_n_local_face_dofs; i++, dof_index++) {
      I(dof_index) = offset_F + i;
    } // for i
  } // for iF

  return I;
}

//------------------------------------------------------------------------------
// Restrictions
//------------------------------------------------------------------------------

/// Restrict to the cell of index iT
Eigen::VectorXd HHOSpace::restrictCell(size_t iT, const Eigen::VectorXd & vh) const
{
  const Cell & T = *mesh().cell(iT);
  
  Eigen::VectorXd vT(dimensionCell(iT));

  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    vT.segment(localOffset(T, F), numLocalDofsFace()) = vh.segment(globalOffset(F), numLocalDofsFace());
  } // for iF
  vT.segment(localOffset(T), numLocalDofsCell()) = vh.segment(globalOffset(T), m_n_local_cell_dofs);
  
  return vT;
}

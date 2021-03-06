#include <cassert>

#include <ddrcore.hpp>
#include <parallel_for.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------

DDRCore::DDRCore(const Mesh & mesh, size_t K, bool use_threads, std::ostream & output)
  : m_mesh(mesh),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_face_bases(mesh.n_faces()),
    m_edge_bases(mesh.n_edges())
{
  m_output << "[DDRCore] Initializing" << std::endl;
  
  // Construct element bases
  std::function<void(size_t, size_t)> construct_all_cell_bases
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iT = start; iT < end; iT++) {
	        this->m_cell_bases[iT].reset( new CellBases(this->_construct_cell_bases(iT)) );
	      } // for iT
      };

  m_output << "[DDRCore] Constructing element bases" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_cell_bases, use_threads);
  
  // Construct face bases
  std::function<void(size_t, size_t)> construct_all_face_bases
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iF = start; iF < end; iF++) {
	        this->m_face_bases[iF].reset( new FaceBases(_construct_face_bases(iF)) );
	      } // for iF
      };
  
  m_output << "[DDRCore] Constructing face bases" << std::endl;
  parallel_for(mesh.n_faces(), construct_all_face_bases, use_threads);

  // Construct edge bases
  std::function<void(size_t, size_t)> construct_all_edge_bases   
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        this->m_edge_bases[iE].reset( new EdgeBases(_construct_edge_bases(iE)) );
	      } // for iF
      };
  
  m_output << "[DDRCore] Constructing edge bases" << std::endl;
  parallel_for(mesh.n_edges(), construct_all_edge_bases, use_threads);
}

//------------------------------------------------------------------------------

DDRCore::CellBases DDRCore::_construct_cell_bases(size_t iT)
{
  const Cell & T = *m_mesh.cell(iT);

  CellBases bases_T;
  
  //------------------------------------------------------------------------------
  // Basis for Pk+1(T)
  //------------------------------------------------------------------------------
  
  MonomialScalarBasisCell basis_Pkpo_T(T, m_K + 1);
  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (m_K + 1));
  boost::multi_array<double, 2> on_basis_Pkpo_T_quad = evaluate_quad<Function>::compute(basis_Pkpo_T, quad_2kpo_T);
  // Orthonormalize and store
  bases_T.Polykpo.reset( new PolyBasisCellType(l2_orthonormalize(basis_Pkpo_T, quad_2kpo_T, on_basis_Pkpo_T_quad)) );   
  // Check that we got the dimension right
  assert( bases_T.Polykpo->dimension() == PolynomialSpaceDimension<Cell>::Poly(m_K + 1) );

  //------------------------------------------------------------------------------
  // Basis for Pk(T), Pk-1(T) and Pk(T)^3
  //------------------------------------------------------------------------------

  // Given that the basis for Pk+1(T) is hierarchical, bases for Pk(T) and
  // Pk-1(T) can be obtained by restricting the former
  bases_T.Polyk.reset( new RestrictedBasis<PolyBasisCellType>(*bases_T.Polykpo, PolynomialSpaceDimension<Cell>::Poly(m_K)) );  
  bases_T.Polyk3.reset( new Poly3BasisCellType(*bases_T.Polyk) );
  if (PolynomialSpaceDimension<Cell>::Poly(m_K - 1) > 0) {
    bases_T.Polykmo.reset( new RestrictedBasis<PolyBasisCellType>(*bases_T.Polykpo, PolynomialSpaceDimension<Cell>::Poly(m_K - 1)) );
  }
  // Check dimension Pk(T)^3
  assert( bases_T.Polyk3->dimension() == 3 * PolynomialSpaceDimension<Cell>::Poly(m_K) );
  
  //------------------------------------------------------------------------------
  // Basis for Gk-1(T)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Cell>::Goly(m_K - 1) > 0) {
    GradientBasis<ShiftedBasis<MonomialScalarBasisCell> >
      basis_Gkmo_T(ShiftedBasis<MonomialScalarBasisCell>(MonomialScalarBasisCell(T, m_K), 1));
    QuadratureRule quad_2kmo_T = generate_quadrature_rule(T, 2 * (m_K -1));  
    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(basis_Gkmo_T, quad_2kmo_T);
    // Orthonormalize and store the basis
    bases_T.Golykmo.reset( new GolyBasisCellType(l2_orthonormalize(basis_Gkmo_T, quad_2kmo_T, basis_Gkmo_T_quad)) );
    // Check that we got the dimension right
    assert( bases_T.Golykmo->dimension() == PolynomialSpaceDimension<Cell>::Goly(m_K - 1) );
  } // if

  //------------------------------------------------------------------------------
  // Bases for Gck(T), Gck+1(T), and Rk-1(T)
  //------------------------------------------------------------------------------

  // Gck+1(T) (orthonormalised)
  GolyComplBasisCell basis_Gckpo_T(T, m_K+1);
  auto on_basis_Gckpo_T_quad = evaluate_quad<Function>::compute(basis_Gckpo_T, quad_2kpo_T);
  bases_T.GolyComplkpo.reset( new GolyComplBasisCellType(l2_orthonormalize(basis_Gckpo_T, quad_2kpo_T, on_basis_Gckpo_T_quad)) );
  // check dimension
  assert( bases_T.GolyComplkpo->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K + 1) );
 
  // A quadrature of order 2k could be also useful below, so we create it here
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * m_K);
  if (PolynomialSpaceDimension<Cell>::GolyCompl(m_K) > 0) {
    // Gck(T)
    GolyComplBasisCell basis_Gck_T(T, m_K);
    auto on_basis_Gck_T_quad = evaluate_quad<Function>::compute(basis_Gck_T, quad_2k_T); 
    bases_T.GolyComplk.reset( new GolyComplBasisCellType(l2_orthonormalize(basis_Gck_T, quad_2k_T, on_basis_Gck_T_quad)) );
    assert( bases_T.GolyComplk->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K) );

    // Basis for curl Gck. We do not want to restart from bases_T.GolyComplk because it is orthonormalised (so a 
    // Family of other bases); if we started from this one, after orthonormalisation, the basis of Rk-1(T) would be
    // a Family of a Family, for which any evaluation could be quite expensive. 
    CurlBasis<GolyComplBasisCell> basis_curl_Gck_T(basis_Gck_T);
    auto basis_curl_Gck_T_quad = evaluate_quad<Function>::compute(basis_curl_Gck_T, quad_2k_T);
    // Orthonormalize and store as basis of Rk-1(T), then check dimension
    bases_T.Rolykmo.reset( new RolyBasisCellType(l2_orthonormalize(basis_curl_Gck_T, quad_2k_T, basis_curl_Gck_T_quad)) );   
    assert( bases_T.Rolykmo->dimension() == PolynomialSpaceDimension<Cell>::Roly(m_K - 1));

  } // if

  //------------------------------------------------------------------------------
  // Basis for Rck(T) and Rck+2(T)
  //------------------------------------------------------------------------------
  // Rck+2(T) (orthonormalised)
  RolyComplBasisCell basis_Rckp2_T(T, m_K+2);
  QuadratureRule quad_2kp2_T = generate_quadrature_rule(T, 2 * (m_K+2));
  auto on_basis_Rckp2_T_quad = evaluate_quad<Function>::compute(basis_Rckp2_T, quad_2kp2_T);
  bases_T.RolyComplkp2.reset( new RolyComplBasisCellType(l2_orthonormalize(basis_Rckp2_T, quad_2kp2_T, on_basis_Rckp2_T_quad)) );
  assert ( bases_T.RolyComplkp2->dimension() == PolynomialSpaceDimension<Cell>::RolyCompl(m_K+2) );

  // Rck(T) (orthonormalised). Could probably also be obtained as a RestrictedBasis of the previous one, but would
  // need to check if the basis for Rck are indeed hierarchical
  if (PolynomialSpaceDimension<Cell>::RolyCompl(m_K) > 0) { 
    RolyComplBasisCell basis_Rck_T(T, m_K);
    auto on_basis_Rck_T_quad = evaluate_quad<Function>::compute(basis_Rck_T, quad_2k_T);
    bases_T.RolyComplk.reset( new RolyComplBasisCellType(l2_orthonormalize(basis_Rck_T, quad_2k_T, on_basis_Rck_T_quad)) );
    assert ( bases_T.RolyComplk->dimension() == PolynomialSpaceDimension<Cell>::RolyCompl(m_K) );
  } // if

  return bases_T;
}

//------------------------------------------------------------------------------

DDRCore::FaceBases DDRCore::_construct_face_bases(size_t iF)
{
  const Face & F = *m_mesh.face(iF);
  
  FaceBases bases_F;

  //------------------------------------------------------------------------------
  // Basis for Pk+1(F)
  //------------------------------------------------------------------------------
  
  MonomialScalarBasisFace basis_Pkpo_F(F, m_K + 1);
  QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * (m_K + 1));
  auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(basis_Pkpo_F, quad_2kpo_F);
  // Orthonormalize and store the basis
  bases_F.Polykpo.reset( new PolyBasisFaceType(l2_orthonormalize(basis_Pkpo_F, quad_2kpo_F, basis_Pkpo_F_quad)) );
  // Check that we got the dimension right
  assert( bases_F.Polykpo->dimension() == PolynomialSpaceDimension<Face>::Poly(m_K + 1) );

  //------------------------------------------------------------------------------
  // Basis for Pk(F), Pk-1(F) and Pk(F)^2
  //------------------------------------------------------------------------------

  // Given that the basis for Pk+1(F) is hierarchical, bases for Pk(F) and
  // Pk-1(F) can be obtained by restricting the former
  bases_F.Polyk.reset( new RestrictedBasis<PolyBasisFaceType>(*bases_F.Polykpo, PolynomialSpaceDimension<Face>::Poly(m_K)) );
  if (PolynomialSpaceDimension<Face>::Poly(m_K - 1) > 0) {
    bases_F.Polykmo.reset( new RestrictedBasis<PolyBasisFaceType>(*bases_F.Polykpo, PolynomialSpaceDimension<Face>::Poly(m_K - 1)) );
  }
  // Basis of Pk(F)^2 as TangentFamily. The two "ancestor" come back to the original MonomialScalarBasisFace, to get its jacobian  
  bases_F.Polyk2.reset( new Poly2BasisFaceType(*bases_F.Polyk,bases_F.Polyk->ancestor().ancestor().jacobian()) );
  // Check dimension
  assert( bases_F.Polyk2->dimension() == 2 * PolynomialSpaceDimension<Face>::Poly(m_K) );
  
  //------------------------------------------------------------------------------
  // Basis for Rk-1(F)
  //------------------------------------------------------------------------------

  // Quadrature useful for various spaces to follow (degree might be too large in certain cases, but that is
  // not a major additional cost in 2D)
  QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * m_K);

  if (PolynomialSpaceDimension<Face>::Roly(m_K - 1) > 0) {
    // Non-orthonormalised basis of Rk-1(F). 
    MonomialScalarBasisFace basis_Pk_F(F, m_K);
    ShiftedBasis<MonomialScalarBasisFace> basis_Pk0_F(basis_Pk_F,1);
    CurlBasis<ShiftedBasis<MonomialScalarBasisFace>> basis_Rkmo_F(basis_Pk0_F);
    // Orthonormalise, store and check dimension
    auto basis_Rkmo_F_quad = evaluate_quad<Function>::compute(basis_Rkmo_F, quad_2k_F);
    bases_F.Rolykmo.reset( new RolyBasisFaceType(l2_orthonormalize(basis_Rkmo_F, quad_2k_F, basis_Rkmo_F_quad)) );
    assert( bases_F.Rolykmo->dimension() == PolynomialSpaceDimension<Face>::Roly(m_K - 1) );
  }
  
  //------------------------------------------------------------------------------
  // Basis for Rck(F)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Face>::RolyCompl(m_K) > 0) {
    // Non-orthonormalised
    RolyComplBasisFace basis_Rck_F(F, m_K);
    auto basis_Rck_F_quad = evaluate_quad<Function>::compute(basis_Rck_F, quad_2k_F);
    // Orthonormalise, store and check dimension
    bases_F.RolyComplk.reset( new RolyComplBasisFaceType(l2_orthonormalize(basis_Rck_F, quad_2k_F, basis_Rck_F_quad)) );
    assert ( bases_F.RolyComplk->dimension() == PolynomialSpaceDimension<Face>::RolyCompl(m_K) );
  }

  //------------------------------------------------------------------------------
  // Basis for Rck+2(F)
  //------------------------------------------------------------------------------

  // Non-orthonormalised
  RolyComplBasisFace basis_Rckp2_F(F, m_K+2);
  QuadratureRule quad_2kp2_F = generate_quadrature_rule(F, 2 * (m_K+2) );
  auto basis_Rckp2_F_quad = evaluate_quad<Function>::compute(basis_Rckp2_F, quad_2kp2_F);
  // Orthonormalise, store and check dimension
  bases_F.RolyComplkp2.reset( new RolyComplBasisFaceType(l2_orthonormalize(basis_Rckp2_F, quad_2kp2_F, basis_Rckp2_F_quad)) );
  assert ( bases_F.RolyComplkp2->dimension() == PolynomialSpaceDimension<Face>::RolyCompl(m_K+2) );

  
  return bases_F;
}

//------------------------------------------------------------------------------

DDRCore::EdgeBases DDRCore::_construct_edge_bases(size_t iE)
{
  const Edge & E = *m_mesh.edge(iE);

  EdgeBases bases_E;

  // Basis for Pk+1(E)
  MonomialScalarBasisEdge basis_Pkpo_E(E, m_K + 1);
  QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (m_K + 1));
  auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(basis_Pkpo_E, quad_2kpo_E);
  bases_E.Polykpo.reset( new PolyEdgeBasisType(l2_orthonormalize(basis_Pkpo_E, quad_2kpo_E, basis_Pkpo_E_quad)) );

  // Basis for Pk(E)
  bases_E.Polyk.reset( new RestrictedBasis<PolyEdgeBasisType>(*bases_E.Polykpo, PolynomialSpaceDimension<Edge>::Poly(m_K)) );
  
  // Basis for Pk-1(E)
  if (PolynomialSpaceDimension<Edge>::Poly(m_K - 1) > 0) {
    // Given that the basis for Pk+1(E) is hierarchical, a basis for Pk-1(E)
    // can be obtained by restricting the former
    bases_E.Polykmo.reset( new RestrictedBasis<PolyEdgeBasisType>(*bases_E.Polykpo, PolynomialSpaceDimension<Edge>::Poly(m_K - 1)) );
  }

  return bases_E;
}

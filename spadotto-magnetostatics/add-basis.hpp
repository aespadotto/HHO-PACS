
#ifndef ADDITIONALBASIS_HH
#define ADDITIONALBASIS_HH

#include <basis.hpp>

namespace HArDCore3D {

  /// Scalar monomial basis on a cell + 2nd order derivatives
  class MSBC_2ndderivs: public MonomialScalarBasisCell
  {
  public:
    static const bool Has2Derivatives = true;
    /// Constructor
    MSBC_2ndderivs(
        const Cell &T, ///< A mesh cell
        size_t degree  ///< The maximum polynomial degree to be considered
    ): MonomialScalarBasisCell(T, degree) {};

    /// Evaluate 2nd order partial derivatives (need it to evaluate curl curl )
    inline FunctionValue twoderivs (size_t i, const VectorRd &x, size_t idx1, size_t idx2) const{
      VectorRd y = _coordinate_transform(x);
      VectorZd powers = m_powers[i];
      //std::cout<<"before: "<<powers(0)<<" "<<powers(1)<<" "<<powers(2)<<std::endl;
      double multip_coeff = 1.;
      multip_coeff*=powers(idx1);
      powers(idx1)=(powers(idx1)==0? 0. :powers(idx1)-1);
      multip_coeff*=powers(idx2);
      powers(idx2)=(powers(idx2)==0? 0. :powers(idx2)-1);
      double value = multip_coeff*std::pow(y(0),powers(0))*std::pow(y(1),powers(1))*std::pow(y(2),powers(2));
      //std::cout<<powers(0)<<" "<<powers(1)<<" "<<powers(2)<<" "<<m_hT<<" "<<value<<" "<<multip_coeff<<std::endl;
      return value / (m_hT*m_hT);
    };

    /// Assemble the hessian for basis element i at x
    inline MatrixRd Hessian (size_t i, const VectorRd &x) const {
      MatrixRd hessian = MatrixRd::Zero();
      for (size_t ii = 0; ii<3; ++ii)
         for (size_t jj = 0; jj<3; ++jj)
             hessian (ii, jj) = twoderivs(i, x, ii, jj);
      return hessian;
    };

  };

  // class of face monomials that performs tangent gradient
  class MSFB: public MonomialScalarBasisFace {
  public:
    MSFB (const Face& F, size_t degree): MonomialScalarBasisFace (F, degree) {};
    inline GradientValue gradient_tan (size_t i, const VectorRd& x) const {
      return m_nF.cross(gradient(i, x).cross(m_nF));
    };
  };
  // need to use an alternative shifted basis type
  template <typename BasisType>
  class ShiftedBasis_tan
  {
  public:
    typedef typename BasisType::FunctionValue FunctionValue;
    typedef typename BasisType::GradientValue GradientValue;
    typedef VectorRd CurlValue;
    typedef double DivergenceValue;

    typedef typename BasisType::GeometricSupport GeometricSupport;

    static const TensorRankE tensorRank = BasisType::tensorRank;
    static const bool hasFunction = BasisType::hasFunction;
    static const bool hasGradient = BasisType::hasGradient;
    static const bool hasCurl = BasisType::hasCurl;
    static const bool hasDivergence = BasisType::hasDivergence;

    /// Constructor
    ShiftedBasis_tan(
        const BasisType &basis, ///< A basis
        const int shift         ///< The shift
        )
        : m_basis(basis),
          m_shift(shift)
    {
      // Do nothing
    }

    /// Return the dimension of the basis
    inline size_t dimension() const
    {
      return m_basis.dimension() - m_shift;
    }

    /// Evaluate the i-th basis function at point x
    inline FunctionValue function(size_t i, const VectorRd &x) const
    {
      static_assert(hasFunction, "Call to function() not available");

      return m_basis.function(i + m_shift, x);
    }

    /// Evaluate the gradient of the i-th basis function at point x
    inline GradientValue gradient(size_t i, const VectorRd &x) const
    {
      static_assert(hasGradient, "Call to gradient() not available");

      return m_basis.gradient(i + m_shift, x);
    }

    /// Evaluate the tangent gradient of the i-th basis function at point x
    inline GradientValue gradient_tan(size_t i, const VectorRd &x) const
    {
      static_assert(hasGradient, "Call to gradient() not available");

      return m_basis.gradient_tan(i + m_shift, x);
    }

    /// Evaluate the curl of the i-th basis function at point x
    inline CurlValue curl(size_t i, const VectorRd &x) const
    {
      static_assert(hasCurl, "Call to curl() not available");

      return m_basis.curl(i + m_shift, x);
    }

    /// Evaluate the divergence of the i-th basis function at point x
    inline DivergenceValue divergence(size_t i, const VectorRd &x) const
    {
      static_assert(hasDivergence, "Call to divergence() not available");

      return m_basis.divergence(i + m_shift, x);
    }

  private:
    BasisType m_basis;
    int m_shift;
  };
  // tangent gradient basis
  template <typename BasisType>
  class TangentGradientBasis
  {
  public:
    typedef VectorRd FunctionValue;
    typedef Eigen::Matrix<double, dimspace, dimspace> GradientValue;
    typedef VectorRd CurlValue;
    typedef double DivergenceValue;

    typedef typename BasisType::GeometricSupport GeometricSupport;

    static const TensorRankE tensorRank = Vector;
    static const bool hasFunction = true;
    static const bool hasGradient = false;
    static const bool hasCurl = false;
    static const bool hasDivergence = false;

    /// Constructor
    TangentGradientBasis(const BasisType &basis)
        : m_scalar_basis(basis)
    {
      static_assert(BasisType::tensorRank == Scalar,
                    "Gradient basis can only be constructed starting from scalar bases");
      static_assert(BasisType::hasGradient,
                    "Gradient basis requires gradient() for the original basis to be available");
      // Do nothing
    }

    /// Compute the dimension of the basis
    inline size_t dimension() const
    {
      return m_scalar_basis.dimension();
    }

    /// Evaluate the i-th basis function at point x
    inline FunctionValue function(size_t i, const VectorRd &x) const
    {
      return m_scalar_basis.gradient_tan(i, x);
    }

  private:
    BasisType m_scalar_basis;
  };


//this is only to check consistence between effective and nominal dimension of monomial basis
class checkerface: public MonomialScalarBasisFace {
public:
  checkerface (const Face& F, size_t degree): MonomialScalarBasisFace (F, degree ) {};
  inline void show () const {
    for (size_t i =0; i<this->dimension(); ++i)
        std::cout<<m_powers[i](0)<<" "<<m_powers[i](1)<<std::endl;
  }
};
class checkercell: public MonomialScalarBasisCell {
public:
  checkercell (const Cell& T, size_t degree): MonomialScalarBasisCell (T, degree ) {};
  inline void show () const {
    for (size_t i =0; i<this->dimension(); ++i)
        std::cout<<m_powers[i](0)<<" "<<m_powers[i](1)<<" "<<m_powers[i](2)<<std::endl;
  }
};

// class to implement direct sum of bases

template <typename BasisType1, typename BasisType2>
  class BasisDirectSum {
    public:

      typedef typename BasisType1::FunctionValue FunctionValue;
      typedef typename BasisType1::GradientValue GradientValue;
      typedef VectorRd CurlValue;
      typedef double DivergenceValue;

      typedef typename BasisType1::GeometricSupport GeometricSupport;

      static const TensorRankE tensorRank = BasisType1::tensorRank;
      static const bool hasFunction = BasisType1::hasFunction;
      static const bool hasGradient = BasisType1::hasGradient;
      static const bool hasCurl = BasisType1::hasCurl;
      static const bool hasDivergence = BasisType1::hasDivergence;

      BasisDirectSum (BasisType1 &basis1, BasisType2 &basis2)
                     : m_basis1(basis1),
                       m_basis2(basis2) {
                         static_assert (std::is_same<typename BasisType2::FunctionValue, FunctionValue>::value &&
                                        std::is_same<typename BasisType2::GeometricSupport, GeometricSupport>::value
                                         , "Inconsistent direct sum of bases initialization");
                       };

      inline size_t dimension () const {
        return m_basis1.dimension()+ m_basis2.dimension();
      };

      inline FunctionValue function (size_t i, const VectorRd &x) const {
        static_assert(hasFunction, "Call to function() not available");
        if (i<m_basis1.dimension())
           return m_basis1.function(i, x);
        else
          return m_basis2.function (i-m_basis1.dimension(), x);
      };

      inline GradientValue gradient (size_t i, const VectorRd &x) const {
        static_assert(hasGradient, "Call to gradient() not available");
        if (i<m_basis1.dimension())
           return m_basis1.gradient(i, x);
        else
          return m_basis2.gradient(i-m_basis1.dimension(), x);
      };

   private:
    BasisType1 m_basis1;
    BasisType2 m_basis2;
  };


  // family_with second derivatives

  template <typename BasisType>
  class Family_2der
  {
  public:
    typedef typename BasisType::FunctionValue FunctionValue;
    typedef typename BasisType::GradientValue GradientValue;
    typedef VectorRd CurlValue;
    typedef double DivergenceValue;

    typedef typename BasisType::GeometricSupport GeometricSupport;

    static const TensorRankE tensorRank = BasisType::tensorRank;
    static const bool hasFunction = BasisType::hasFunction;
    static const bool hasGradient = BasisType::hasGradient;
    static const bool hasCurl = BasisType::hasCurl;
    static const bool hasDivergence = BasisType::hasDivergence;
    static const bool Has2Derivatives = BasisType::Has2Derivatives;

    /// Constructor
    Family_2der(
        const BasisType &basis,       ///< The basis in which the family is expressed
        const Eigen::MatrixXd &matrix ///< The coefficient matrix whose i-th line contains the coefficient of the expansion of the i-th function of the family in the basis
        )
        : m_basis(basis),
          m_matrix(matrix)
    {
      assert((size_t)matrix.cols() == basis.dimension() || "Inconsistent family initialization");
    }

    /// Dimension of the family. This is actually the number of functions in the family, not necessarily linearly independent
    inline size_t dimension() const
    {
      return m_matrix.rows();
    }

    /// Evaluate the i-th function at point x
    FunctionValue function(size_t i, const VectorRd &x) const
    {
      static_assert(hasFunction, "Call to function() not available");

      FunctionValue f = m_matrix(i, 0) * m_basis.function(0, x);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        f += m_matrix(i, j) * m_basis.function(j, x);
      } // for j
      return f;
    }

    /// Evaluate the i-th function at a quadrature point iqn, knowing all the values of ancestor basis functions at the quadrature nodes (provided by eval_quad)
    FunctionValue function(size_t i, size_t iqn, const boost::multi_array<FunctionValue, 2> &ancestor_value_quad) const
    {
      static_assert(hasFunction, "Call to function() not available");

      FunctionValue f = m_matrix(i, 0) * ancestor_value_quad[0][iqn];
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        f += m_matrix(i, j) * ancestor_value_quad[j][iqn];
      } // for j
      return f;
    }


    /// Evaluate the gradient of the i-th function at point x
    GradientValue gradient(size_t i, const VectorRd &x) const
    {
      static_assert(hasGradient, "Call to gradient() not available");

      GradientValue G = m_matrix(i, 0) * m_basis.gradient(0, x);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        G += m_matrix(i, j) * m_basis.gradient(j, x);
      } // for j
      return G;
    }

    /// Evaluate the gradient of the i-th function at a quadrature point iqn, knowing all the gradients of ancestor basis functions at the quadrature nodes (provided by eval_quad)
    GradientValue gradient(size_t i, size_t iqn, const boost::multi_array<GradientValue, 2> &ancestor_gradient_quad) const
    {
      static_assert(hasGradient, "Call to gradient() not available");

      GradientValue G = m_matrix(i, 0) * ancestor_gradient_quad[0][iqn];
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        G += m_matrix(i, j) * ancestor_gradient_quad[j][iqn];
      } // for j
      return G;
    }

    /// Evaluate the curl of the i-th function at point x
    CurlValue curl(size_t i, const VectorRd &x) const
    {
      static_assert(hasCurl, "Call to curl() not available");

      CurlValue C = m_matrix(i, 0) * m_basis.curl(0, x);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        C += m_matrix(i, j) * m_basis.curl(j, x);
      } // for j
      return C;
    }

    /// Evaluate the curl of the i-th function at a quadrature point iqn, knowing all the curls of ancestor basis functions at the quadrature nodes (provided by eval_quad)
    CurlValue curl(size_t i, size_t iqn, const boost::multi_array<CurlValue, 2> &ancestor_curl_quad) const
    {
      static_assert(hasCurl, "Call to curl() not available");

      CurlValue C = m_matrix(i, 0) * ancestor_curl_quad[0][iqn];
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        C += m_matrix(i, j) * ancestor_curl_quad[j][iqn];
      } // for j
      return C;
    }

    /// Evaluate the divergence of the i-th function at point x
    DivergenceValue divergence(size_t i, const VectorRd &x) const
    {
      static_assert(hasDivergence, "Call to divergence() not available");

      DivergenceValue D = m_matrix(i, 0) * m_basis.divergence(0, x);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        D += m_matrix(i, j) * m_basis.divergence(j, x);
      } // for j
      return D;
    }

    /// Evaluate the divergence of the i-th function at a quadrature point iqn, knowing all the divergences of ancestor basis functions at the quadrature nodes (provided by eval_quad)
    DivergenceValue divergence(size_t i, size_t iqn, const boost::multi_array<DivergenceValue, 2> &ancestor_divergence_quad) const
    {
      static_assert(hasDivergence, "Call to divergence() not available");

      DivergenceValue D = m_matrix(i, 0) * ancestor_divergence_quad[0][iqn];
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        D += m_matrix(i, j) * ancestor_divergence_quad[j][iqn];
      } // for j
      return D;
    }

    /// Return the coefficient matrix
    inline const Eigen::MatrixXd &matrix() const
    {
      return m_matrix;
    }

    /// Return the ancestor
    inline const BasisType &ancestor() const
    {
      return m_basis;
    }

    /// Evaluate the i-th 2nd derivative at point x
    FunctionValue twodervis (size_t i, const VectorRd &x, size_t idx1, size_t idx2) const
    {
      static_assert(Has2Derivatives, "Call to 2nd deriv() not available");

      FunctionValue f = m_matrix(i, 0) * m_basis.twoderivs(0, x, idx1, idx2);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        f += m_matrix(i, j) * m_basis.twoderivs(j, x, idx1, idx2);
      } // for j
      return f;
    }

    MatrixRd twodervis (size_t i, const VectorRd &x) const
    {
      static_assert(Has2Derivatives, "Call to Hessian () not available");

      MatrixRd M = m_matrix(i, 0) * m_basis.twoderivs(0, x);
      for (auto j = 1; j < m_matrix.cols(); j++)
      {
        M += m_matrix(i, j) * m_basis.twoderivs(j, x);
      } // for j
      return M;
    }

  protected:
    BasisType m_basis;
    Eigen::MatrixXd m_matrix;
  };

  // variant of class CurlBasis allowing curl calculation
  // template argument must be a monomial scalar basis
  // in future extension could be given for families
  //-->maybe to fix it should treat it as a shifted basis, with shift = 3 (exclude tensorization of constant basis function)

        template <typename BasisType>
        class CurlBasis_permissive
        {
        public:
          typedef VectorRd FunctionValue;
          typedef Eigen::Matrix<double, dimspace, dimspace> GradientValue;
          typedef VectorRd CurlValue; //mod
          typedef double DivergenceValue;

          typedef typename BasisType::GeometricSupport GeometricSupport;

          static const TensorRankE tensorRank = Vector;
          static const bool hasFunction = true;
          static const bool hasGradient = false;
          static const bool hasCurl = true; //mod
          static const bool hasDivergence = false;

          /// Constructor
          CurlBasis_permissive(const BasisType &basis)
              : m_basis(basis),
                m_tensorized_basis (basis)
          {
            static_assert((BasisType::tensorRank == Scalar && std::is_same<typename BasisType::GeometricSupport, Cell>::value),
                          "Permissive Curl basis can only be constructed starting from scalar bases on elements");
            static_assert(BasisType::Has2Derivatives,
                          "Curl basis requires 2nd derivatives for the original basis to be available");
          }

          /// Compute the dimension of the basis
          inline size_t dimension() const
          {
            return m_tensorized_basis.dimension()-3;
          }

          /// Evaluate the i-th basis function at point x
          inline FunctionValue function(size_t i, const VectorRd &x) const
          {
            return m_tensorized_basis.curl(i+3, x);
          }

          /// Evaluate the i-th basis curl at point x
          inline CurlValue curl(size_t i, const VectorRd &x) const
          {
          size_t shift = i+3;
          size_t generator = shift/3;
          MatrixRd hessian = m_basis.Hessian (generator, x);
          size_t non0_comp = shift%3;
          CurlValue curl_curl = VectorRd::Zero();
          for (size_t ii = 0; ii<3; ++ii)
            curl_curl (ii) = (ii==non0_comp ?
                              -hessian.trace()+hessian(ii, ii):
                              hessian(non0_comp,ii));
          return curl_curl;
          }

        private:
          size_t m_degree;
          BasisType m_basis;
          TensorizedVectorFamily<BasisType, 3> m_tensorized_basis;
        };


/// function to flush out from  a family null and repeated or linearly correlated elements (based on numerical evaluation)
/// works for vector valued basis (originally thought to filter a curl rec basis)

template <typename BasisType>
Family<BasisType> Filter (const BasisType &broadbasis, const QuadratureRule &quad_rule) {
    Eigen::MatrixXd new_matrix(0,0);
    std::vector<size_t>inserted;
    auto evaluation = evaluate_quad<Function>::compute(broadbasis, quad_rule);
    for (size_t i = 0; i<broadbasis.dimension(); ++i){
        bool to_be_inserted = true;

        // check whether it is null
        double is_null = 0.;
        for (size_t iqn = 0; iqn< quad_rule.size(); ++iqn) {
            is_null+=std::abs(evaluation[i][iqn].norm());
        }//for iqn
        if (is_null<1e-12) {to_be_inserted = false;
        //std::cout<<"Is null!"<<std::endl;
         }

        for (size_t j = 0; j<inserted.size(); ++j){
            // two basis elements are alligned if |<a|b>|==||a||*||b||; filter them
            double product = 0.;
            double normisq = 0.;
            double normjsq = 0.;
            for (size_t iqn = 0; iqn< quad_rule.size(); ++iqn) {
                product+=std::abs(evaluation[i][iqn].adjoint()*evaluation[inserted[j]][iqn])*quad_rule[iqn].w;
                normisq+=std::abs(evaluation[i][iqn].adjoint()*evaluation[i][iqn])*quad_rule[iqn].w;
                normjsq+=std::abs(evaluation[inserted[j]][iqn].adjoint()*evaluation[inserted[j]][iqn])*quad_rule[iqn].w;
            }//for iqn
            //std::cout<<"[debugg] "<<product<<" "<<normisq<<" "<<normjsq<<std::endl;
            if (std::abs(product*product-normisq*normjsq)<1e-12) {to_be_inserted = false;
            //std::cout<<"Alligned"<<std::endl;
             }
        }//for j
    if (to_be_inserted) {
        inserted.push_back(i);
        //std::cout<<"Inserted: "<<i<<std::endl;
        new_matrix.conservativeResize(new_matrix.rows()+1, broadbasis.dimension());
        new_matrix.row(new_matrix.rows()-1).setZero();
        new_matrix(new_matrix.rows() - 1, i) = 1.;
       }
    }// for i
    return Family<BasisType> (broadbasis, new_matrix);
};

} //end of namespace HArDCore3D

#endif

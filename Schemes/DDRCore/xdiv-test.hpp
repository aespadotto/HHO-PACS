#ifndef XDIV_TEST_HPP
#define XDIV_TEST_HPP

#include <boost/math/constants/constants.hpp>

namespace HArDCore3D
{
  
  static const double PI = boost::math::constants::pi<double>();
  using std::sin;

  //------------------------------------------------------------------------------
  
  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  constant_vector = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
		      return Eigen::Vector3d(1., 2., 3.);
		    };

  static std::function<double(const Eigen::Vector3d&)>
  div_constant_vector = [](const Eigen::Vector3d & x) -> double {
			  return 0.;
			};

  //------------------------------------------------------------------------------
  
  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  trigonometric_vector = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
			   return Eigen::Vector3d(
						  sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
						  sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
						  sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2))
						  );
			 };

  static std::function<double(const Eigen::Vector3d&)>
  div_trigonometric_vector = [](const Eigen::Vector3d & x) -> double {
			       return PI * (
					    cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2))
					    + sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2))
					    + sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2))
					    );
			     };

  //------------------------------------------------------------------------------

  template<typename T>
  double squared_l2_error(
			  const std::function<T(const Eigen::Vector3d &)> & f,
			  const Eigen::VectorXd & fX,
			  const boost::multi_array<T, 2> & fX_basis_quad,
			  const QuadratureRule & quad_X
			  )
  {
    // Check that the dimensions are consistent
    assert(
	   fX_basis_quad.shape()[0] == (size_t)fX.size() &&
	   fX_basis_quad.shape()[1] == quad_X.size()
	   );
    
    double err = 0.;

    for (size_t iqn = 0; iqn < quad_X.size(); iqn++) {
      T f_iqn = f(quad_X[iqn].vector());
      
      T fX_iqn = fX(0) * fX_basis_quad[0][iqn];
      for (size_t i = 1; i < fX_basis_quad.shape()[0]; i++) {
	fX_iqn += fX(i) * fX_basis_quad[i][iqn];
      } // for i

      T diff_iqn = f_iqn - fX_iqn;

      err += quad_X[iqn].w * scalar_product(diff_iqn, diff_iqn);
    } // for iqn

    return err;
  }

} // end of namespace HArDCore3D

#endif

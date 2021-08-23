#ifndef DDRSPACES_TEST_HPP
#define DDRSPACES_TEST_HPP

#include <boost/math/constants/constants.hpp>

namespace HArDCore3D
{
  
  static const double PI = boost::math::constants::pi<double>();
  using std::sin;

  //------------------------------------------------------------------------------
  
  static std::function<double(const Eigen::Vector3d&)>
  trigonometric_scalar = [](const Eigen::Vector3d & x) -> double {
			   return sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2));
			 };
  
  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  grad_trigonometric_scalar = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
				return PI * Eigen::Vector3d(
							    cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
							    sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2)),
							    sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2))
							    );
			      };
  //------------------------------------------------------------------------------

  static std::function<double(const Eigen::Vector3d&)>
  constant_scalar = [](const Eigen::Vector3d & x) -> double {
		      return 1.;
		    };

  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  grad_constant_scalar = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
			   return Eigen::Vector3d::Zero();
			 };

  //------------------------------------------------------------------------------

  static std::function<double(const Eigen::Vector3d&)>
  linear_scalar = [](const Eigen::Vector3d & x) -> double {
		    return 1. + x(0) + 2. * x(1) + 3. * x(2);
		  };

  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  grad_linear_scalar = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
			 return Eigen::Vector3d(1., 2., 3.);
		       };

  //------------------------------------------------------------------------------
  
  static std::function<double(const Eigen::Vector3d&)>
  quadratic_scalar = [](const Eigen::Vector3d & x) -> double {
		       return linear_scalar(x) + std::pow(x(0), 2) + 2. * std::pow(x(1), 2) + 3. * std::pow(x(2), 2);
		     };

  static std::function<Eigen::Vector3d(const Eigen::Vector3d&)>
  grad_quadratic_scalar = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
			    return grad_linear_scalar(x) + Eigen::Vector3d(2. * x(0), 4. * x(1), 6. * x(2));
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

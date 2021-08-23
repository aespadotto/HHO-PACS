#ifndef DDR_CORE_TEST_HPP
#define DDR_CORE_TEST_HPP

#include <boost/math/constants/constants.hpp>

#include <basis.hpp>

namespace HArDCore3D
{
  
  static const double PI = boost::math::constants::pi<double>();
  using std::sin;

  static auto q = [](const Eigen::Vector3d & x) -> double {
		    return sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2));
		  };

  static auto grad_q = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
			 return PI * Eigen::Vector3d(
						     cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
						     sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2)),
						     sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2))
						     );
		       };

  static auto v = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
		    return Eigen::Vector3d(
					   sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
					   sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
					   sin(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2))
					   );
		  };

  static auto curl_v
  = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
      return PI * Eigen::Vector3d(
				  sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2)) - sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2)),
				  sin(PI * x(0)) * sin(PI * x(1)) * cos(PI * x(2)) - cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)),
				  cos(PI * x(0)) * sin(PI * x(1)) * sin(PI * x(2)) - sin(PI * x(0)) * cos(PI * x(1)) * sin(PI * x(2))
				  );
    };
   
} // end of namespace HArDCore3D
#endif

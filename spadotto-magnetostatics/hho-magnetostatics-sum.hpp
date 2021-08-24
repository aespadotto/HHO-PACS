// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr) (original version in Schemes/HHO-magnetostatics)
// Partially adapted by Aurelio Spadotto

#ifndef HHO_MAGNETOSTATICS_SUM_HPP
#define HHO_MAGNETOSTATICS_SUM_HPP

#include <iostream>
#include <memory>
#include <tuple>

#include <boost/math/constants/constants.hpp>

#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <BoundaryConditions.hpp>

#include <yspace.hpp>
#include "xspace-sum-evol.hpp"
#include "cartesianproducthhospace.hpp"

namespace HArDCore3D
{

  /// Assemble a magnetostatic problem using the HHO method
  struct HHOMagnetostatics
  {
    typedef CartesianProductHHOSpace<2> ZSpace;

    typedef Eigen::SparseMatrix<double> SystemMatrixType;

    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> CurrentDensityType;
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> VectorPotentialType;
    typedef std::function<double(const Eigen::Vector3d &)> PressureType;

    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> MagneticFieldType;
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> PressureGradientType;

    typedef std::pair<Eigen::MatrixXd, Eigen::VectorXd> LocalContribution;

    enum AssemblyTypeE
      {
       Full,
       StaticCondensation
      };

    /// Constructor
    HHOMagnetostatics(
                      const Mesh & mesh,                ///< Mesh
                      size_t K,                         ///< Polynomial degree
                      const BoundaryConditions & bc,    ///< Boundary conditions
                      bool use_threads = true,          ///< Use parallelism
                      std::ostream & output = std::cout ///< Output stream to print status messages
                      );

    /// Returns the mesh
    inline const Mesh & mesh() const
    {
      return m_mesh;
    }

    /// Returns the degree
    inline size_t degree() const
    {
      return m_degree;
    }

    /// Returns the space Xh
    inline const XSpace & xSpace() const
    {
      return m_xspace;
    }

    /// Returns the space Yh
    inline const YSpace & ySpace() const
    {
      return m_yspace;
    }

    /// Returns the space Zh
    inline const ZSpace & zSpace() const
    {
      return m_zspace;
    }

    /// Returns the linear system matrix
    inline const SystemMatrixType & systemMatrix() const {
      return m_A;
    }

    /// Returns the linear system matrix
    inline SystemMatrixType & systemMatrix() {
      return m_A;
    }

    /// Returns the linear system right-hand side vector
    inline const Eigen::VectorXd & systemVector() const {
      return m_b;
    }

    /// Returns the linear system right-hand side vector
    inline Eigen::VectorXd & systemVector() {
      return m_b;
    }

    /// Total number of local face DOFs
    inline size_t numLocalDofsFace() const {
      return m_xspace.numLocalDofsFace() + m_yspace.numLocalDofsFace();
    }

    /// Total number of local cell DOFs
    inline size_t numLocalDofsCell() const {
      return m_xspace.numLocalDofsCell() + m_yspace.numLocalDofsCell();
    }

    /// Assemble the global system
    template<AssemblyTypeE AssemblyType = StaticCondensation>
    void assembleLinearSystem(
                              const Mesh & m,
                              const CartesianProductHHOSpace<2> & Zspace,
                              const YSpace & Yspace,
                              const CurrentDensityType & f,  ///< Current density
                              const VectorPotentialType & u, ///< Vector potential
                              const PressureType & p,         ///< Pressure
                              const std::string bc_type,      ///< BCs
                              const MagneticFieldType & b
                              );

    /// Returns the dimension of the linear system
    template<AssemblyTypeE AssemblyType>
    inline size_t linearSystemDimension() const
    {
      if constexpr(AssemblyType == StaticCondensation) {
          return m_mesh.n_faces() * zSpace().numLocalDofsFace();
        } else if constexpr(AssemblyType == Full) {
          return zSpace().dimension();
        }
    }

  //private:
    /// Compute the local contribution for the element of index iT (TOUCHED)
    LocalContribution _compute_local_contribution(
                                                  size_t iT,                     ///< Element index
                                                  const CurrentDensityType & f,  ///< Current density
                                                  const VectorPotentialType & u, ///< Vector potential
                                                  const PressureType & p,         ///< Pressure
                                                  const std::string bc_type,
                                                  const MagneticFieldType & b
  );

    /// Assemble the local contribution for the element of index iT after performing static condensation
    void _assemble_statically_condensed(
                                        size_t iT,                                        ///< Element index
                                        const LocalContribution & lsT,                    ///< Local contribution
                                        std::list<Eigen::Triplet<double> > & my_triplets, ///< List of triplets
                                        Eigen::VectorXd & my_rhs,                          ///< Vector
                                        const std::string bc_type
    );

   /// Calculates cell unknowns for the element of index iT after solving statically condensed system (needed to calculate L2 error)
   /// NEW METHOD
    void _reconstruct_cell_unknowns(
                                   size_t iT,                                        ///< Element index
                                   Eigen::VectorXd & face_unknowns,                  ///< Face unknowns
                                   Eigen::VectorXd & cell_unknowns,                  ///< Vector (of cell unknowns)
                                   const CurrentDensityType & f,                     ///< Current density
                                   const VectorPotentialType & u,                    ///< Vector potential
                                   const PressureType & p,                            ///< Pressure
                                   const std::string bc_type,
                                   const MagneticFieldType & b
                                   );

    /// Assemble the local contribution for the element of index iT into the global system
    void _assemble_full(
                        size_t iT,                                        ///< Element index
                        const LocalContribution & lsT,                    ///< Local contribution
                        std::list<Eigen::Triplet<double> > & my_triplets, ///< List of triplets
                        Eigen::VectorXd & my_rhs,                          ///< Vector
                        const std::string bc_type
    );
    const Mesh & m_mesh;
    size_t m_degree;
    BoundaryConditions m_bc;
    bool m_use_threads;
    std::ostream & m_output;

    XSpace m_xspace;
    YSpace m_yspace;
    ZSpace m_zspace;

    SystemMatrixType m_A;
    Eigen::VectorXd m_b;
  };

  //------------------------------------------------------------------------------

  template<HHOMagnetostatics::AssemblyTypeE AssemblyType>
  void HHOMagnetostatics::assembleLinearSystem(
                                               const Mesh & m,
                                               const CartesianProductHHOSpace <2> & Zspace,
                                               const YSpace & Yspace,
                                               const CurrentDensityType & f,
                                               const VectorPotentialType & u,
                                               const PressureType & p,
                                               const std::string bc_type,
                                               const MagneticFieldType & b)
  {
    // Initialize the system matrix and vector
    m_A.resize(linearSystemDimension<AssemblyType>(), linearSystemDimension<AssemblyType>());
    m_b = Eigen::VectorXd::Zero(linearSystemDimension<AssemblyType>());


    // Define a functor that assembles local contributions for a stride of elements
    std::function<void(size_t, size_t, std::list<Eigen::Triplet<double> > * , Eigen::VectorXd *)> assemble_all;
    if constexpr(AssemblyType == StaticCondensation) {
        assemble_all = [this, f, u, p, bc_type, &b](
                                       size_t start,
                                       size_t end,
                                       std::list<Eigen::Triplet<double> > * my_triplets,
                                       Eigen::VectorXd * my_rhs
                                       )->void
                       {
                         for (size_t iT = start; iT < end; iT++) {
                           this->_assemble_statically_condensed(
                                                                iT,
                                                                this->_compute_local_contribution(iT, f, u, p, bc_type, b),
                                                                *my_triplets,
                                                                *my_rhs,
                                                                bc_type
                                                                );
                         } // for iT
                       };
      } else if constexpr(AssemblyType == Full) {
        assemble_all = [this, f, u, p, bc_type, &b](
                                       size_t start,
                                       size_t end,
                                       std::list<Eigen::Triplet<double> > * my_triplets,
                                       Eigen::VectorXd * my_rhs
                                       )->void
                       {
                         for (size_t iT = start; iT < end; iT++) {
                           this->_assemble_full(
                                                iT,
                                                this->_compute_local_contribution(iT, f, u, p, bc_type, b),
                                                *my_triplets,
                                                *my_rhs,
                                                bc_type
                                                );
                         } // for iT
                       };
      }


    if (m_use_threads) {
      m_output << "[HHOMagnetostatics] Parallel assembly" << std::endl;

      // Select the number of threads
      unsigned nb_threads_hint = std::thread::hardware_concurrency();
      unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

      // Compute the batch size and the remainder
      unsigned nb_elements = mesh().n_cells();
      unsigned batch_size = nb_elements / nb_threads;
      unsigned batch_remainder = nb_elements % nb_threads;

      // Create vectors of triplets and vectors
      std::vector<std::list<Eigen::Triplet<double> > > triplets(nb_threads + 1);
      std::vector<Eigen::VectorXd> rhs(nb_threads + 1);

      for (unsigned i = 0; i < nb_threads + 1; i++) {
        rhs[i] = Eigen::VectorXd::Zero(this->linearSystemDimension<AssemblyType>());
      } // for i

      // Assign a task to each thread
      std::vector<std::thread> my_threads(nb_threads);
      for (unsigned i = 0; i < nb_threads; ++i) {
        int start = i * batch_size;
        my_threads[i] = std::thread(assemble_all, start, start + batch_size, &triplets[i], &rhs[i]);
      }

      // Execute the elements left
      int start = nb_threads * batch_size;
      assemble_all(start, start + batch_remainder, &triplets[nb_threads], &rhs[nb_threads]);

      // Wait for the other thread to finish their task
      std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

      // Create matrix from triplets
      size_t n_triplets = 0;
      for (auto triplets_thread : triplets) {
        n_triplets += triplets_thread.size();
      }
      std::vector<Eigen::Triplet<double> > all_triplets(n_triplets);
      auto triplet_index = all_triplets.begin();
      for (auto triplets_thread : triplets) {
        triplet_index = std::copy(triplets_thread.begin(), triplets_thread.end(), triplet_index);
      }
      m_A.setFromTriplets(all_triplets.begin(), all_triplets.end());
      for (auto rhs_thread : rhs) {
        m_b += rhs_thread;
      }
    } else {
      m_output << "[HHOMagnetostatics] Sequential assembly" << std::endl;
      std::list<Eigen::Triplet<double> > triplets;
      assemble_all(0, mesh().n_cells(), &triplets, &m_b);
      m_A.setFromTriplets(triplets.begin(), triplets.end());
    }

  }


  //------------------------------------------------------------------------------
  // Exact solutions
  //------------------------------------------------------------------------------

  static const double PI = boost::math::constants::pi<double>();
  using std::sin;
  using std::cos;

  //------------------------------------------------------------------------------
  // Trigonometric solution ( boundary condition in B (or Dirichlet))

  static HHOMagnetostatics::VectorPotentialType
  trigonometric_u = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                      return Eigen::Vector3d(
                                             sin(PI*x(1))*sin(PI*x(2)),
                                             sin(PI*x(0))*sin(PI*x(2)),
                                             sin(PI*x(0))*sin(PI*x(1))
                                             );
                    };

  static HHOMagnetostatics::MagneticFieldType
  trigonometric_b = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                      return PI * Eigen::Vector3d(
                                                  sin(PI*x(0))*(cos(PI*x(1)) - cos(PI*x(2))),
                                                  sin(PI*x(1))*(cos(PI*x(2)) - cos(PI*x(0))),
                                                  sin(PI*x(2))*(cos(PI*x(0)) - cos(PI*x(1)))
                                                  );
                    };

  static HHOMagnetostatics::PressureType
  trigonometric_p = [](const Eigen::Vector3d & x) -> double {
                      return sin(PI*x(0))*sin(PI*x(1))*sin(PI*x(2));
                    };

  static HHOMagnetostatics::PressureGradientType
  trigonometric_grad_p = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                           return PI * Eigen::Vector3d(
                                                       cos(PI*x(0))*sin(PI*x(1))*sin(PI*x(2)),
                                                       sin(PI*x(0))*cos(PI*x(1))*sin(PI*x(2)),
                                                       sin(PI*x(0))*sin(PI*x(1))*cos(PI*x(2))
                                                       );
                         };

  static HHOMagnetostatics::CurrentDensityType
  trigonometric_f = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                      return 2*std::pow(PI,2)*Eigen::Vector3d(sin(PI*x(1))*sin(PI*x(2)), sin(PI*x(0))*sin(PI*x(2)), sin(PI*x(0))*sin(PI*x(1)))
                        + trigonometric_grad_p(x);
                    };

  //------------------------------------------------------------------------------
  // Linear solution

  static HHOMagnetostatics::VectorPotentialType
  linear_u = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
               return Eigen::Vector3d(x(0), -x(1), 0.);
             };

  static HHOMagnetostatics::MagneticFieldType
  linear_b = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
               return Eigen::Vector3d::Zero();
             };

  static HHOMagnetostatics::PressureType
  linear_p = [](const Eigen::Vector3d & x) -> double {
               return x(0) + 2. * x(1);
             };

  static HHOMagnetostatics::PressureGradientType
  linear_grad_p = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
                    return Eigen::Vector3d(1., 2., 0.);
                  };

  static HHOMagnetostatics::CurrentDensityType
  linear_f = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
               return linear_grad_p(x);
             };


    //------------------------------------------------------------------------------
    // Trigonometric solution (homogeneous boundary condition in H (or Neumann))

    static HHOMagnetostatics::VectorPotentialType
            trigonometric_u_H = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return Eigen::Vector3d(
                sin(PI*x(0))*cos(PI*x(1))*cos(PI*x(2)),
                cos(PI*x(0))*sin(PI*x(1))*cos(PI*x(2)),
                -2*cos(PI*x(0))*cos(PI*x(1))*sin(PI*x(2))
        );
    };

    static HHOMagnetostatics::MagneticFieldType
            trigonometric_b_H = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return  PI*Eigen::Vector3d(
                3*cos(PI*x(0))*sin(PI*x(1))*sin(PI*x(2)),
                -3*sin(PI*x(0))*cos(PI*x(1))*sin(PI*x(2)),
                0
                );
    };

    static HHOMagnetostatics::PressureType
            trigonometric_p_H = [](const Eigen::Vector3d & x) -> double {
        return 0.;
    };

    static HHOMagnetostatics::PressureGradientType
            trigonometric_grad_p_H = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return PI * Eigen::Vector3d(
                0.,
                0.,
                0.
        );
    };

    static HHOMagnetostatics::CurrentDensityType
            trigonometric_f_H = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return PI*PI*Eigen::Vector3d(
                3*sin(PI*x(0))*cos(PI*x(1))*cos(PI*x(2)),
                3*cos(PI*x(0))*sin(PI*x(1))*cos(PI*x(2)),
                -6*cos(PI*x(0))*cos(PI*x(1))*sin(PI*x(2))
                );
    };



    //------------------------------------------------------------------------------
    // Trigonometric solution (non homogeneous mixed bundary conditions)

    static HHOMagnetostatics::VectorPotentialType
            trigonometric_u_M = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return Eigen::Vector3d(
                -4*sin(0.5*PI*(x(0)-1))*cos(PI*x(1))*cos(PI*x(2)),
                cos(0.5*PI*(x(0)-1))*sin(PI*x(1))*cos(PI*x(2)),
                cos(0.5*PI*(x(0)-1))*cos(PI*x(1))*sin(PI*x(2))
        );
    };

    static HHOMagnetostatics::MagneticFieldType
            trigonometric_b_M = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return  PI*Eigen::Vector3d(
                0,
                 4.5*sin(0.5*PI*(x(0)-1))*cos(PI*x(1))*sin(PI*x(2)),
                -4.5*sin(0.5*PI*(x(0)-1))*sin(PI*x(1))*cos(PI*x(2))
        );
    };

    static HHOMagnetostatics::PressureType
            trigonometric_p_M = [](const Eigen::Vector3d & x) -> double {
        return 0.;
    };

    static HHOMagnetostatics::PressureGradientType
            trigonometric_grad_p_M = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return PI * Eigen::Vector3d(
                0.,
                0.,
                0.
        );
    };

    static HHOMagnetostatics::CurrentDensityType
            trigonometric_f_M = [](const Eigen::Vector3d & x) -> Eigen::Vector3d {
        return 0.25*9*PI*PI*Eigen::Vector3d(
                -4*sin(0.5*PI*(x(0)-1))*cos(PI*x(1))*cos(PI*x(2)),
                cos(0.5*PI*(x(0)-1))*sin(PI*x(1))*cos(PI*x(2)),
                cos(0.5*PI*(x(0)-1))*cos(PI*x(1))*sin(PI*x(2))
        );
    };

} // namespace HArDCore3D

#endif

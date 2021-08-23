// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <thread>

#include <boost/program_options.hpp>

#include <mesh.hpp>
#include <mesh_builder.hpp>

#include <BoundaryConditions/BoundaryConditions.hpp>

#include "xspace.hpp"
#include "yspace.hpp"
#include "cartesianproducthhospace.hpp"

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Mesh filenames
//------------------------------------------------------------------------------

const std::string mesh_dir = "/home/aurelio/HArDCore3D/meshes/";
std::string default_mesh = mesh_dir + "Cubic-Cells/RF_fmt/gcube_2x2x2";
std::string default_meshtype = "RF";

//------------------------------------------------------------------------------
  
int main(int argc, const char* argv[])
{

  // Program options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<std::string>(), "Set the mesh")
    ("meshtype,t", boost::program_options::value<std::string>(), "Set the mesh type (TG,MSH,RF)")
    ("degree,k", boost::program_options::value<size_t>()->default_value(1), "The polynomial degree of the HHO space")
    ("pthread,p", boost::program_options::value<bool>()->default_value(true), "Use thread-based parallelism")
    ("export-matrix,e", "Export matrix to Matrix Market format")
    ("static-condensation,s", "Perform static condensation");
  
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  // Display the help options
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  // Select the mesh
  std::string mesh_file = (vm.count("mesh") ? vm["mesh"].as<std::string>() : default_mesh);
  std::string mesh_type = (vm.count("meshtype") ? vm["meshtype"].as<std::string>() : default_meshtype);

  std::cout << FORMAT(25) << "[main] Mesh file" << mesh_file << std::endl;
  
  // Select the degree
  
  size_t K = (vm.count("degree") ? vm["degree"].as<size_t>() : 0);
  std::cout << FORMAT(25) << "[main] Degree" << K << std::endl;

  // Build the mesh
  MeshBuilder meshbuilder = MeshBuilder(mesh_file, mesh_type);
  std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();

  // Reorder mesh faces so that boundary faces are put at the end
  BoundaryConditions bc("D", *mesh_ptr.get());
  bc.reorder_faces();

  // Create the HHO spaces
  bool use_threads = (vm.count("pthread") ? vm["pthread"].as<bool>() : true);
  std::cout << "[main] " << (use_threads ? "Parallel execution" : "Sequential execution") << std:: endl;
  
  XSpace xspace(*mesh_ptr, K + 1, use_threads);
  YSpace yspace(*mesh_ptr, K + 1, use_threads);
  CartesianProductHHOSpace<2> zspace(typename CartesianProductHHOSpace<2>::CartesianFactorsArray{{ &xspace, &yspace}});

  std::cout << "XSpace" << std::endl;
  std::cout << "- num local DOFs cell : " << xspace.numLocalDofsCell() << std::endl;
  std::cout << "- num local DOFs face : " << xspace.numLocalDofsFace() << std::endl;
  
  std::cout << "YSpace" << std::endl;
  std::cout << "- num local DOFs cell : " << yspace.numLocalDofsCell() << std::endl;
  std::cout << "- num local DOFs face : " << yspace.numLocalDofsFace() << std::endl;

  std::cout << "ZSpace" << std::endl;
  std::cout << "- num local DOFs cell : " << zspace.numLocalDofsCell() << std::endl;
  std::cout << "- num local DOFs face : " << zspace.numLocalDofsFace() << std::endl;
  
  for (size_t iT = 0; iT < mesh_ptr->n_cells(); iT++) {
    const Cell & T = *mesh_ptr->cell(iT);
    
    std::cout << "dimension ZT : " << zspace.dimensionCell(T) << ", "
              << "localOffset comp0 T" << iT << " : " << zspace.localOffset(T, 0) << ", "
              << "localOffset comp1 T" << iT << " : " << zspace.localOffset(T, 1)
              << std::endl;

    for (size_t iF = 0; iF < T.n_faces(); iF++) {
      const Face & F = *T.face(iF);
        
      std::cout << "-  localOffset comp0 F" << iF << " : " << zspace.localOffset(T, F, 0) << ", "
                << "-  localOffset comp1 F" << iF << " : " << zspace.localOffset(T, F, 1)
                << std::endl;

    } // for iF

    auto IT_pT = zspace.globalCellBoundaryDofIndices(T);
    // std::cout << "Boundary DOF indices : " << std::flush;
    // for (auto it_IT_pT = IT_pT.begin(); it_IT_pT != IT_pT.end(); it_IT_pT++) {
    //   std::cout << *it_IT_pT << (it_IT_pT == IT_pT.end() - 1 ? "\n" : ", ") << std::flush;      
    // } // for it_IT_pT
    auto IT_T = zspace.globalCellDofIndices(T);
    // std::cout << "Internal DOF indices : " << std::flush;
    // for (auto it_IT_T = IT_T.begin(); it_IT_T != IT_T.end(); it_IT_T++) {
    //   std::cout << *it_IT_T << (it_IT_T == IT_T.end() - 1 ? "\n" : ", ") << std::flush;      
    // } // for it_IT_T
    // std::vector<size_t> IT(zspace.dimensionCell(T));
    // typename std::vector<size_t>::iterator it_IT;
    // it_IT = std::copy(IT_pT.begin(), IT_pT.end(), it_IT);
    // std::copy(IT_T.begin(), IT_T.end(), it_IT);    
    
  } // for iT
  
  std::cout << "[main] Done" << std::endl;
  
  return 0;
}

#include <GetPot>
#include <Eigen/Dense>
#include "mesh.hpp"
#include "mesh_builder.hpp"
#include "vtu_writer.hpp"

#include <string>


int main (int argc, char** argv) {

using namespace HArDCore3D;

GetPot cl(argc, argv);
const std::string mesh_file = cl.follow("Cubic-Cells/RF_fmt/gcube_2x2x2", 2, "-m", "--mesh");
const std::string mesh_type = cl.follow ("RF", 2, "-t","--meshtype");
const std::string mesh_dir = "../../meshes/";
std::string mesh=mesh_dir+mesh_file;

MeshBuilder meshbuilder = MeshBuilder(mesh, mesh_type);
std::unique_ptr<Mesh> mesh_ptr = meshbuilder.build_the_mesh();
  


std::string filename = "HHOmesh.vtu";
VtuWriter plotdata(mesh_ptr.get());
plotdata.write_to_vtu(filename, Eigen::VectorXd::Zero(mesh_ptr->n_vertices()));

  return 0;	
}

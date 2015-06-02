
// this file has the interpolation operator, and the FAC coarsening functions.
// currently FAC coarsening is inefficient, as it runs over all elements instead of
// using just local. That was done initially because there isn't a local_elements function
// but only an active_local_elements(). This function should be easy to fix.
// note that all calls to MGElemIDConverter are unnecessary. This object was only useful
// when reparitioning was on, but for good performance it needs to be turned off.

#include <vector>

// Local includes
#include "libmesh/boundary_info.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh_base.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/system.h"
#include "libmesh/threads.h"
#include "libmesh/wrapped_function.h"

using namespace libMesh;

#include <fstream>
#define MG_TOOL_PRINT_STATEMENTS_ACTIVE 0

// here this structure holds the element neighbors of dofs

const unsigned int max_neighbor_size = 8;
unsigned int current_interp_count = 0;


// this function is just for debugging

void w(unsigned int i) {

if (MG_TOOL_PRINT_STATEMENTS_ACTIVE)
std::cout << "PrintMGTool: " << i << std::endl << std::flush;
}

unsigned int MGCountLevelsFAC(MeshBase & mesh)
{
 ConstElemRange range
  (mesh.active_elements_begin(),
  mesh.active_elements_end());
unsigned int count = 1;
unsigned int max_levels = 1;
      // Iterate over the elements in the range
      for (ConstElemRange::const_iterator elem_it=range.begin(); elem_it != range.end(); ++elem_it)
        {
          count = 1;
          const Elem* elem = *elem_it;
          const Elem* parent = elem;
          while (parent->parent())
          {
            parent = parent->parent();
            count++;
          }
// PetscPrintf(PETSC_COMM_SELF, "(Elem=%D,level=%D,processor=%D)\n", elem->id(), count, elem->processor_id());
          if (max_levels < count)
            max_levels = count;
        }

return max_levels;
}


void build_interpolation(EquationSystems & es_fine, EquationSystems & es_coarse, std::string system_name, PetscMatrix<Number>& Interpolation, PetscVector<Number> & fine_level, PetscVector<Number> & coarse_level)
{

const double zero_threshold_mg_tool = 1e-30;

/* std::string quick_out = "processor_output_";
std::stringstream stringstr;
stringstr << es_fine.get_system(system_name).processor_id();
std::string processor_number = stringstr.str();


std::ofstream fout(quick_out + processor_number);
*/

unsigned int skip_count = 0;
unsigned int zero_count = 0;
unsigned int total_count = 0;

unsigned int processor_id = es_fine.get_system(system_name).processor_id();

std::stringstream stringstr;
stringstr << es_fine.get_system(system_name).processor_id();
std::string processor_number = stringstr.str();

w(1);

 ConstElemRange range
  (es_fine.get_mesh().active_local_elements_begin(),
  es_fine.get_mesh().active_local_elements_end());

  System & system_fine = es_fine.get_system(system_name);
  System & system_coarse = es_coarse.get_system(system_name);

  const unsigned int n_variables = system_coarse.n_vars();
  const unsigned int dim = system_coarse.get_mesh().mesh_dimension();
  const DofMap& dof_map_fine = system_fine.get_dof_map();
  const DofMap& dof_map_coarse = system_coarse.get_dof_map();


// PetscPrintf(PETSC_COMM_SELF, "(%D, %D) processor: %D\n", dof_map_fine.first_dof(),dof_map_fine.last_dof(), processor_id);

  std::vector<unsigned int> nz;
  std::vector<unsigned int> nnz;
  nz.resize(fine_level.local_size());
  nnz.resize(fine_level.local_size());
  std::vector<unsigned int> coarse_dof_indices, fine_dof_indices, coarse_dof_indices_2, fine_dof_indices_2;

  for (unsigned int var=0; var<n_variables; var++)
    {
      const Variable& variable = dof_map_coarse.variable(var);


      // Iterate over the elements in the range
      for (ConstElemRange::const_iterator elem_it=range.begin(); elem_it != range.end(); ++elem_it)
        {

          bool element_is_coarse = 0;

          const Elem* elem = *elem_it;
          const Elem* elem_parent;
          const Elem* elem_coarse;
          if (elem->parent())
          elem_parent = elem->parent();
          else
          elem_parent = elem;

          elem_coarse = es_coarse.get_mesh().elem(elem_parent->id());

          if (!elem_coarse->active()) // if element not active on coarse mesh, then elem is
                                        // not a new refinement. So element_is_coarse = 1
          {
          element_is_coarse = 1;
          elem_coarse = es_coarse.get_mesh().elem(elem->id());
          }

          dof_map_fine.dof_indices (elem_parent, coarse_dof_indices_2, var); // this is a parent
                                                        // if elem had a parent

          // Per-subdomain variables don't need to be projected on
          // elements where they're not active
          if (!variable.active_on_subdomain(elem->subdomain_id()))
       continue;

         dof_map_fine.dof_indices (elem, fine_dof_indices, var);
         dof_map_coarse.dof_indices( elem_coarse, coarse_dof_indices, var); // parent or element itself
                                        // parent if elem is a refinement, elem itself otherwise

         const unsigned int coarse_n_dofs =
         libmesh_cast_int<unsigned int>(coarse_dof_indices.size());

         const unsigned int fine_n_dofs =
         libmesh_cast_int<unsigned int>(fine_dof_indices.size());




w(2);
         if (!element_is_coarse)
         {
       // This section adds the values into the n_z, nn_z vectors

w(3);

       for (unsigned int j = 0; j < fine_n_dofs; j++)  // continue if on this processor
       if (fine_dof_indices[j] <= dof_map_fine.last_dof() && fine_dof_indices[j] >= dof_map_fine.first_dof())
         {
total_count++;
            bool is_not_coarse_node = 1;
            unsigned int coarse_index = 0;

                // check if this fine dof is a coarse node
w(4);
            for (unsigned int k = 0; k < coarse_n_dofs; k++)
            if (coarse_dof_indices_2[k] == fine_dof_indices[j])
             { is_not_coarse_node = 0; coarse_index = k; break;}

w(5);
            // find out how many coarse_n_dofs are run on this same processor
            const unsigned int fine_position = fine_dof_indices[j] - dof_map_fine.first_dof();
            if (is_not_coarse_node) {
            unsigned int new_dofs_nnz = 0;
            unsigned int new_dofs_nz = elem_coarse->n_nodes();
            for (unsigned int k = 0; k < coarse_n_dofs; k++)
              if (dof_map_coarse.first_dof() <= coarse_dof_indices[k] && coarse_dof_indices[k] <= dof_map_coarse.last_dof())
                { new_dofs_nnz++; new_dofs_nz--; }

w(6);
                  nnz[ fine_position ] = new_dofs_nnz;
                   nz[ fine_position ] = new_dofs_nz;
                   if (new_dofs_nnz == 0 && new_dofs_nz == 0)
		PetscPrintf(PETSC_COMM_WORLD, "Warning! Interpolation Matrix Created an empty row!\n");
               }
               else
               {
                 if (dof_map_coarse.first_dof() <= coarse_dof_indices[coarse_index] && coarse_dof_indices[coarse_index] <= dof_map_coarse.last_dof()) // if it is on processor
                 {
                  nnz[ fine_position ] = 1;
                   nz[ fine_position ] = 0;
                 }
                 else
                 {
                 nnz[ fine_position ] = 0;
                  nz[ fine_position ] = 1;
                 }
               }
// could use FEInterface::shape to check how many 0s here
          }  // if fine_dof is on processor
          else
          skip_count++;
      }
      else // else (elem_coarse->active())
      {
     w(7);         // the finer mesh. But it is accessing an old element

          for (unsigned int j = 0; j < fine_n_dofs; j++)
          {
              if (fine_dof_indices[j] <= dof_map_fine.last_dof() && fine_dof_indices[j] >= dof_map_fine.first_dof())
	      {
	      total_count++;

              if (coarse_dof_indices[j] <= dof_map_coarse.last_dof() && coarse_dof_indices[j] >= dof_map_coarse.first_dof())
{
              nnz[ fine_dof_indices[j] - dof_map_fine.first_dof() ] = 1;
               nz[ fine_dof_indices[j] - dof_map_fine.first_dof() ] = 0;
}
              else
{
             nnz[ fine_dof_indices[j] - dof_map_fine.first_dof() ] = 0;
              nz[ fine_dof_indices[j] - dof_map_fine.first_dof() ] = 1;
}
	      }
else skip_count++; // PetscPrintf(PETSC_COMM_WORLD, "another skip\n");
          }
      }



 }}

PetscPrintf(PETSC_COMM_WORLD, "\n\n\nTotal Count: %D, skip count: %D\n\n\n", total_count, skip_count);
w(10);

for (unsigned int i = 0; i < nnz.size(); i++)
{
if (nz[i] > coarse_level.size() - coarse_level.local_size() )
nz[i] = coarse_level.size() - coarse_level.local_size();
if ( nnz[i] > coarse_level.local_size())
 nnz[i] = coarse_level.local_size();

if (nz[i] == 0 && nnz[i] == 0)
zero_count++;
}

PetscPrintf(PETSC_COMM_SELF, "Zeroes local: %D, processor: %D\n", zero_count, processor_id);

std::stringstream stringstr2;
stringstr2 << current_interp_count;
std::string interp_count = stringstr2.str();

std::string fout_1 = "before_init";
std::ofstream fout(fout_1 + processor_number + "_" + interp_count);
fout << coarse_level.size() << "coarse level\n";
for (unsigned int i = 0; i  < nnz.size(); i++)
fout << nnz[i] << ", " << nz[i] << std::endl;
fout.close();
current_interp_count++;

w(11);

Interpolation.init(fine_level.size(), coarse_level.size(), fine_level.local_size(), coarse_level.local_size(), nnz, nz);

w(12);

// Now we add in the matrix values


  for (unsigned int var=0; var<n_variables; var++)
    {
      const Variable& variable = dof_map_fine.variable(var);

      const FEType& base_fe_type = variable.type();


      if (base_fe_type.family == SCALAR)
        continue;


      // Iterate over the elements in the range
      for (ConstElemRange::const_iterator elem_it=range.begin(); elem_it != range.end(); ++elem_it)
        {
w(13);
          bool element_is_coarse = 0;

          const Elem* elem = *elem_it;
          const Elem* elem_parent;
          const Elem* elem_coarse;
          if (elem->parent())
          elem_parent = elem->parent();
	  else
	  elem_parent = elem;

          elem_coarse = es_coarse.get_mesh().elem(elem_parent->id());

          if (!elem_coarse->active()) // if element not active on coarse mesh, then elem is
					// not a new refinement. So element_is_coarse = 1
          {
          element_is_coarse = 1;
          elem_coarse = es_coarse.get_mesh().elem(elem->id());
          }

          dof_map_fine.dof_indices (elem_parent, coarse_dof_indices_2, var); // this is a parent
							// if elem had a parent

          // Per-subdomain variables don't need to be projected on
          // elements where they're not active
          if (!variable.active_on_subdomain(elem->subdomain_id()))
       continue;

         dof_map_fine.dof_indices (elem, fine_dof_indices, var);
         dof_map_coarse.dof_indices( elem_coarse, coarse_dof_indices, var); // parent or element itself
					// parent if elem is a refinement, elem itself otherwise

         const unsigned int coarse_n_dofs =
         libmesh_cast_int<unsigned int>(coarse_dof_indices.size());

         const unsigned int fine_n_dofs =
         libmesh_cast_int<unsigned int>(fine_dof_indices.size());




w(14);
         if (!element_is_coarse) // if elem is a refined element compared to coarse mesh
         {

       for (unsigned int j = 0; j < fine_n_dofs; j++)
         {
w(15);
            // find out how many coarse_n_dofs are run on this same processor
           const dof_id_type fine_global_dof = fine_dof_indices[j];




   const Point point = FEInterface::inverse_map (dim, base_fe_type, elem_coarse, elem->point(j));

 // fine_dof_indices[j], coarse_dof_indices[k], enter them right into the matrix

            for (unsigned int k = 0; k < coarse_n_dofs; k++)
            {
w(17);
              const dof_id_type coarse_global_dof = coarse_dof_indices[k];
             Real value = FEInterface::shape(dim, base_fe_type, elem_coarse, k, point);
             if (value > zero_threshold_mg_tool && fine_global_dof <= dof_map_fine.last_dof() && fine_global_dof >= dof_map_fine.first_dof())
             Interpolation.set(fine_global_dof, coarse_global_dof, value);
            }
      }
      }
      else
      {
w(18);
          for (unsigned int j = 0; j < fine_n_dofs; j++)
          {
           const dof_id_type fine_global_dof = fine_dof_indices[j];
           const dof_id_type coarse_global_dof = coarse_dof_indices[j];

              if (fine_global_dof <= dof_map_fine.last_dof() && fine_global_dof >= dof_map_fine.first_dof())
      {
// PetscPrintf(PETSC_COMM_SELF, "(%D, %D)\n", fine_global_dof, coarse_global_dof);
Interpolation.set(fine_global_dof, coarse_global_dof, 1.);
      }
            }

w(19);

          }
      }



 }

Interpolation.close();

}



// end of function


void flag_elements_FAC(MeshBase & _mesh)
{


// first find maximum level

  MeshBase::element_iterator e_it = _mesh.elements_begin();
  const MeshBase::element_iterator e_end = _mesh.elements_end();

  unsigned int max_level = 0;
  unsigned int total_elem = 0;

  for (; e_it != e_end; ++e_it) // find maximum level
  {
  total_elem++;
  unsigned int level_count = 0;
  Elem* elem = *e_it;
  Elem* tp = elem;

 while (tp->parent() != NULL)
    {
    tp = tp->parent();


    level_count++;
    }

  if (level_count > max_level)
   max_level = level_count;
  }
  e_it = _mesh.active_elements_begin();
  const MeshBase::element_iterator e_end_2 = _mesh.active_elements_end();

  for (; e_it != e_end_2; ++e_it)  // refine just the maximum level
  {
  Elem* elem = *e_it;

  Elem* parent = elem->parent();

  unsigned int level_count = 0;
  Elem* tp = elem;

 while (tp->parent() != NULL)
    {
    tp = tp->parent();
    level_count++;
    }

  if (parent && level_count == max_level)
  elem->set_refinement_flag(Elem::COARSEN);
  }

e_it = _mesh.elements_begin();

}

void flag_elements_uniformly(MeshBase & _mesh)
{
  MeshBase::element_iterator e_it = _mesh.active_elements_begin();
  const MeshBase::element_iterator e_end = _mesh.active_elements_end();
  for (; e_it != e_end; ++e_it)
  {
  Elem* elem = *e_it;

  Elem* parent = elem->parent();

  if (parent)
  elem->set_refinement_flag(Elem::COARSEN);

  if (elem->parent())
  elem->parent()->set_refinement_flag(Elem::COARSEN_INACTIVE);

  }

}

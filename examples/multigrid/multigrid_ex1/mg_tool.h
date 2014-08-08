


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

#include <fstream>

// here this structure holds the element neighbors of dofs

const unsigned int max_neighbor_size = 8;

struct neighbors {
const Elem* neighbors_ [max_neighbor_size];
unsigned int n_neighbors;
unsigned int var;
};

// this function is just for debugging

void w(unsigned int i) { std::cout << "PrintMGTool: " << i << std::endl << std::flush; }


void build_interpolation(EquationSystems & es_fine, EquationSystems & es_coarse, std::string system_name, PetscMatrix<Number>& Interpolation, PetscVector<Number> & fine_level, PetscVector<Number> & coarse_level)
{


const double zero_threshold_mg_tool = 0;

/* std::string quick_out = "processor_output_";
std::stringstream stringstr;
stringstr << es_fine.get_system(system_name).processor_id();
std::string processor_number = stringstr.str();

std::ofstream fout(quick_out + processor_number);
*/

std::stringstream stringstr;
stringstr << es_fine.get_system(system_name).processor_id();
std::string processor_number = stringstr.str();

w(1);

 ConstElemRange range
  (es_coarse.get_mesh().active_local_elements_begin(),
  es_coarse.get_mesh().active_local_elements_end());

  System & system_fine = es_fine.get_system(system_name);
  System & system_coarse = es_coarse.get_system(system_name);

  const unsigned int n_variables = system_coarse.n_vars();
  const unsigned int dim = system_coarse.get_mesh().mesh_dimension();
  const DofMap& dof_map_fine = system_fine.get_dof_map();
  const DofMap& dof_map_coarse = system_coarse.get_dof_map();


  std::vector< neighbors > dof_elem;
  dof_elem.resize(fine_level.local_size());

  for (unsigned int i = 0; i < fine_level.local_size(); i++)
  {
  for (unsigned int j = 0; j < max_neighbor_size; j++)
  dof_elem[i].neighbors_[j] = NULL;
  dof_elem[i].n_neighbors = 0;
  dof_elem[i].var = 0;
  }
  const unsigned int max_nodes_per_elem = 8;

  const unsigned int max_dofs_elem = max_neighbor_size*max_nodes_per_elem;
  unsigned int interacting_dofs_nz = 0;
  unsigned int interacting_dofs_nnz = 0;

  std::vector<unsigned int> interacting_dofs;
  

  interacting_dofs.resize(max_dofs_elem);
  unsigned int n_interacting_dofs = 0;

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
          const Elem* elem = *elem_it;
          const Elem* elem_full = es_fine.get_mesh().elem(elem->id());      

          // check if this element is interesting



          // Per-subdomain variables don't need to be projected on
          // elements where they're not active
          if (!variable.active_on_subdomain(elem->subdomain_id()))
       continue;

         dof_map_coarse.dof_indices (elem, coarse_dof_indices, var); 
         dof_map_fine.dof_indices (elem_full, coarse_dof_indices_2, var);
         const unsigned int coarse_n_dofs =
         libmesh_cast_int<unsigned int>(coarse_dof_indices.size());

         if (!elem_full->active())
         for (unsigned int i = 0; i < elem_full->n_children(); i++)
         {
            Elem * child = elem_full->child(i);
          // Update the DOF indices for this element based on
          // the new mesh
            dof_map_fine.dof_indices (child, fine_dof_indices, var);


            const unsigned int fine_n_dofs =
            libmesh_cast_int<unsigned int>(fine_dof_indices.size());

      
 
       // This section adds the values into the n_z, nn_z vectors

       for (unsigned int j = 0; j < fine_n_dofs; j++)  // continue if on this processor
       if (fine_dof_indices[j] <= dof_map_fine.last_dof() && fine_dof_indices[j] >= dof_map_fine.first_dof())
         {
            bool is_not_coarse_node = 1;
            unsigned int coarse_index = 0;

                // check if this fine dof is a coarse node

            for (unsigned int k = 0; k < coarse_n_dofs; k++)
            if (coarse_dof_indices_2[k] == fine_dof_indices[j])
             { is_not_coarse_node = 0; coarse_index = k;}


            // find out how many coarse_n_dofs are run on this same processor
            const unsigned int fine_position = fine_dof_indices[j] - dof_map_fine.first_dof();
            if (is_not_coarse_node) {
           /* for (unsigned int k = 0; k < coarse_n_dofs; k++)
              if (dof_map_coarse.first_dof() <= coarse_dof_indices[k] && coarse_dof_indices[k] <= dof_map_coarse.last_dof())
                { new_dofs_nnz++; new_dofs_nz--; } 


                  nnz[ fine_position ] += new_dofs_nnz;
                   nz[ fine_position ] += new_dofs_nz;
           */

              // here we record the elements corresponding to each dof that is not coarse

              dof_elem[fine_position].neighbors_[dof_elem[fine_position].n_neighbors++] = elem_full;
              dof_elem[fine_position].var = var; 

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
      }
      else // else (elem->active())
      {
          dof_map_fine.dof_indices(elem_full, fine_dof_indices, var); // note that this technically is not 'fine', but simply on
                                                                      // the finer mesh. But it is accessing an old element
          const unsigned int fine_n_dofs = 
          libmesh_cast_int<unsigned int>(fine_dof_indices.size());

          for (unsigned int j = 0; j < fine_n_dofs; j++)
          {
              if (fine_dof_indices[j] <= dof_map_fine.last_dof() && fine_dof_indices[j] >= dof_map_fine.first_dof()) {
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
          }
      }



 }}

w(10);

  for (unsigned int i = 0; i < nnz.size(); i++)
  {
    n_interacting_dofs = 0;
    interacting_dofs_nz = 0;
    interacting_dofs_nnz = 0;

    for (unsigned int j = 0; j < dof_elem[i].n_neighbors; j++) // note that no coarse nodes will be counted here
    {
     const Elem * elem = dof_elem[i].neighbors_[j];
     const Elem * elem_coarse = es_coarse.get_mesh().elem(elem->id());
         dof_map_fine.dof_indices (elem, coarse_dof_indices_2, dof_elem[i].var);
         dof_map_coarse.dof_indices (elem_coarse, coarse_dof_indices, dof_elem[i].var);

         for (unsigned int k = 0; k < coarse_dof_indices_2.size(); k++)
         {
            bool should_continue = 1;
            for (unsigned int l = 0; l < n_interacting_dofs; l++)
               if (interacting_dofs[l] == coarse_dof_indices_2[k])
                 should_continue = 0;
            if (should_continue)
            {
              if (coarse_dof_indices[k] <= dof_map_coarse.last_dof() && coarse_dof_indices[k] >= dof_map_coarse.first_dof())
                interacting_dofs_nnz++;
              else
                interacting_dofs_nz++;

              interacting_dofs[n_interacting_dofs++] = coarse_dof_indices_2[k];
            }
         }
    }
   // now I've built the array of indices
  nz[i] += interacting_dofs_nz;
  nnz[i] += interacting_dofs_nnz;

  }


for (unsigned int i = 0; i < nnz.size(); i++)
{
if (nz[i] > coarse_level.size() - coarse_level.local_size() )
nz[i] = coarse_level.size() - coarse_level.local_size();
if ( nnz[i] > coarse_level.local_size())
 nnz[i] = coarse_level.local_size();
}

std::string fout_1 = "before_init";
std::ofstream fout(fout_1 + processor_number);
fout << coarse_level.size() << "coarse level\n";
for (unsigned int i = 0; i  < nnz.size(); i++)
fout << nnz[i] << ", " << nz[i] << std::endl;
fout.close();

w(2);

Interpolation.init(fine_level.size(), coarse_level.size(), fine_level.local_size(), coarse_level.local_size(), nnz, nz);

w(3);

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
          const Elem* elem = *elem_it;
          const Elem* elem_full = es_fine.get_mesh().elem(elem->id());

          // Per-subdomain variables don't need to be projected on
          // elements where they're not active
          if (!variable.active_on_subdomain(elem->subdomain_id()))
            continue;


         dof_map_coarse.dof_indices (elem, coarse_dof_indices, var);
         const unsigned int coarse_n_dofs =
         libmesh_cast_int<unsigned int>(coarse_dof_indices.size());

         if (!elem_full->active())
         for (unsigned int i = 0; i < elem_full->n_children(); i++)
         {
            Elem * child = elem_full->child(i);
          // Update the DOF indices for this element based on
          // the new mesh
            dof_map_fine.dof_indices (child, fine_dof_indices, var);

            const unsigned int fine_n_dofs =
            libmesh_cast_int<unsigned int>(fine_dof_indices.size());


       for (unsigned int j = 0; j < fine_n_dofs; j++)
         {

            // find out how many coarse_n_dofs are run on this same processor
           const dof_id_type fine_global_dof = fine_dof_indices[j];




   const Point point = FEInterface::inverse_map (dim, base_fe_type, elem_full, child->point(j));

 // fine_dof_indices[j], coarse_dof_indices[k], enter them right into the matrix

            for (unsigned int k = 0; k < coarse_n_dofs; k++)
            {    
              const dof_id_type coarse_global_dof = coarse_dof_indices[k];  
             Real value = FEInterface::shape(dim, base_fe_type, elem_full, k, point);
             if (value > zero_threshold_mg_tool && fine_global_dof <= dof_map_fine.last_dof() && fine_global_dof >= dof_map_fine.first_dof())
             Interpolation.set(fine_global_dof, coarse_global_dof, value);
            }
      }
      }
      else
      {
          dof_map_fine.dof_indices(elem_full, fine_dof_indices, var); // note that this technically is not 'fine', but simply on
                                                                      // the finer mesh. But it is accessing an old element
          const unsigned int fine_n_dofs =
          libmesh_cast_int<unsigned int>(fine_dof_indices.size());

          for (unsigned int j = 0; j < fine_n_dofs; j++)
          {
           const dof_id_type fine_global_dof = fine_dof_indices[j];
           const dof_id_type coarse_global_dof = coarse_dof_indices[j];

              if (fine_global_dof <= dof_map_fine.last_dof() && fine_global_dof >= dof_map_fine.first_dof())
              Interpolation.set(fine_global_dof, coarse_global_dof, 1);
            }



          }
      }



 }

Interpolation.close();

}

      
 
// end of function




void flag_elements_FAC(MeshBase & _mesh)
{


// first find maximum level

  MeshBase::element_iterator e_it = _mesh.active_elements_begin();
  const MeshBase::element_iterator e_end = _mesh.active_elements_end();

  unsigned int max_level = 0;

  for (; e_it != e_end; ++e_it)
  {
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

  for (; e_it != e_end; ++e_it)
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

  if (elem->parent())
  elem->parent()->set_refinement_flag(Elem::COARSEN_INACTIVE);

  }

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



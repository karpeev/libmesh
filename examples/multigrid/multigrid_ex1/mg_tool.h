


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
#define MG_TOOL_PRINT_STATEMENTS_ACTIVE 0

// here this structure holds the element neighbors of dofs

const unsigned int max_neighbor_size = 8;


// this function is just for debugging

void w(unsigned int i) { 

if (MG_TOOL_PRINT_STATEMENTS_ACTIVE)
std::cout << "PrintMGTool: " << i << std::endl << std::flush; 
}

class MGElemIDConverter {

private:
std::vector<unsigned int> id_on_fine;
std::vector<const Elem*>  elem_fine;
unsigned int n; // current position

public:
void resize(unsigned int);
bool add_elem(const Elem*);
unsigned int c_to_f(unsigned int);
};

unsigned int MGCountLevelsFAC(MeshBase & mesh)
{
 ConstElemRange range
  (mesh.active_local_elements_begin(),
  mesh.active_local_elements_end());
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
          if (max_levels < count)
            max_levels = count;
        }

return max_levels;
}


void build_interpolation(EquationSystems & es_fine, EquationSystems & es_coarse, std::string system_name, PetscMatrix<Number>& Interpolation, PetscVector<Number> & fine_level, PetscVector<Number> & coarse_level, MGElemIDConverter& id_converter)
{

const double zero_threshold_mg_tool = 1e-8;

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
          const Elem* elem_full = es_fine.get_mesh().elem(id_converter.c_to_f(elem->id()));      
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
            unsigned int new_dofs_nnz = 0;
            unsigned int new_dofs_nz = elem_full->n_nodes();
            for (unsigned int k = 0; k < coarse_n_dofs; k++)
              if (dof_map_coarse.first_dof() <= coarse_dof_indices[k] && coarse_dof_indices[k] <= dof_map_coarse.last_dof())
                { new_dofs_nnz++; new_dofs_nz--; } 


                  nnz[ fine_position ] = new_dofs_nnz;
                   nz[ fine_position ] = new_dofs_nz;

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
          const Elem* elem_full = es_fine.get_mesh().elem(id_converter.c_to_f(elem->id()));

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

unsigned int MGElemIDConverter::c_to_f(unsigned int  id)
{
if (id < id_on_fine.size())
return id_on_fine[id];
else {
std::cout << "Warning: MGElemIDConverter::c_to_f(unsigned int) recieved too large an id: " << id << ", ";
std::cout << "while max is " << id_on_fine.size() << std::endl; }

return 0; // if nothing showed up, default to 0
}

void MGElemIDConverter::resize(unsigned int new_size)
{
id_on_fine.resize(new_size);
elem_fine.resize(new_size);

n = 0;
}

bool MGElemIDConverter::add_elem(const Elem* new_elem) // returns 1 if taken, 0 if not taken
{
for (unsigned int i = 0; i < n; i++)
if (elem_fine[i] == new_elem)
return 0;

if (n >= id_on_fine.size())
return 0;

id_on_fine[n] = new_elem->id();
elem_fine[n] = new_elem;
n++;

return 1;
}

void flag_elements_FAC(MeshBase & _mesh, MGElemIDConverter & id_converter)
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

  id_converter.resize(total_elem);

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

for (; e_it != e_end; ++e_it)
{
Elem* elem = *e_it;
if (elem->refinement_flag() != Elem::COARSEN)
id_converter.add_elem(elem);
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



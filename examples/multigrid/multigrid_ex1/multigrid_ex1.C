// #include "limbesh/system_projection.h"

#include <vector>
#include "libmesh/petsc_matrix.h"
#include <iostream>
#include "libmesh/petsc_vector.h"
#include <algorithm>
#include <math.h>
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/gmv_io.h"
#include "libmesh/gnuplot_io.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/distributed_vector.h"  
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/perf_log.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/system_subset_by_subdomain.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/getpot.h"
#include <iostream>
#include <algorithm>
#include <math.h>
#include "libmesh/vtk_io.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"

// headers to get complex values working. From Stoke's Example

#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"

// adaptivity headers

#include "libmesh/tecplot_io.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/error_vector.h"
#include "libmesh/exact_error_estimator.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/patch_recovery_error_estimator.h"
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/hp_coarsentest.h"
#include "libmesh/hp_singular.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/perf_log.h"
#include "libmesh/getpot.h"
#include "libmesh/exact_solution.h"
#include "libmesh/string_to_enum.h"
#include "mg_tool.h"

using namespace libMesh;

void assemble_helmholtz(EquationSystems& es,
                      const std::string& system_name);

// assemble just the matrix for use in rediscretization. We don't need to make RHS there

void assemble_helmholtz_matrix(EquationSystems& es,
                      const std::string& system_name, PetscMatrix<Number>*);

// error checking function

void e(unsigned int i) { std::cout << "PrintStatement: " << i << '\n' << std::flush; }

Real exact_solution (const Real x,
                     const Real y,
                     const Real z);
Real exact_solution_I (const Real x,
                      const Real y,
                      const Real z);

Number exact_solution(const Point& p, const Parameters&,const std::string&,const std::string&);
Number exact_solution_I(const Point& p, const Parameters&,const std::string&,const std::string&);

bool singularity = true;
bool print_matrix = false;

Real gamma_R = 0.0;
Real gamma_I =  0.0;

unsigned int dim = 3;

int main (int argc, char** argv)
{
  LibMeshInit init (argc, argv);

 #ifndef LIBMESH_ENABLE_AMR
    libmesh_example_assert(false, "--enable-amr");
  #else
  
    GetPot input_file("multigrid_ex1.in");
    const unsigned int n_levels_coarsen = input_file("n_levels_coarsen", 3);
    			gamma_R       = input_file("gamma_R", 1.0);
			gamma_I	      = input_file("gamma_I", 0.0);
    const unsigned int max_r_steps    = input_file("max_r_steps", 3);
    const unsigned int max_r_level    = input_file("max_r_level", 3);
    const Real refine_percentage      = input_file("refine_percentage", 0.5);
    const Real coarsen_percentage     = input_file("coarsen_percentage", 0.5);
    const unsigned int uniform_refine = input_file("uniform_refine",0);
    const std::string refine_type     = input_file("refinement_type", "h");
    const std::string approx_type     = input_file("approx_type", "LAGRANGE");
    const unsigned int approx_order   = input_file("approx_order", 1);
    const std::string element_type    = input_file("element_type", "tensor");
    const int extra_error_quadrature  = input_file("extra_error_quadrature", 0);
    const int max_linear_iterations   = input_file("max_linear_iterations", 5000);
    const bool output_intermediate    = input_file("output_intermediate", false);
    print_matrix                      = input_file("print_matrix", false);
    const unsigned int x_range        = input_file("x_range", 2);
    const unsigned int y_range        = input_file("y_range", 2);
    const unsigned int z_range        = input_file("z_range", 2);
    const Real x_size                 = input_file("x_size", 1);
    const Real y_size                 = input_file("y_size", 1);
    const Real z_size                 = input_file("z_size", 1);


//                               ADAPTIVELY REFINE AND SOLVE
// ===================================================================================

    dim = input_file("dimension", 3);
    const std::string indicator_type = input_file("indicator_type", "kelly");
    singularity = input_file("singularity", true);
  
    libmesh_example_assert(dim <= LIBMESH_DIM, "3D support");
  
    std::string approx_name = "";
    if (element_type == "tensor")
      approx_name += "bi";
    if (approx_order == 1)
      approx_name += "linear";
    else if (approx_order == 2)
      approx_name += "quadratic";
    else if (approx_order == 3)
      approx_name += "cubic";
    else if (approx_order == 4)
      approx_name += "quartic";
  
    std::string output_file = approx_name;
    output_file += "_";
    output_file += refine_type;
    if (uniform_refine == 0)
      output_file += "_adaptive.m";
    else
      output_file += "_uniform.m";
  
    std::ofstream out (output_file.c_str());
    out << "% dofs     L2-error  " << std::endl;
    out << "e = [" << std::endl;
  

  std::cout << "Running " << argv[0];

  for (int i=1; i<argc; i++)
    std::cout << " " << argv[i];

  std::cout << std::endl << std::endl;

  libmesh_example_assert(3 <= LIBMESH_DIM, "3D support");
  Mesh mesh(init.comm());
  MeshTools::Generation::build_cube (mesh,x_range, y_range, z_range,0., x_size,0.,y_size, 0.,z_size,HEX8);

  std::cout << x_range << ", " << y_range << ", " << z_range << "\n";

    if (element_type == "simplex")
      MeshTools::Modification::all_tri(mesh);

  if (approx_order > 1 || refine_type != "h")
      mesh.all_second_order();
  
    MeshRefinement mesh_refinement(mesh);
    mesh_refinement.refine_fraction() = refine_percentage;
    mesh_refinement.coarsen_fraction() = coarsen_percentage;
    mesh_refinement.max_h_level() = max_r_level;

  mesh.print_info();

  EquationSystems equation_systems (mesh);

LinearImplicitSystem& system = equation_systems.add_system<LinearImplicitSystem> ("Helmholtz");

  system.add_variable("u_R", static_cast<Order>(approx_order),
                        Utility::string_to_enum<FEFamily>(approx_type));
  system.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));

  equation_systems.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  equation_systems.init();

   equation_systems.parameters.set<unsigned int>("linear solver maximum iterations")
      = max_linear_iterations;
  
    equation_systems.parameters.set<Real>("linear solver tolerance") =
      std::pow(TOLERANCE, 2.5);

  equation_systems.print_info();

  ExactSolution exact_sol_I(equation_systems);
  ExactSolution exact_sol(equation_systems);
    exact_sol_I.attach_exact_value(exact_solution_I);
    exact_sol.attach_exact_value(exact_solution);
  
    exact_sol_I.extra_quadrature_order(extra_error_quadrature);
    exact_sol.extra_quadrature_order(extra_error_quadrature);
  
    for (unsigned int r_step=0; r_step<max_r_steps; r_step++)
      {
        std::cout << "Beginning Solve " << r_step << std::endl;
  
        system.solve();
  
        std::cout << "System has: " << equation_systems.n_active_dofs()
                  << " degrees of freedom."
                  << std::endl;
  
        std::cout << "Linear solver converged at step: "
                  << system.n_linear_iterations()
                  << ", final residual: "
                  << system.final_linear_residual()
                  << std::endl;
  
  #ifdef LIBMESH_HAVE_EXODUS_API
        if (output_intermediate)
          {
            std::ostringstream outfile;
            outfile << "lshaped_" << r_step << ".e";
            ExodusII_IO (mesh).write_equation_systems (outfile.str(),
                                                 equation_systems);
          }
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
  
        exact_sol.compute_error("Helmholtz", "u_R");
        exact_sol_I.compute_error("Helmholtz", "u_C");  


     Real l2_error_I = exact_sol_I.l2_error("Helmholtz", "u_C");
     Real l2_error_R = exact_sol.l2_error("Helmholtz", "u_R");

     Real l2_total_error = sqrt(l2_error_I*l2_error_I + l2_error_R*l2_error_R);

     std::cout << "L2-error is " << l2_total_error << std::endl;

        out << equation_systems.n_active_dofs() << " "
            << l2_total_error << " "; 
        if (r_step+1 != max_r_steps)
          {
            std::cout << "  Refining the mesh..." << std::endl;
  
            if (uniform_refine == 0)
              {
  
                ErrorVector error;
  
                if (indicator_type == "exact")
                  {
                    ExactErrorEstimator error_estimator;
  
                    error_estimator.attach_exact_value(exact_solution);
  
                    error_estimator.estimate_error (system, error);
                  }
                else if (indicator_type == "patch")
                  {
                    PatchRecoveryErrorEstimator error_estimator;
  
                    error_estimator.estimate_error (system, error);
                  }
                else if (indicator_type == "uniform")
                  {
                    UniformRefinementEstimator error_estimator;
  
                    error_estimator.estimate_error (system, error);
                  }
                else
                  {
                    libmesh_assert_equal_to (indicator_type, "kelly");
  
                    KellyErrorEstimator error_estimator;
		    error_estimator.error_norm = L2;
  
                    error_estimator.estimate_error (system, error);
                  }
  
                std::ostringstream ss;
  	      ss << r_step;
  #ifdef LIBMESH_HAVE_EXODUS_API
  	      std::string error_output = "error_"+ss.str()+".e";
  #else
  	      std::string error_output = "error_"+ss.str()+".gmv";
  #endif
                error.plot_error( error_output, mesh );
  
                mesh_refinement.flag_elements_by_error_fraction (error);
  
                if (refine_type == "p")
                  mesh_refinement.switch_h_to_p_refinement();
                if (refine_type == "matchedhp")
                  mesh_refinement.add_p_to_h_refinement();
                if (refine_type == "hp")
                  {
                    HPCoarsenTest hpselector;
                    hpselector.select_refinement(system);
                  }
                if (refine_type == "singularhp")
                  {
                    libmesh_assert (singularity);
                    HPSingularity hpselector;
                    hpselector.singular_points.push_back(Point());
                    hpselector.select_refinement(system);
                  }
  
                mesh_refinement.refine_and_coarsen_elements();
              }
  
            else if (uniform_refine == 1)
              {
                if (refine_type == "h" || refine_type == "hp" ||
                    refine_type == "matchedhp")
                  mesh_refinement.uniformly_refine(1);
                if (refine_type == "p" || refine_type == "hp" ||
                    refine_type == "matchedhp")
                  mesh_refinement.uniformly_p_refine(1);
              }
  
            equation_systems.reinit ();
          }
      }


// 				END ADAPTIVELY REFINE AND SOLVE
// ===================================================================================
//      			BEGIN MULTIGRID LEVEL BUILDING




  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems ("lshaped.e",
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
  
    out << "];" << std::endl;
    out << "hold on" << std::endl;
    out << "plot(e(:,1), e(:,2), 'bo-');" << std::endl;
    out << "plot(e(:,1), e(:,3), 'ro-');" << std::endl;
    out << "xlabel('dofs');" << std::endl;
    out << "title('" << approx_name << " elements');" << std::endl;
    out << "legend('L2-error');" << std::endl;
  #endif // #ifndef LIBMESH_ENABLE_AMR

  // Here I begin experimentation

e(1);
  mesh.skip_partitioning(1); // prevents reparitioning upon each coarsening

  Mesh mesh_2(mesh);

				// equation_system_2, and equation_system_3 will be looped over

 EquationSystems equation_systems_2 (mesh_2);

  LinearImplicitSystem& system_2 = equation_systems_2.add_system<LinearImplicitSystem> ("Helmholtz");
  system_2.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
  system_2.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
  equation_systems_2.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  PetscVector<Number> solution_fine(equation_systems.comm());
  solution_fine.init(*system.solution);
  solution_fine = *system.solution;

  system_2.project_solution_on_reinit();
  system.project_solution_on_reinit();

  equation_systems_2.init();
e(2);

  PetscVector<Number> **level_vector = new PetscVector<Number>* [n_levels_coarsen];
  PetscMatrix<Number> **level_interp = new PetscMatrix<Number>* [n_levels_coarsen];
  PetscMatrix<Number> **level_A      = new PetscMatrix<Number>* [n_levels_coarsen];
  MGElemIDConverter       *id_converter = new MGElemIDConverter [n_levels_coarsen];

  PetscMatrix<Number> **level_restrct = new PetscMatrix<Number>* [n_levels_coarsen];

  for (unsigned int i = 0; i < n_levels_coarsen; i++)
{
  level_vector[i] = new PetscVector<Number>(system_2.comm());
  level_interp[i] = new PetscMatrix<Number>(system_2.comm());
  level_A[i]      = new PetscMatrix<Number>(system_2.comm());
  level_restrct[i] = new PetscMatrix<Number>(system_2.comm());

  unsigned int n_processors = level_A[i]->n_processors();

  level_A[i]->init(n_processors, n_processors, 1, 1, 1, 1);
  level_vector[i]->init(n_processors, 1, 0);

}
e(3);

  PetscVector<Number> * petsc_vec;
  PetscMatrix<Number> * petsc_mat;

  petsc_vec = (PetscVector<Number>*) system.rhs;
  petsc_mat = (PetscMatrix<Number>*) system.matrix;

  level_vector[0]->swap(*petsc_vec);
  system_2.assemble();
  level_A[0]->swap(*petsc_mat);

  *level_vector[0] = solution_fine;

  *system.solution = *level_vector[0];

  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems ("solution_0.e",
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API


 std::cout << level_vector[0]->size() << " size_of_level\n";
 std::cout << system.rhs->size() << " size of rhs\n"; 
 std::cout << level_A[0]->m() << " size_of_mat_m\n";
 std::cout << system.matrix->m() << " size of matrix\n";
  
  #ifdef LIBMESH_HAVE_EXODUS_API
  std::stringstream stringstr;
  stringstr << 0; 
  std::string title_number = stringstr.str();
  std::string title = "lshaped_processor_";
  std::string title_end = ".e";
    ExodusII_IO (mesh_2).write_equation_systems (title+title_number+title_end,
                                         equation_systems_2);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

e(4);

  // nnz on, nz off

 
  MeshRefinement mesh_refinement_2(mesh_2);
  mesh_refinement_2.coarsen_by_parents();
  mesh_refinement_2.clean_refinement_flags();
  flag_elements_FAC(mesh_2, id_converter[0]);
  mesh_refinement_2.coarsen_elements();
  equation_systems_2.reinit();                  // performed coarsening for first round

e(5);
  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh_2).write_equation_systems ("lshaped_1.e",   // Print the coarsened
                                         equation_systems_2);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

  Mesh mesh_3(mesh_2);

EquationSystems equation_systems_3 (mesh_3);

  LinearImplicitSystem& system_3 = equation_systems_3.add_system<LinearImplicitSystem> ("Helmholtz");
  system_3.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
  system_3.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
  equation_systems_3.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  equation_systems_3.init();
  system_3.project_solution_on_reinit();


e(6);

 petsc_vec = (PetscVector<Number>*) system_2.rhs;
 petsc_mat = (PetscMatrix<Number>*) system_2.matrix;

 level_vector[1]->swap(*petsc_vec);
 level_A[1]->swap(*petsc_mat);

 


 assemble_helmholtz_matrix(equation_systems_2, "Helmholtz", level_A[1]);

  std::cout << level_vector[1]->size() << " size_of_level\n";
  std::cout << system_2.rhs->size() << " size of rhs\n";
  std::cout << level_A[1]->m() << " size_of_mat_m\n";
  std::cout << system_2.matrix->m() << " size of matrix\n";

  #ifdef LIBMESH_HAVE_EXODUS_API
  stringstr.clear();
  stringstr << 1; 
  title_number = stringstr.str();
  title = "lshaped_processor_";
    ExodusII_IO (mesh_2).write_equation_systems (title+title_number+title_end,
                                         equation_systems_2);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

  *system_2.solution = *level_vector[1];
  MeshRefinement mesh_refinement_3(mesh_3);
  mesh_refinement_3.coarsen_by_parents();
  flag_elements_FAC(mesh_3, id_converter[1]);
  mesh_refinement_3.coarsen_elements();
  equation_systems_3.reinit();

 petsc_vec = (PetscVector<Number>*) system_3.rhs;
 petsc_mat = (PetscMatrix<Number>*) system_3.matrix;
 level_vector[2]->swap(*petsc_vec);
 level_A[2]->swap(*petsc_mat);

 assemble_helmholtz_matrix(equation_systems_3, "Helmholtz", level_A[2]);


 std::cout << level_vector[2]->size() << " size_of_level\n";
 std::cout << system_3.rhs->size() << " size of rhs\n";
  std::cout << level_A[2]->m() << " size_of_mat_m\n";
 std::cout << system_3.matrix->m() << " size of matrix\n";


e(7);
 build_interpolation(equation_systems, equation_systems_2, "Helmholtz", *level_interp[0], *level_vector[0], *level_vector[1], id_converter[0]);
 level_interp[0]->get_transpose(*level_restrct[0]);
 std::cout << level_interp[0]->m() << ", " << level_interp[0]->n() << " :: ";
 std::cout << level_restrct[0]->m() << ", " << level_restrct[0]->n() << "\n";
 level_restrct[0]->vector_mult(*level_vector[1], *level_vector[0]);
std::cout << system.solution->size() << " size of solution\n";
 *system_2.solution = *level_vector[1];

  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh_2).write_equation_systems ("solution_1.e",
                                         equation_systems_2);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API


/*  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh_2).write_equation_systems ("lshaped_2.e",   // Print the coarsened
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
*/
e(8);
 for (unsigned int i = 2; i < n_levels_coarsen; i++)
 {
e(7+i);

  // assemble_helmholtz(equation_systems_3, "Helmholtz", level_A[i]);

  build_interpolation(equation_systems_2, equation_systems_3, "Helmholtz", *level_interp[i-1], *level_vector[i-1], *level_vector[i], id_converter[i-1]);

  *system_3.solution = *level_vector[i];
 level_interp[i-1]->get_transpose(*level_restrct[i-1]);
e(101);
 level_restrct[i-1]->vector_mult(*level_vector[i], *level_vector[i-1]);
 *system_3.solution = *level_vector[i];

std::string solution_word = "solution_";
std::string dot_e_word = ".e";

  
  stringstr << i;
  std::string title_number = stringstr.str();

  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh_3).write_equation_systems (solution_word + title_number + dot_e_word,
                                         equation_systems_3);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
e(102);

  if (i != n_levels_coarsen-1)
  {
    flag_elements_FAC(mesh_3, id_converter[i]);
    flag_elements_FAC(mesh_2, id_converter[0]);
    mesh_refinement_2.coarsen_elements();
    mesh_refinement_3.coarsen_elements();
    equation_systems_2.reinit();
    equation_systems_3.reinit();
/*
    level_vector[i+1]->init(*system_3.solution);
    *level_vector[i+1] = *system_3.solution;
*/
    petsc_vec = (PetscVector<Number>*) system_3.rhs;
    petsc_mat = (PetscMatrix<Number>*) system_3.matrix;
    level_vector[i+1]->swap(*petsc_vec);
    level_A[i+1]->swap(*petsc_mat);

 assemble_helmholtz_matrix(equation_systems_3, "Helmholtz", level_A[i+1]);

 std::cout << level_vector[i+1]->size() << " size_of_level\n";
 std::cout << system_3.rhs->size() << " size of rhs\n";
 std::cout << level_A[0]->m() << " size_of_mat_m\n";
 std::cout << system_3.matrix->m() << " size of matrix\n";

  }

 }

e(1000);
if (uniform_refine) { // here if uniformly refine, we compare project_vector to interp operator
std::ostringstream str_unif;
str_unif << system_2.processor_id();
std::string unif_name = "uniform_test_project_vector_";

std::ofstream fout_2(unif_name + str_unif.str());

 PetscVector<Number> solution_unif(system_3.comm());
 solution_unif.init(*system_3.solution);
 solution_unif = *system_3.solution;

  MeshRefinement mesh_refinement_4(mesh_3);
  mesh_refinement_4.coarsen_by_parents();
  mesh_refinement_4.uniformly_refine();
  equation_systems_3.reinit();

  system_3.solution->print(fout_2);
  fout_2.close();

  level_interp[n_levels_coarsen-2]->vector_mult(*system_3.solution,solution_unif);
  std::string unif_name_2 = "uniform_test_interp_matrix_";
  std::ofstream fout_3(unif_name_2+str_unif.str());
  system_3.solution->print(fout_3);
  fout_3.close();

  }
e(1001);
for (unsigned int i = 0; i < n_levels_coarsen; i++)
{
 delete level_A[i];
 delete level_vector[i];
 delete level_interp[i];
 delete level_restrct[i];
}
 delete [] level_A; delete [] level_vector; delete [] level_interp; delete [] level_restrct;
e(1002);

/*  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems ("lshaped_interpolated_fine.e",
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
*/


  return 0;
}

void assemble_helmholtz(EquationSystems& es,
                      const std::string& system_name)
{

  libmesh_assert_equal_to (system_name, "Helmholtz");

  const MeshBase& mesh = es.get_mesh();

  const unsigned int dim = mesh.mesh_dimension();

  LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> ("Helmholtz");

  const unsigned int u_R_var = system.variable_number("u_R");
  const unsigned int u_C_var = system.variable_number("u_C");


  const DofMap& dof_map = system.get_dof_map();

  FEType fe_type = dof_map.variable_type(0);

  AutoPtr<FEBase> fe (FEBase::build(dim, fe_type));

  QGauss qrule (dim, FIFTH);

  fe->attach_quadrature_rule (&qrule);

  AutoPtr<FEBase> fe_face (FEBase::build(dim, fe_type));

  QGauss qface(dim-1, FIFTH);

  fe_face->attach_quadrature_rule (&qface);

  const std::vector<Real>& JxW = fe->get_JxW();

  const std::vector<Point>& q_point = fe->get_xyz();

  const std::vector<std::vector<Real> >& phi = fe->get_phi();

  const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
  KRR(Ke), KRC(Ke), KCR(Ke),KCC(Ke);

  DenseSubVector<Number>
  FR(Fe), FC(Fe);


  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_R;
  std::vector<dof_id_type> dof_indices_C;


  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el ; ++el)
    {
      const Elem* elem = *el;

      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_R, u_R_var);
      dof_map.dof_indices (elem, dof_indices_C, u_C_var);

      const unsigned int n_dofs = dof_indices.size();
      const unsigned int n_u_R_dofs = dof_indices_R.size();
      const unsigned int n_u_C_dofs = dof_indices_C.size();

      fe->reinit (elem);

      Ke.resize (n_dofs,
                 n_dofs);

      Fe.resize (n_dofs);

      KRR.reposition(u_R_var*n_u_R_dofs, u_R_var*n_u_R_dofs, n_u_R_dofs, n_u_R_dofs);
      KRC.reposition(u_R_var*n_u_R_dofs, u_C_var*n_u_R_dofs, n_u_R_dofs, n_u_C_dofs);
      KCR.reposition(u_C_var*n_u_C_dofs, u_R_var*n_u_C_dofs, n_u_C_dofs, n_u_R_dofs);
      KCC.reposition(u_C_var*n_u_C_dofs, u_C_var*n_u_C_dofs, n_u_C_dofs, n_u_C_dofs);

      FR.reposition(u_R_var*n_u_R_dofs, n_u_R_dofs);
      FC.reposition(u_C_var*n_u_R_dofs, n_u_C_dofs);

      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {

/*          for (unsigned int i=0; i<phi.size(); i++)
            for (unsigned int j=0; j<phi.size(); j++)
              {
                Ke(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);
              }
*/

	for (unsigned int i = 0; i < n_u_R_dofs; i++)
	  for (unsigned int j = 0; j < n_u_R_dofs; j++)
	    KRR(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);

        for (unsigned int i = 0; i < n_u_R_dofs; i++)
          for (unsigned int j = 0; j < n_u_C_dofs; j++)
            KRC(i,j) +=  -JxW[qp]*gamma_I*phi[i][qp]*phi[j][qp];

        for (unsigned int i = 0; i < n_u_C_dofs; i++)
          for (unsigned int j = 0; j < n_u_R_dofs; j++)
            KCR(i,j) += JxW[qp]* gamma_I*phi[i][qp]*phi[j][qp];

        for (unsigned int i = 0; i < n_u_R_dofs; i++)
          for (unsigned int j = 0; j < n_u_R_dofs; j++)
            KCC(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);

          {
            const Real x = q_point[qp](0);
            const Real y = q_point[qp](1);
	    const Real z = q_point[qp](2);
            const Real eps = 1.e-3;


// for now assume that the exact solution is real

            const Real fRxyz = -(exact_solution(x,y-eps,z) +
                               exact_solution(x,y+eps,z) +
                               exact_solution(x-eps,y,z) +
                               exact_solution(x+eps,y,z) +
			       exact_solution(x,y,z+eps) +
			       exact_solution(x,y,z-eps) -
                               6.*exact_solution(x,y,z))/eps/eps + exact_solution(x,y,z)*gamma_R - gamma_I*exact_solution_I(x,y,z);
	    const Real fCxyz = -(exact_solution_I(x,y-eps,z) +
                               exact_solution_I(x,y+eps,z) +
                               exact_solution_I(x-eps,y,z) +
                               exact_solution_I(x+eps,y,z) +
                               exact_solution_I(x,y,z+eps) +
                               exact_solution_I(x,y,z-eps) -
                               6.*exact_solution_I(x,y,z))/eps/eps + gamma_R*exact_solution_I(x,y,z)+ gamma_I*exact_solution(x,y,z);

	for (unsigned int i = 0; i < n_u_R_dofs; i++)
	  FR(i) += JxW[qp]*fRxyz*phi[i][qp];
	for (unsigned int i = 0; i < n_u_C_dofs; i++)
	  FC(i) += JxW[qp]*fCxyz*phi[i][qp];


/*
            for (unsigned int i=0; i<phi.size(); i++)
              Fe(i) += JxW[qp]*fxyz*phi[i][qp]; */
          }
        }

      {

        for (unsigned int side=0; side<elem->n_sides(); side++)
          if (elem->neighbor(side) == NULL)
            {
              const std::vector<std::vector<Real> >&  phi_face = fe_face->get_phi();

              const std::vector<Real>& JxW_face = fe_face->get_JxW();

              const std::vector<Point >& qface_point = fe_face->get_xyz();

              fe_face->reinit(elem, side);


              for (unsigned int qp=0; qp<qface.n_points(); qp++)
                {

                  const Real xf = qface_point[qp](0);
                  const Real yf = qface_point[qp](1);
		  const Real zf = qface_point[qp](2);
                  const Real penalty = 1.e10;

                  const Real value = exact_solution(xf, yf, zf);
/*
                  for (unsigned int i=0; i<phi_face.size(); i++)
                    for (unsigned int j=0; j<phi_face.size(); j++)
                      Ke(i,j) += JxW_face[qp]*penalty*phi_face[i][qp]*phi_face[j][qp];

                  for (unsigned int i=0; i<phi_face.size(); i++)
                    Fe(i) += JxW_face[qp]*penalty*value*phi_face[i][qp];

*/
        for (unsigned int i = 0; i < phi_face.size(); i++)
          for (unsigned int j = 0; j < phi_face.size(); j++)
            KRR(i,j) += penalty*JxW_face[qp]*phi_face[i][qp]*phi_face[j][qp];

        for (unsigned int i = 0; i < phi_face.size(); i++)
          for (unsigned int j = 0; j < phi_face.size(); j++)
            KCC(i,j) += penalty*JxW_face[qp]*phi_face[i][qp]*phi_face[j][qp];

        for (unsigned int i = 0; i < phi_face.size(); i++)
          FR(i) += penalty*JxW_face[qp]*value*phi_face[i][qp];

       const Real valueI = exact_solution_I(xf,yf,zf);

        for (unsigned int i = 0; i < phi_face.size(); i++)
          FC(i) += penalty*JxW_face[qp]*valueI*phi_face[i][qp];
                }
            }
      }

      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);

if (print_matrix) {
Ke.print(std::cout);
std::cout << "\n ================================================================ \n";
		    }
    }
}


Number exact_solution(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution(p(0),p(1),p(2));
}

Number exact_solution_I(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution_I(p(0),p(1),p(2));
}


void assemble_helmholtz_matrix(EquationSystems& es,
                      const std::string& system_name, PetscMatrix<Number>* matrix)
{
        
        libmesh_assert_equal_to (system_name, "Helmholtz");
        
        const MeshBase& mesh = es.get_mesh();
        
        const unsigned int dim = mesh.mesh_dimension();
        
        LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> ("Helmholtz");
        
        const unsigned int u_R_var = system.variable_number("u_R");
        const unsigned int u_C_var = system.variable_number("u_C");
        
        
        const DofMap& dof_map = system.get_dof_map();
        
        FEType fe_type = dof_map.variable_type(0);
        
        AutoPtr<FEBase> fe (FEBase::build(dim, fe_type));
        
        QGauss qrule (dim, FIFTH);
        
        fe->attach_quadrature_rule (&qrule);
        
        AutoPtr<FEBase> fe_face (FEBase::build(dim, fe_type));
        
        QGauss qface(dim-1, FIFTH);
        
        fe_face->attach_quadrature_rule (&qface);
        
        const std::vector<Real>& JxW = fe->get_JxW();
        
        const std::vector<std::vector<Real> >& phi = fe->get_phi();
        
        const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();
        
        DenseMatrix<Number> Ke;
        
        DenseSubMatrix<Number>
        KRR(Ke), KRC(Ke), KCR(Ke),KCC(Ke);
    
        
        
        std::vector<dof_id_type> dof_indices;
        std::vector<dof_id_type> dof_indices_R;
        std::vector<dof_id_type> dof_indices_C;
        
        
        MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
        const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
        
        for ( ; el != end_el ; ++el)
        {
            const Elem* elem = *el;
            
            dof_map.dof_indices (elem, dof_indices);
            dof_map.dof_indices (elem, dof_indices_R, u_R_var);
            dof_map.dof_indices (elem, dof_indices_C, u_C_var);
            
            const unsigned int n_dofs = dof_indices.size();
            const unsigned int n_u_R_dofs = dof_indices_R.size();
            const unsigned int n_u_C_dofs = dof_indices_C.size();
            
            fe->reinit (elem);
            
            Ke.resize (n_dofs,
                       n_dofs);
            

            
            KRR.reposition(u_R_var*n_u_R_dofs, u_R_var*n_u_R_dofs, n_u_R_dofs, n_u_R_dofs);
            KRC.reposition(u_R_var*n_u_R_dofs, u_C_var*n_u_R_dofs, n_u_R_dofs, n_u_C_dofs);
            KCR.reposition(u_C_var*n_u_C_dofs, u_R_var*n_u_C_dofs, n_u_C_dofs, n_u_R_dofs);
            KCC.reposition(u_C_var*n_u_C_dofs, u_C_var*n_u_C_dofs, n_u_C_dofs, n_u_C_dofs);
            
 
            for (unsigned int qp=0; qp<qrule.n_points(); qp++)
            {
                
                /*          for (unsigned int i=0; i<phi.size(); i++)
                 for (unsigned int j=0; j<phi.size(); j++)
                 {
                 Ke(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);
                 }
                 */
                
                for (unsigned int i = 0; i < n_u_R_dofs; i++)
                    for (unsigned int j = 0; j < n_u_R_dofs; j++)
                        KRR(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);
                
                for (unsigned int i = 0; i < n_u_R_dofs; i++)
                    for (unsigned int j = 0; j < n_u_C_dofs; j++)
                        KRC(i,j) +=  -JxW[qp]*gamma_I*phi[i][qp]*phi[j][qp];
                
                for (unsigned int i = 0; i < n_u_C_dofs; i++)
                    for (unsigned int j = 0; j < n_u_R_dofs; j++)
                        KCR(i,j) += JxW[qp]* gamma_I*phi[i][qp]*phi[j][qp];
                
                for (unsigned int i = 0; i < n_u_R_dofs; i++)
                    for (unsigned int j = 0; j < n_u_R_dofs; j++)
                        KCC(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp] + gamma_R*phi[i][qp]*phi[j][qp]);
                

            }
            
            {
                
                for (unsigned int side=0; side<elem->n_sides(); side++)
                    if (elem->neighbor(side) == NULL)
                    {
                        const std::vector<std::vector<Real> >&  phi_face = fe_face->get_phi();
                        
                        const std::vector<Real>& JxW_face = fe_face->get_JxW();
                        
                        
                        fe_face->reinit(elem, side);
                        
                        
                        for (unsigned int qp=0; qp<qface.n_points(); qp++)
                        {
                            
                            const Real penalty = 1.e10;
                            
                            /*
                             for (unsigned int i=0; i<phi_face.size(); i++)
                             for (unsigned int j=0; j<phi_face.size(); j++)
                             Ke(i,j) += JxW_face[qp]*penalty*phi_face[i][qp]*phi_face[j][qp];
                             

                             */
                            for (unsigned int i = 0; i < phi_face.size(); i++)
                                for (unsigned int j = 0; j < phi_face.size(); j++)
                                    KRR(i,j) += penalty*JxW_face[qp]*phi_face[i][qp]*phi_face[j][qp];
                            
                            for (unsigned int i = 0; i < phi_face.size(); i++)
                                for (unsigned int j = 0; j < phi_face.size(); j++)
                                    KCC(i,j) += penalty*JxW_face[qp]*phi_face[i][qp]*phi_face[j][qp];
                            
                      
                            
                        }
                    }
            }
            
            dof_map.constrain_element_matrix (Ke, dof_indices);
            
            matrix->add_matrix (Ke, dof_indices);
            
            if (print_matrix) {
                Ke.print(std::cout);
                std::cout << "\n ================================================================ \n";
		    }
        }
    }


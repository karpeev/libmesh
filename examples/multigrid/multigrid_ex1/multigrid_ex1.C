#include <libmesh/mesh_data.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscmath.h>
#include <petscdraw.h>

#include <vector>
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
#include "libmesh/petsc_matrix.h"
#include <iostream>
#include "libmesh/petsc_vector.h"

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

PetscBool multigrid_ex1_print_statements_active = PETSC_TRUE;

using namespace libMesh;
using std::vector;

void assemble_helmholtz(EquationSystems& es,
                      const std::string& system_name);

// error checking function

void e(unsigned int i) { 
if (multigrid_ex1_print_statements_active)
PetscPrintf(PETSC_COMM_SELF, "PrintStatement: %D \n", i);
}

void e(unsigned int i, unsigned int j) {
if (multigrid_ex1_print_statements_active)
PetscPrintf(PETSC_COMM_SELF, "PrintStatement: %D (%D)\n", i, j);
}


void PrintStatus(std::string s)
{
std::cout << s << std::endl;
}

Real exact_solution (const Real x,
                     const Real y,
                     const Real z);
Real exact_solution_I (const Real x,
                      const Real y,
                      const Real z);

Number exact_solution(const Point& p, const Parameters&,const std::string&,const std::string&);
Number exact_solution_I(const Point& p, const Parameters&,const std::string&,const std::string&);

bool singularity = true;
PetscBool print_matrix = PETSC_FALSE;

Real gamma_R = 0.0;
Real gamma_I =  0.0;
PetscBool use_bc = PETSC_FALSE;

unsigned int dim = 3;

// this program is divided into three sections.

// section one builds the mesh
// section two runs the solver, with a switch for using geometric multigrid
// the third is outside main, as is the matrix assembly.
// note that currently FAC is implemented incorrectly. It uses a global smoother
// but in fact FAC is supposed to use a local smoother. It also adds in the coarse effect
// of the boundary coarse nodes. (See McCormick)



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
    unsigned int uniform_refine       = input_file("uniform_refine",0);
    const std::string refine_type     = input_file("refinement_type", "h");
    const std::string approx_type     = input_file("approx_type", "LAGRANGE");
    const unsigned int approx_order   = input_file("approx_order", 1);
    const std::string element_type    = input_file("element_type", "tensor");
    const int extra_error_quadrature  = input_file("extra_error_quadrature", 0);
    const int max_linear_iterations   = input_file("max_linear_iterations", 5000);
    const bool output_intermediate    = input_file("output_intermediate", false);
    const unsigned int x_range        = input_file("x_range", 2);
    const unsigned int y_range        = input_file("y_range", 2);
    const unsigned int z_range        = input_file("z_range", 2);
    const Real x_size                 = input_file("x_size", 1);
    const Real y_size                 = input_file("y_size", 1);
    const Real z_size                 = input_file("z_size", 1);
    const unsigned int n_uniform_refinements = input_file("n_uniform_refinements", 3);

    PetscBool use_gmg = PETSC_FALSE;
    PetscBool mesh_build = PETSC_FALSE;
    PetscBool use_galerkin = PETSC_FALSE;
    PetscOptionsGetBool(NULL, "-use_galerkin", &use_galerkin, NULL);
    PetscOptionsGetBool(NULL, "-use_debug", &multigrid_ex1_print_statements_active, NULL);
    PetscOptionsGetBool(NULL, "-print_matrix", &print_matrix, NULL);
    PetscOptionsGetBool(NULL, "-mesh_build", &mesh_build, NULL);
    PetscOptionsGetBool(NULL, "-use_bc", &use_bc, NULL);
    PetscOptionsGetBool(NULL, "-use_gmg", &use_gmg, NULL);
    PetscBool print_interp = PETSC_FALSE;
    PetscOptionsGetBool(NULL, "-write_interp", &print_interp,NULL);
    PetscBool use_null_space = PETSC_FALSE;
    PetscOptionsGetBool(NULL, "-use_null_space", &use_null_space, NULL);

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

  Mesh mesh(init.comm());

if (mesh_build)
{
int count_unif =(int) n_uniform_refinements;

  libmesh_example_assert(3 <= LIBMESH_DIM, "3D support");
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
if (count_unif <= 0)
uniform_refine = 0;
else
uniform_refine = 1;
count_unif--;
if (uniform_refine)
std::cout << "UNIFORM REFINEMENT\n";
else
std::cout << "NON-UNIFORM REFINEMENT\n";

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

  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems ("lshaped.e",
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API
mesh.write("refined_cube.xda");

    out << "];" << std::endl;
    out << "hold on" << std::endl;
    out << "plot(e(:,1), e(:,2), 'bo-');" << std::endl;
    out << "plot(e(:,1), e(:,3), 'ro-');" << std::endl;
    out << "xlabel('dofs');" << std::endl;
    out << "title('" << approx_name << " elements');" << std::endl;
    out << "legend('L2-error');" << std::endl;
  #endif // #ifndef LIBMESH_ENABLE_AMR

}
// 				END ADAPTIVELY REFINE AND SOLVE
// ===================================================================================
//      			BEGIN MULTIGRID LEVEL BUILDING
else // here we are solving instead of mesh_building. Note we don't do both in one go
     // because we don't want the mesh builder to use Multigrid
{

// MeshData  data(mesh);
PrintStatus("loading mesh . . .");

  mesh.read("refined_cube.xda");

  EquationSystems equation_systems (mesh);


PrintStatus("building system 1 . . .");
LinearImplicitSystem& system = equation_systems.add_system<LinearImplicitSystem> ("Helmholtz");

  system.add_variable("u_R", static_cast<Order>(approx_order),
                        Utility::string_to_enum<FEFamily>(approx_type));
  system.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));

  equation_systems.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  equation_systems.init();

 std::cout << "\nNumber of Levels Test: " << MGCountLevelsFAC(mesh) << std::endl << std::endl;


  // Here I begin experimentation



PrintStatus("initializing storage . . .");
  PetscVector<Number> **level_vector = new PetscVector<Number>* [n_levels_coarsen];
  PetscMatrix<Number> **level_interp = new PetscMatrix<Number>* [n_levels_coarsen];
  PetscMatrix<Number> **level_A      = new PetscMatrix<Number>* [n_levels_coarsen];
  MGElemIDConverter    *id_converter = new MGElemIDConverter    [n_levels_coarsen];

  PetscMatrix<Number> **level_restrct = new PetscMatrix<Number>* [n_levels_coarsen];

  for (unsigned int i = 0; i < n_levels_coarsen; i++)
{
  level_vector[i] = new PetscVector<Number>(system.comm());
  level_interp[i] = new PetscMatrix<Number>(system.comm());
  level_A[i]      = new PetscMatrix<Number>(system.comm());
  level_restrct[i] = new PetscMatrix<Number>(system.comm());

  unsigned int n_processors = level_A[i]->n_processors();

  level_A[i]->init(n_processors, n_processors, 1, 1, 1, 1);
  level_vector[i]->init(n_processors, 1, 0);

}
e(3);

  PetscVector<Number> * petsc_vec;
  PetscMatrix<Number> * petsc_mat;

PrintStatus("gathering first data . . .");

  system.assemble();
  petsc_vec = (PetscVector<Number>*) system.rhs;
  petsc_mat = (PetscMatrix<Number>*) system.matrix;

PetscPrintf(PETSC_COMM_SELF, "AND CONTINUING ONTO SWAPPING!\n");
  level_A[0]->swap(*petsc_mat);
  level_vector[0]->swap(*petsc_vec);

if (use_gmg) {


  mesh.skip_partitioning(1); // prevents reparitioning upon each coarsening

PrintStatus("building mesh 2 . . .");
  Mesh mesh_2(mesh);




                                // equation_system_2, and equation_system_3 will be looped over
PrintStatus("setting up system 2 . . .");

 EquationSystems equation_systems_2 (mesh_2);

  LinearImplicitSystem& system_2 = equation_systems_2.add_system<LinearImplicitSystem> ("Helmholtz");
  system_2.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
  system_2.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
  equation_systems_2.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  equation_systems_2.init();
e(2);


PrintStatus("coarsening mesh 2 . . .");

 system_2.project_solution_on_reinit() = false;

  MeshRefinement mesh_refinement_2(mesh_2);
  mesh_refinement_2.coarsen_by_parents();
  mesh_refinement_2.clean_refinement_flags();
  flag_elements_FAC(mesh_2, id_converter[0]);
PetscPrintf(PETSC_COMM_WORLD, ":\n");
  mesh_refinement_2.coarsen_elements();
PetscPrintf(PETSC_COMM_WORLD, ":\n");
   equation_systems_2.reinit();
flag_elements_FAC_end(id_converter[0]);

e(5);

PrintStatus("building mesh 3 . . .");

  Mesh mesh_3(mesh_2);

PrintStatus("building system 3 . . .");

EquationSystems equation_systems_3 (mesh_3);

  LinearImplicitSystem& system_3 = equation_systems_3.add_system<LinearImplicitSystem> ("Helmholtz");
  system_3.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
  system_3.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
  equation_systems_3.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

  equation_systems_3.init();

e(6);

PrintStatus("gathering second data . . .");


 system_2.assemble();

 petsc_vec = (PetscVector<Number>*) system_2.rhs;
 petsc_mat = (PetscMatrix<Number>*) system_2.matrix;


 level_vector[1]->swap(*petsc_vec);
 level_A[1]->swap(*petsc_mat);
 
 MatAssemblyBegin(level_A[1]->mat(),MAT_FINAL_ASSEMBLY);
 MatAssemblyEnd(level_A[1]->mat(),MAT_FINAL_ASSEMBLY);

 system_3.project_solution_on_reinit() = false;
PrintStatus("coarsening mesh 3 . . .");
  MeshRefinement mesh_refinement_3(mesh_3);
  mesh_refinement_3.coarsen_by_parents();
  flag_elements_FAC(mesh_3, id_converter[1]);
  mesh_refinement_3.coarsen_elements();
  equation_systems_3.reinit();
  flag_elements_FAC_end(id_converter[1]);

PrintStatus("gathering 3rd data . . .");

system_3.assemble();
 petsc_vec = (PetscVector<Number>*) system_3.rhs;
 petsc_mat = (PetscMatrix<Number>*) system_3.matrix;
 level_vector[2]->swap(*petsc_vec);





  level_A[2]->swap(*petsc_mat);

e(1);
 build_interpolation(equation_systems, equation_systems_2, "Helmholtz", *level_interp[0], *level_vector[0], *level_vector[1], id_converter[0]);
e(2);

if (print_interp)
{
VecAssemblyBegin(level_vector[0]->vec());
VecAssemblyEnd(level_vector[0]->vec());
VecAssemblyBegin(level_vector[1]->vec());
VecAssemblyEnd(level_vector[1]->vec());

PetscRandom rctx;
PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
PetscRandomSetType(rctx,PETSCRAND);
VecSetRandom(level_vector[1]->vec(), rctx);

 level_interp[0]->vector_mult(*level_vector[0], *level_vector[1]);
e(3);

system.solution->init(*level_vector[0]);
system_2.solution->init(*level_vector[1]);
 *system.solution = *level_vector[0];
 *system_2.solution = *level_vector[1];
 ExodusII_IO (mesh).write_equation_systems("interpolated.e", equation_systems);
 ExodusII_IO (mesh_2).write_equation_systems("interpolated_from.e", equation_systems_2);
}
e(4);


 for (unsigned int i = 2; i < n_levels_coarsen; i++)
 {
e(7+i);


PrintStatus("gathering more data . . .");

  build_interpolation(equation_systems_2, equation_systems_3, "Helmholtz", *level_interp[i-1], *level_vector[i-1], *level_vector[i], id_converter[i-1]);

e(101);
  


if (print_interp && i == 2)
{
VecAssemblyBegin(level_vector[1]->vec());
VecAssemblyEnd(level_vector[1]->vec());
VecAssemblyBegin(level_vector[2]->vec());
VecAssemblyEnd(level_vector[2]->vec());

PetscRandom rctx;
PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
PetscRandomSetType(rctx,PETSCRAND);
VecSetRandom(level_vector[2]->vec(), rctx);

 level_interp[1]->vector_mult(*level_vector[1], *level_vector[2]);

system_2.solution->init(*level_vector[1]);
system_3.solution->init(*level_vector[2]);
 *system_2.solution = *level_vector[1];
 *system_3.solution = *level_vector[2];
 ExodusII_IO (mesh_2).write_equation_systems("interpolated_2.e", equation_systems_2);
 ExodusII_IO (mesh_3).write_equation_systems("interpolated_from_2.e", equation_systems_3);


 ConstElemRange range
  (mesh_2.active_elements_begin(),
  mesh_2.active_elements_end());

}


  if (i != n_levels_coarsen-1)
  {
    flag_elements_FAC(mesh_3, id_converter[i]);
    flag_elements_FAC(mesh_2, id_converter[0]);
    mesh_refinement_2.coarsen_elements();
    mesh_refinement_3.coarsen_elements();
    equation_systems_2.reinit();
    equation_systems_3.reinit();
    flag_elements_FAC_end(id_converter[i]);
    flag_elements_FAC_end(id_converter[0]);

system_3.assemble();
    petsc_vec = (PetscVector<Number>*) system_3.rhs;
    petsc_mat = (PetscMatrix<Number>*) system_3.matrix;
    level_vector[i+1]->swap(*petsc_vec);
    level_A[i+1]->swap(*petsc_mat);

  MatAssemblyBegin(level_A[i+1]->mat(),MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(level_A[i+1]->mat(), MAT_FINAL_ASSEMBLY);

  }

 }


}

PetscErrorCode ierr;

PrintStatus("adding components to PCMG . . .");

PetscInt nlevels = n_levels_coarsen;
PC pc;
KSP ksp;
KSPCreate(PETSC_COMM_WORLD,&ksp);
e(1);

MatNullSpace sp;

if (use_null_space)
{
PetscBool check_valid = PETSC_FALSE;
MatNullSpaceCreate(PetscObjectComm((PetscObject) level_A[0]->mat()), PETSC_TRUE, 0, NULL, &sp);
MatSetNullSpace(level_A[0]->mat(), sp);
MatNullSpaceTest(sp, level_A[0]->mat(), &check_valid);
std::cout << "Null Space Test: " << check_valid << "\n";
}



KSPSetOperators(ksp, level_A[0]->mat(),level_A[0]->mat());
KSPGetPC(ksp,&pc);
if (use_gmg) {
PCSetType(pc,PCMG);
PCMGSetLevels(pc,nlevels,NULL);
PCMGSetType(pc,PC_MG_MULTIPLICATIVE);
e(1006);


 if (!use_galerkin)
 PCMGSetResidual(pc, nlevels - 1, NULL, level_A[0]->mat());

for (unsigned int i = 1; i < nlevels; i++)
{
e(1007);
PetscPrintf(PETSC_COMM_WORLD, "Setting a residual . . .\n");

PCMGSetInterpolation(pc,i,level_interp[nlevels-1-i]->mat());

if (use_null_space)
{
PetscBool check_valid = PETSC_FALSE;
MatNullSpaceCreate(PetscObjectComm((PetscObject) level_A[nlevels-i]->mat()), PETSC_TRUE, 0, NULL, &sp);
MatSetNullSpace(level_A[nlevels-i]->mat(), sp);
MatNullSpaceTest(sp, level_A[nlevels-i]->mat(), &check_valid);
std::cout << "Null Space Test: " << check_valid << "\n";
}

  if (!use_galerkin)
  PCMGSetResidual(pc, i-1, NULL, level_A[nlevels-i]->mat());

e(1008);
}
}
e(3);
KSPSetFromOptions(ksp);


e(1010);

Vec sol;
VecDuplicate(level_vector[0]->vec(), &sol);
e(1013);

if (use_gmg && print_matrix)
{
MatView(level_A[nlevels-1]->mat(), PETSC_VIEWER_STDOUT_WORLD);
}

PetscPrintf(PETSC_COMM_WORLD, "KSP Setup . . .\n");
ierr = KSPSetUp(ksp);CHKERRQ(ierr);

PetscPrintf(PETSC_COMM_WORLD, "System Size: %D\n", level_A[0]->m());
PetscPrintf(PETSC_COMM_WORLD, "KSP Solving . . .\n");

ierr = KSPSolve(ksp,level_vector[0]->vec(),sol);CHKERRQ(ierr);

VecSet(sol, 1.);
PetscVector<Number> sol_check(sol,system.comm());
system.solution->init(sol_check);
level_A[0]->vector_mult(*system.solution, sol_check);

            ExodusII_IO (mesh).write_equation_systems ("ksp_solution.e",
                                                 equation_systems);


PetscInt its = 0;
KSPGetIterationNumber(ksp,&its);

PetscPrintf(PETSC_COMM_WORLD, "Iterations %D\n", its);

PetscVector<Number> solution(sol,init.comm());
*system.solution = solution;

  #ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems ("lshape_solved_solution.e",
                                         equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

PrintStatus("cleaning up . . .");

for (unsigned int i = 0; i < n_levels_coarsen; i++)
{
 delete level_A[i];
 delete level_vector[i];
 delete level_interp[i];
 delete level_restrct[i];
}
 delete [] level_A; delete [] level_vector; delete [] level_interp; delete [] level_restrct;
}

PrintStatus("Complete.");

  return 0;
}

void assemble_helmholtz(EquationSystems& es,
                      const std::string& system_name)
{
 if (use_bc)
std::cout << "used penalty\n";

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


          }
        }

      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);
      }

system.matrix->close();
system.rhs->close();
PetscPrintf(PETSC_COMM_WORLD, "First Part Done\n");


if (use_bc) // add in boundary conditions
{
  unsigned int processor_id = system.processor_id();

  unsigned int n_variables = system.n_vars();
  vector<PetscBool> bddy_dof_done;
  bddy_dof_done.resize(system.rhs->local_size()); // this tells if this dof has been inserted.
					   // note that we use this per variable.

PetscPrintf(PETSC_COMM_SELF, "system.rhs->local_size()=%D\n", system.rhs->local_size());

// currently, I simply set the RHS on the boundary to zero
// later it should be altered to take general boundary conditions

  for (unsigned int var=0; var<n_variables; var++)
  {
   el     = mesh.active_local_elements_begin();
   for (unsigned int i = 0; i < bddy_dof_done.size(); i++)
    bddy_dof_done[i] = PETSC_FALSE;
 unsigned int node_count = 0;
   for ( ; el != end_el ; ++el)
    {
      const Elem* elem = *el;

      // This tells what nodes of the element are on the boundary

      vector<PetscBool> node_on_side;

        for (unsigned int side=0; side<elem->n_sides(); side++)
            if (elem->neighbor(side)==NULL)
            {
              for (unsigned int node = 0; node < elem->n_nodes(); node++)
              {
		PetscBool check_on_processor = PETSC_FALSE;
		PetscBool check_if_row_done = PETSC_FALSE;

		unsigned int dof = elem->get_node(node)->dof_number(0,var,0);

		if (dof <= dof_map.last_dof() && dof >= dof_map.first_dof())
		  check_on_processor = PETSC_TRUE;

               if (elem->is_node_on_side(node, side) && check_on_processor)
			if (bddy_dof_done[dof - dof_map.first_dof()] == PETSC_FALSE) // avoid double-counting
                        { bddy_dof_done[dof-dof_map.first_dof()] = PETSC_TRUE; node_count++;}
              }
            }

       // now node_on_side knows what nodes are of interest, so we start our looping

      //        system.matrix->zero_rows(rows, 1.);
    //          system.rhs->insert(rhs, rows);
    }

vector<unsigned int> row;
vector<double> rhs;
rhs.resize(node_count);
row.resize(node_count);

unsigned int index = 0;
for (unsigned int i = 0; i < bddy_dof_done.size(); i++)
if (bddy_dof_done[i])
{
rhs[index] = 0;
row[index++] = dof_map.first_dof() + i;
}
system.matrix->zero_rows(row, 1.);
system.rhs->insert(rhs, row);

}
e(102, processor_id);
system.matrix->close();
system.rhs->close();
e(103, processor_id);
}

PetscPrintf(PETSC_COMM_SELF, "matrix assembled.\n");
}


Number exact_solution(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution(p(0),p(1),p(2));
}

Number exact_solution_I(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution_I(p(0),p(1),p(2));
}


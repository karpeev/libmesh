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

PetscBool debug = PETSC_TRUE;

using namespace libMesh;
using std::vector;

void assemble_helmholtz(EquationSystems& es,
                      const std::string& system_name);

#include "dm_sample.h"


Real gamma_R = 0.0;
Real gamma_I = 0.0;
bool use_bc  = false;

unsigned int dim = 3;
bool singularity = true;
bool print_matrix = false;

Real exact_solution (const Real x,
                     const Real y,
                     const Real z);
Real exact_solution_I (const Real x,
                      const Real y,
                      const Real z);

Number exact_solution(const Point& p, const Parameters&,const std::string&,const std::string&);
Number exact_solution_I(const Point& p, const Parameters&,const std::string&,const std::string&);


// this program is divided into three sections.

// section one builds the mesh
// section two runs the solver, with a switch for using geometric multigrid
// the third is outside main, as is the matrix assembly.
// note that currently FAC is implemented incorrectly. It uses a global smoother
// but in fact FAC is supposed to use a local smoother. It also adds in the coarse effect
// of the boundary coarse nodes. (See McCormick)



int main (int argc, char** argv)
{
  PetscErrorCode ierr;

  LibMeshInit init (argc, argv);

#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#else

  //bool  verbose                       = on_command_line("--verbose");

  // Problem
  gamma_R                             = command_line_value("gamma_R", 1.0);
  gamma_I	                      = command_line_value("gamma_I", 0.0);
  use_bc                              = on_command_line("--dirichlet_bc");
  dim                                 = command_line_value("dim", 3);
  singularity                         = on_command_line("--rhs_singularity");

  // Mesh
  const unsigned int x_range          = command_line_value("nx", 2);
  const unsigned int y_range          = command_line_value("ny", 2);
  const unsigned int z_range          = command_line_value("nz", 2);
  const Real x_size                   = command_line_value("xlen", 1.0);
  const Real y_size                   = command_line_value("ylen", 1.0);
  const Real z_size                   = command_line_value("ylen", 1.0);

  // Approximation
  const std::string approx_type       = command_line_value("approx_type", std::string("LAGRANGE"));
  const unsigned int approx_order     = command_line_value("approx_order", 1);
  const std::string element_type      = command_line_value("element_type", std::string("tensor"));     // simplex|tensor

  // Hierarchy generation
  bool build_mesh                    = on_command_line("--build_mesh");
  bool save_mesh                     = on_command_line("--save_mesh");
  std::string mesh_filename           = command_line_value("mesh_filename",std::string("cube"));

  // Refinement
  const unsigned int max_r_steps           = command_line_value("n_refinements", 3);         // total number of refinements
  const unsigned int n_uniform_refinements = command_line_value("n_uniform_refinements", 3); // of which n_uniform_refinements are uniform (possibly all)
  const std::string refine_type            = command_line_value("refinement_type", std::string("h"));     // the remaining refinements are of type 'refinement_type': h|p|hp|matchedhp|singularhp
  const Real refine_percentage             = command_line_value("refine_percentage", 0.5);   // refinement/coarsening percentages
  const Real coarsen_percentage            = command_line_value("coarsen_percentage", 0.5);  //    for the non-uniform refinements
  const std::string indicator_type         = command_line_value("refinement_indicator_type", std::string("kelly")); // exact|patch|uniform|kelly
  const unsigned int max_r_level           = command_line_value("max_element_refinement_level", 3);    // FIXME: how is this different from n_refinements?

  // Solver
  bool  solve                         = on_command_line("--solve");
  //bool  use_null_space                = on_command_line("--use_null_space");
  PetscInt mg_levels = 3;
  ierr  = PetscOptionsGetInt(NULL,"-pc_mg_levels",&mg_levels,NULL);
  CHKERRABORT(init.comm().get(),ierr);
  const unsigned int n_levels_coarsen = (unsigned int) mg_levels;
  PetscBool use_galerkin = PETSC_FALSE;
  PetscOptionsGetBool(NULL, "-pc_mg_galerkin", &use_galerkin, NULL);




  //                              REFINE AND GENERATE THE HIERARCHY
  // ===================================================================================
  libmesh_example_requires(dim <= LIBMESH_DIM, "3D support");

  Mesh mesh(init.comm());

  if (build_mesh) {
    int count_unif = (int) n_uniform_refinements;

    libmesh_example_requires(3 <= LIBMESH_DIM, "3D support");
    libMesh::out << "Building a cubic mesh: [0.0," << x_size <<"]x[0.0," << y_size << "]x[0.0," << z_size << "]\n";
    libMesh::out << "Numbers of elements: " << x_range << "x" << y_range << "x" << z_range << "\n";
    libMesh::out << "Using element type " << element_type << std::endl;

    MeshTools::Generation::build_cube (mesh,x_range, y_range, z_range,0., x_size,0.,y_size, 0.,z_size,HEX8);
    if (element_type == "simplex") MeshTools::Modification::all_tri(mesh);



    libMesh::out << "Approximation type " << approx_type << ", approximation order " << approx_order << std::endl;
    if (approx_order > 1 || refine_type != "h") {
      mesh.all_second_order();
    }

    mesh.print_info();

    EquationSystems equation_systems (mesh);

    LinearImplicitSystem& system = equation_systems.add_system<LinearImplicitSystem> ("Helmholtz");

    system.add_variable("u_R", static_cast<Order>(approx_order),
    Utility::string_to_enum<FEFamily>(approx_type));
    system.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));

    ExactSolution exact_sol_I(equation_systems);
    ExactSolution exact_sol(equation_systems);

    equation_systems.init();
    equation_systems.print_info();

    libMesh::out << "Mesh refinement parameters:\n";
    libMesh::out << "\t refinement percentage " << refine_percentage << std::endl;
    libMesh::out << "\t coarsen percentage " << coarsen_percentage << std::endl;
    MeshRefinement mesh_refinement(mesh);
    mesh_refinement.refine_fraction() = refine_percentage;
    mesh_refinement.coarsen_fraction() = coarsen_percentage;
    mesh_refinement.max_h_level() = max_r_level;

    for (unsigned int r_step=0; r_step<max_r_steps; r_step++) {
      bool uniform_refine;
      if (count_unif <= 0)
        uniform_refine = false;
      else
        uniform_refine = true;
      count_unif--;
      if (uniform_refine) {
        libMesh::out << "UNIFORM REFINEMENT\n";
      } else {
	libMesh::out << "NON-UNIFORM REFINEMENT\n";
      }
      libMesh::out << "System has: " << equation_systems.n_active_dofs()
                << " degrees of freedom."
                << std::endl;

      if (r_step < max_r_steps) {
        libMesh::out << "  Refining the mesh..." << std::endl;
        if (!uniform_refine) {

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
          mesh_refinement.flag_elements_by_error_fraction (error);
          if (refine_type == "p"){
            mesh_refinement.switch_h_to_p_refinement();
          }
          if (refine_type == "matchedhp") {
            mesh_refinement.add_p_to_h_refinement();
          }
          if (refine_type == "hp") {
            HPCoarsenTest hpselector;
            hpselector.select_refinement(system);
          }
          if (refine_type == "singularhp") {
            libmesh_assert (singularity);
            HPSingularity hpselector;
            hpselector.singular_points.push_back(Point());
            hpselector.select_refinement(system);
          }
          mesh_refinement.refine_and_coarsen_elements();
        } else if (uniform_refine) {
                if (refine_type == "h" || refine_type == "hp" || refine_type == "matchedhp")
                  mesh_refinement.uniformly_refine(1);
                if (refine_type == "p" || refine_type == "hp" || refine_type == "matchedhp")
                  mesh_refinement.uniformly_p_refine(1);
        }
        equation_systems.reinit ();
      }
    }
    if (save_mesh) {
      std::string filename = "refined_"+mesh_filename+".xda";
      libMesh::out << "Saving mesh to file '" << filename << "'\n";
      mesh.write("refined_"+mesh_filename+".xda");
    }
#endif // #ifndef LIBMESH_ENABLE_AMR
  }
  // 				END HIERARCHY GENERATION
  // ===================================================================================
  //      			BEGIN MULTIGRID LEVEL BUILDING
  if (solve) {
    libMesh::out << "Solving Helmholtz with gamma = " << gamma_R << " + i*" << gamma_I << std::endl;
    if (!build_mesh) {
      std::string filename = "refined_"+mesh_filename+".xda";
      libMesh::out << "\tUsing mesh from file '" << filename << "'\n";
      mesh.read(filename);
    }

    EquationSystems equation_systems(mesh);

    LinearImplicitSystem & system = equation_systems.add_system<LinearImplicitSystem> ("Helmholtz");
    system.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
    system.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
    equation_systems.get_system("Helmholtz").attach_assemble_function (assemble_helmholtz);

    libMesh::out << "Initializing levels storage ...\n";

    dm** dm_levels;
    dm_levels = new dm*[n_levels_coarsen];

    libMesh::out << "Building level-0\n";
    dm_levels[0] = new dm("Helmholtz");
    dm_levels[0]->set_eq(&equation_systems, approx_order,approx_type);

    libMesh::out << "Level-0 rhs\n";
    PetscVector<Number> level_vector(init.comm());
    dm_levels[0]->copy_rhs(level_vector);

    libMesh::out << "\nNumber of Levels: " << MGCountLevelsFAC(mesh) << std::endl << std::endl;

    PetscMatrix<Number> **level_interp = new PetscMatrix<Number>* [n_levels_coarsen];

    for (unsigned int i = 0; i < n_levels_coarsen; i++) {
      level_interp[i] = new PetscMatrix<Number>(system.comm());
    }


    for (unsigned int k = 0; k < n_levels_coarsen-1; k++) {

      dm_levels[k+1] = new dm("Helmholtz");

      libMesh::out << "Building level " << k+2 <<" . . .\n";


      dm_levels[k]->coarsen(*dm_levels[k+1]);

      dm_levels[k]->createInterpolation(*dm_levels[k+1], *level_interp[k]);

      if (!use_galerkin) dm_levels[k+1]->assemble();
    }


    libMesh::out << "Configuring KSP . . .\n";
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);
    CHKERRABORT(init.comm().get(),ierr);

    KSPSetOperators(ksp, dm_levels[0]->get_mat(),dm_levels[0]->get_mat());
    libMesh::out << "Configuring default PC: PCMG . . .\n";

    PetscInt nlevels = n_levels_coarsen;
    PC pc;
    ierr = KSPGetPC(ksp,&pc);
    CHKERRABORT(init.comm().get(),ierr);

    ierr = PCSetType(pc,PCMG);
    CHKERRABORT(init.comm().get(),ierr);
    ierr = PCMGSetLevels(pc,nlevels,NULL);
    CHKERRABORT(init.comm().get(),ierr);

    PCMGSetType(pc,PC_MG_MULTIPLICATIVE);
    CHKERRABORT(init.comm().get(),ierr);

    libMesh::out << "Setting up interpolations\n";

    for (unsigned int i = 1; i < n_levels_coarsen; i++) {
      ierr = PCMGSetInterpolation(pc,i,level_interp[nlevels-1-i]->mat());
      CHKERRABORT(init.comm().get(),ierr);
    }


    if (!use_galerkin) {
      libMesh::out << "NOT using Galerking MG; configuring smoothers manually.\n";
      for (unsigned int i = 0; i < n_levels_coarsen; i++) {
	KSP smoother;
	ierr = PCMGGetSmoother(pc, nlevels - 1 - i, &smoother);
	CHKERRABORT(init.comm().get(),ierr);
	KSPSetOperators(smoother, dm_levels[i]->get_mat(), dm_levels[i]->get_mat());
	CHKERRABORT(init.comm().get(),ierr);
      }
    }

    libMesh::out << "Setting KSP from options\n";
    ierr = KSPSetFromOptions(ksp);
    CHKERRABORT(init.comm().get(),ierr);

    Vec sol;
    ierr = VecDuplicate(level_vector.vec(), &sol);
    CHKERRABORT(init.comm().get(),ierr);
    libMesh::out << "Setting rhs\n";
    ierr = VecSet(level_vector.vec(), 1.0);
    CHKERRABORT(init.comm().get(),ierr);

    libMesh::out << "KSPSetUp() . . .\n";
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    CHKERRABORT(init.comm().get(),ierr);


    libMesh::out << "KSPSolve() . . .\n";

    ierr = KSPSolve(ksp,level_vector.vec(),sol);
    CHKERRABORT(init.comm().get(),ierr);

    PetscInt its = 0;
    ierr = KSPGetIterationNumber(ksp,&its);
    CHKERRABORT(init.comm().get(),ierr);

    libMesh::out << "Solver returned after " << its << " iterations\n";

    libMesh::out << "cleaning up . . .\n";

    for (unsigned int i = 0; i < n_levels_coarsen; i++) {
      delete dm_levels[i];
    }

    delete [] dm_levels;
    delete [] level_interp;
  }
  libMesh::out << "Done\n";
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


  libMesh::out << "Assembling bulk Helmholtz.\n";
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
  libMesh::out << "Assembled bulk Helmholtz.\n";

  if (use_bc) {// add in boundary conditions
    libMesh::out << "Assembling BCs.\n";
    unsigned int processor_id = system.processor_id();

    unsigned int n_variables = system.n_vars();
    vector<PetscBool> bddy_dof_done;
    bddy_dof_done.resize(system.rhs->local_size()); // this tells if this dof has been inserted.
    // note that we use this per variable.

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
    system.matrix->close();
    system.rhs->close();
    libMesh::out << "Assembed BCs.\n";
  }

  libMesh::out << "Helmholtz assembled.\n";
}


Number exact_solution(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution(p(0),p(1),p(2));
}

Number exact_solution_I(const Point& p, const Parameters&, const std::string&, const std::string &)
{
return exact_solution_I(p(0),p(1),p(2));
}

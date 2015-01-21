class dm
{
Mesh * mesh;
LinearImplicitSystem * sys;
EquationSystems * eq;
std::string sys_name;
bool set;
bool memory_allocation;

unsigned int approx_order;
std::string approx_type;

public:
Mat get_mat();
dm(std::string);
void set_eq(EquationSystems*, Mesh*, LinearImplicitSystem*, const unsigned int&, const std::string&);
void coarsen(dm&);
void createInterpolation(dm&, PetscMatrix<Number>&);
bool is_set();
~dm();
void assemble();
void copy_rhs(PetscVector<Number> &);
};

dm::~dm()
{
if (memory_allocation)
{
delete mesh;
delete eq;
}
}

Mat dm::get_mat()
{
PetscMatrix<Number> * mat =(PetscMatrix<Number>*) sys->matrix;
return mat->mat();
}

void dm::copy_rhs(PetscVector<Number> & rhs)
{
PetscVector<Number> * rhs_ = (PetscVector<Number>*) sys->rhs;
rhs.init(*rhs_);
}

void dm::assemble()
{ eq->get_system(sys_name).assemble(); }

bool dm::is_set() { return set; }

void dm::set_eq(EquationSystems* eq_in,Mesh* mesh_, LinearImplicitSystem* sys_, const unsigned int & a_o, const std::string & a_t)
{
mesh = mesh_;
sys = sys_;
eq = eq_in;
set = 1;
eq->init();
assemble();
approx_type = a_t;
approx_order = a_o;
memory_allocation = 0;
}

dm::dm(std::string  sys_name_in) : sys_name(sys_name_in) {set = 0; }

void dm::coarsen(dm & coarse_dm)
{
PetscErrorCode ierr;

if (set && !coarse_dm.is_set()) {

  eq->get_mesh().skip_partitioning(1); // later may wish to save old paritioning bool
				   // and return to previous state
  Mesh * mesh_mg = new Mesh(*mesh);
  coarse_dm.eq = new EquationSystems(*mesh_mg);
  coarse_dm.mesh = mesh_mg;

// This is a temporary explicit system defining
// Need a system deep copy here

  LinearImplicitSystem & system = coarse_dm.eq->add_system<LinearImplicitSystem>(sys_name);
  system.add_variable("u_R", static_cast<Order>(approx_order), Utility::string_to_enum<FEFamily>(approx_type));
  system.add_variable("u_C", static_cast<Order>(approx_order),Utility::string_to_enum<FEFamily>(approx_type));
  coarse_dm.sys = &system; 
 
  coarse_dm.approx_order = approx_order;
  coarse_dm.approx_type = approx_type;

  coarse_dm.eq->get_system(sys_name).attach_assemble_function(assemble_helmholtz);

  coarse_dm.sys->project_solution_on_reinit() = false;
  coarse_dm.eq->init();

  MeshRefinement mesh_refinement(*mesh_mg);
  mesh_refinement.coarsen_by_parents();
  mesh_refinement.clean_refinement_flags();
  flag_elements_FAC(*mesh_mg);
  mesh_refinement.coarsen_elements();
  coarse_dm.eq->reinit();
  coarse_dm.assemble();
  coarse_dm.set = 1;
  coarse_dm.memory_allocation = 1;

  }
else
ierr =  PetscPrintf(PETSC_COMM_SELF, "DM set-status wrong\n");

}

void dm::createInterpolation(dm& coarse_dm, PetscMatrix<Number>& Interp)
{
build_interpolation(*eq, *coarse_dm.eq, sys_name, Interp,*((PetscVector<Number>*) sys->rhs), *((PetscVector<Number>*) coarse_dm.sys->rhs));
}

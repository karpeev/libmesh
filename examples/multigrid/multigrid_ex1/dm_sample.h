class dm
{
EquationSystems & eq;
std::string & sys_name;

public:
dm(EquationSystems&, std::string&);
void coarsen(dm&);
void createInterpolation(dm&, PetscMatrix<Number>*);
};

dm::dm(EquationSystems& eq_in, std::string & sys_name_in) : eq(eq_in), sys_name(sys_name_in)
{

}

void coarsen(dm & coarse_dm)
{

}

void createInterpolation(dm& coarse_dm, PetscMatrix<Number>*)
{
}

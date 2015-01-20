example_name  = multigrid_ex1
check_SCRIPTS = run.sh
install_dir   = $(examples_install_path)/multigrid/ex1
data          = multigrid_ex1.C mg_tool.h dm_sample.h exact_solution.C multigrid_ex1.in run.sh
sources       = $(data) run.sh

CLEANFILES = out_0.vtu out.pvtu

##############################################
# include common example environment
include $(top_srcdir)/examples/Make.common

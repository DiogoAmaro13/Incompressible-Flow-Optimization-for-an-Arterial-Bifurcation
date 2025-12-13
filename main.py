import dolfinx.fem
import dolfinx.io
import mpi4py.MPI
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc
import ufl
import viskex

import multiphenicsx.fem
import multiphenicsx.fem.petsc

# Import components
from parameters import (
    nu, reg_smooth, reg_l2, mixing_ratio,
    inlet_tag, wall_tag, control_tag, obs_tag
)
from mesh_builder import create_domain_and_mesh


def main():
    # ============================================================== MESH CREATION ======================================================= #
    comm = mpi4py.MPI.COMM_WORLD
    mesh, subdomains, boundaries_and_interfaces = create_domain_and_mesh(comm)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # --- Boundary and Interface Indices ---
    boundaries_inlet = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == inlet_tag]
    boundaries_walls = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == wall_tag]
    boundaries_control = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == control_tag]
    interfaces_obs = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == obs_tag]
    boundaries_inlet_and_walls = boundaries_and_interfaces.indices[
        np.isin(boundaries_and_interfaces.values, (inlet_tag, wall_tag))
    ]

    # integration measure dS on the observation interface
    integration_entities_on_Gamma_obs = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet, mesh.topology, interfaces_obs)

    if integration_entities_on_Gamma_obs.size == 0:
        integration_entities_on_Gamma_obs_flat = np.array([], dtype=np.int32)
    else:
        # Reorder facets for dS measure on interfaces
        integration_entities_on_Gamma_obs_reshaped = integration_entities_on_Gamma_obs.reshape(-1, 4)
        connected_cells_to_Gamma_obs = integration_entities_on_Gamma_obs_reshaped[:, [0, 2]]
        subdomain_ordering = (
            subdomains.values[connected_cells_to_Gamma_obs[:, 0]] <
            subdomains.values[connected_cells_to_Gamma_obs[:, 1]]
        )
        if len(subdomain_ordering) > 0 and np.any(subdomain_ordering):
            integration_entities_on_Gamma_obs_reshaped[subdomain_ordering] = \
                integration_entities_on_Gamma_obs_reshaped[subdomain_ordering][:, [2, 3, 0, 1]]

        integration_entities_on_Gamma_obs_flat = integration_entities_on_Gamma_obs_reshaped.flatten()


    dx = ufl.Measure("dx", subdomain_data=subdomains)
    ds = ufl.Measure("ds", subdomain_data=boundaries_and_interfaces)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=[(obs_tag, np.array(integration_entities_on_Gamma_obs_flat, dtype=np.int32))])

    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector([n[1], -n[0]])

    try:
        viskex.dolfinx.plot_mesh(mesh)
        viskex.dolfinx.plot_mesh_tags(mesh, subdomains, "subdomains")
        viskex.dolfinx.plot_mesh_tags(mesh, boundaries_and_interfaces, "boundaries and interfaces")
    except Exception:
        pass

    # ============================================================== FUNCTION SPACES ======================================================= #
    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,))) 
    Ph = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))                      
    Uh = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,))) # control space must live in the state velocity space
    Zh = Vh.clone()
    Wh = Ph.clone()

    # ============================================================== RESTRICTIONS ======================================================= #
    dofs_Vh = np.arange(Vh.dofmap.index_map.size_local + Vh.dofmap.index_map.num_ghosts)
    dofs_Ph = np.arange(Ph.dofmap.index_map.size_local + Ph.dofmap.index_map.num_ghosts)
    dofs_Zh = dofs_Vh
    dofs_Wh = dofs_Ph

    dofs_Uh = dolfinx.fem.locate_dofs_topological(
        Uh, boundaries_and_interfaces.dim, boundaries_control
    )

    rest_Vh = multiphenicsx.fem.DofMapRestriction(Vh.dofmap, dofs_Vh)
    rest_Ph = multiphenicsx.fem.DofMapRestriction(Ph.dofmap, dofs_Ph)
    rest_Uh = multiphenicsx.fem.DofMapRestriction(Uh.dofmap, dofs_Uh)
    rest_Zh = multiphenicsx.fem.DofMapRestriction(Zh.dofmap, dofs_Zh)
    rest_Wh = multiphenicsx.fem.DofMapRestriction(Wh.dofmap, dofs_Wh)

    restrictions = [rest_Vh, rest_Ph, rest_Uh, rest_Zh, rest_Wh]

    # ============================================================== TRIAL AND TEST FUNCTIONS ======================================================= #
    (v, p) = (ufl.TrialFunction(Vh), ufl.TrialFunction(Ph))
    (w, q) = (ufl.TestFunction(Vh), ufl.TestFunction(Ph))
    u = ufl.TrialFunction(Uh)
    r = ufl.TestFunction(Uh)
    (z, b) = (ufl.TrialFunction(Zh), ufl.TrialFunction(Wh))
    (s, d) = (ufl.TestFunction(Zh), ufl.TestFunction(Wh))

    # ============================================================== PROBLEM DATA ======================================================= #
    x = ufl.SpatialCoordinate(mesh)
    v_d = ufl.as_vector((
        (mixing_ratio * 10.0 * (x[1]**3 - x[1]**2 - x[1] + 1.0))
        + ((1.0 - mixing_ratio) * 10.0 * (-x[1]**3 - x[1]**2 + x[1] + 1.0)),
        0.0
    ))
    zero_scalar = petsc4py.PETSc.ScalarType(0)
    zero_vector = np.zeros((2, ), dtype=petsc4py.PETSc.ScalarType)
    ff = dolfinx.fem.Constant(mesh, zero_vector)

    def inlet_profile(points: "np.ndarray[np.float64]") -> "np.ndarray[petsc4py.PETSc.ScalarType]":
        """Poiseuille parabolic inlet velocity profile evaluator."""
        values = np.zeros((2, points.shape[1]))
        values[0, :] = 10.0 * (points[1, :] + 1.0) * (1.0 - points[1, :])
        return values

    g = dolfinx.fem.Function(Vh)
    g.interpolate(inlet_profile)
    bc0 = dolfinx.fem.Function(Vh) # zero by default

    # ============================================================== OPTIMALITY CONDITIONS FORMS ======================================================= #

    def tracking_term(v_arg: ufl.Argument, w_arg: ufl.Argument) -> ufl.core.expr.Expr:
        """Term for the observation interface on the boundary (dS)."""
        return ufl.inner(v_arg, w_arg)("-")

    def penalty_term(u_arg: ufl.Argument, r_arg: ufl.Argument) -> ufl.core.expr.Expr:
        """Regularization term for the control (ds)."""
        return reg_smooth * ufl.inner(ufl.grad(u_arg) * t, ufl.grad(r_arg) * t) + reg_l2 * ufl.inner(u_arg, r_arg)

    a = [[tracking_term(v, w) * dS(obs_tag), None, None, nu * ufl.inner(ufl.grad(z), ufl.grad(w)) * ufl.dx, - ufl.inner(b, ufl.div(w)) * ufl.dx],
          [None, None, None, - ufl.inner(ufl.div(z), q) * ufl.dx, None],
          [None, None, penalty_term(u, r) * ds(control_tag), - ufl.inner(z, r) * ds(control_tag), None],
          [nu * ufl.inner(ufl.grad(v), ufl.grad(s)) * ufl.dx, - ufl.inner(p, ufl.div(s)) * ufl.dx, - ufl.inner(u, s) * ds(control_tag), None, None],
          [- ufl.inner(ufl.div(v), d) * ufl.dx, None, None, None, None]]

    # Right-hand side vector F (5x1)
    f = [tracking_term(v_d, w) * dS(obs_tag),
          None,
          None,
          ufl.inner(ff, s) * ufl.dx,
          None]

    # Additional terms (zero on boundaries for L_vv, L_zz)
    a[0][0] += dolfinx.fem.Constant(mesh, zero_scalar) * ufl.inner(v, w) * (ds(inlet_tag) + ds(wall_tag))
    a[3][3] = dolfinx.fem.Constant(mesh, zero_scalar) * ufl.inner(z, s) * (ds(inlet_tag) + ds(wall_tag))

    # Fill None terms in F with zero forms
    f[1] = ufl.inner(dolfinx.fem.Constant(mesh, zero_scalar), q) * ufl.dx
    f[2] = ufl.inner(dolfinx.fem.Constant(mesh, zero_vector), r) * ufl.dx
    f[4] = ufl.inner(dolfinx.fem.Constant(mesh, zero_scalar), d) * ufl.dx

    # ============================================================== BOUNDARY CONDITIONS ======================================================= #
    bdofs_Y_velocity_1 = dolfinx.fem.locate_dofs_topological(
        (Vh, Vh), mesh.topology.dim - 1, boundaries_inlet)
    bdofs_Y_velocity_2 = dolfinx.fem.locate_dofs_topological(
        (Vh, Vh), mesh.topology.dim - 1, boundaries_walls)
    bdofs_Q_velocity_12 = dolfinx.fem.locate_dofs_topological(
        (Zh, Vh), mesh.topology.dim - 1, boundaries_inlet_and_walls)

    bc = [dolfinx.fem.dirichletbc(g, bdofs_Y_velocity_1, Vh),
          dolfinx.fem.dirichletbc(bc0, bdofs_Y_velocity_2, Vh),
          dolfinx.fem.dirichletbc(bc0, bdofs_Q_velocity_12, Zh)]

    # ============================================================== SOLUTION FUNCTIONS ======================================================= #
    v_opt, p_opt = dolfinx.fem.Function(Vh), dolfinx.fem.Function(Ph)
    u_opt = dolfinx.fem.Function(Uh)
    z_opt, b_opt = dolfinx.fem.Function(Zh), dolfinx.fem.Function(Wh)

    # ============================================================== COST FUNCTIONAL ======================================================= #
    J = 0.5 * tracking_term(v_opt - v_d, v_opt - v_d) * dS(obs_tag) + 0.5 * penalty_term(u_opt, u_opt) * ds(control_tag)
    J_cpp = dolfinx.fem.form(J)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }

    # ==================================================== UNCONTROLLED COST FUNCTIONAL ======================================================= #
    print("--- Solving Uncontrolled State Problem ---")
    a_state = [[ufl.replace(a[i][j], {s: w, d: q}) if a[i][j] is not None else None for j in (0, 1)] for i in (3, 4)]
    f_state = [ufl.replace(f[i], {s: w, d: q}) for i in (3, 4)]
    bc_state = [bc[0], bc[1]]
    restriction_state = [restrictions[i] for i in (0, 1)] # For v and p

    problem_state = multiphenicsx.fem.petsc.LinearProblem(
        a_state, f_state, bcs=bc_state, u=(v_opt, p_opt),
        petsc_options_prefix="stokes_neumann_control_state_", petsc_options=petsc_options,
        kind="mpi", restriction=restriction_state
    )
    problem_state.solve()
    del problem_state

    J_uncontrolled = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_cpp), op=mpi4py.MPI.SUM)
    print(f"Uncontrolled J = {J_uncontrolled:.6e}")
    try:
        viskex.dolfinx.plot_vector_field(v_opt, "uncontrolled state velocity")
    except Exception:
        pass

    # ==================================================== OPTIMAL CONTROL ======================================================= #
    print("\n--- Solving Optimal Control Problem")
    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, f, bcs=bc, u=(v_opt, p_opt, u_opt, z_opt, b_opt),
        petsc_options_prefix="stokes_neumann_control_", petsc_options=petsc_options,
        kind="mpi", restriction=restrictions
    )
    problem.solve()
    del problem

    J_controlled = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_cpp), op=mpi4py.MPI.SUM)
    print(f"Optimal J = {J_controlled:.6e}")

    try:
        viskex.dolfinx.plot_vector_field(v_opt, "state velocity", glyph_factor=1e-2)
        viskex.dolfinx.plot_scalar_field(p_opt, "state pressure")
        viskex.dolfinx.plot_vector_field(u_opt, "control", glyph_factor=1e-1)
        viskex.dolfinx.plot_vector_field(z_opt, "adjoint velocity", glyph_factor=1e-1)
    except Exception:
        pass

if __name__ == "__main__":
    main()
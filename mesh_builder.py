import dolfinx.io
import dolfinx.mesh
import gmsh
import mpi4py.MPI
import numpy as np


from parameters import (
    mesh_size, Y, X, L, B, H_1, H_2, L_1, L_2, N, M,
    inlet_tag, wall_tag, control_tag, obs_tag, mu4, mu5, mu6
)

def create_domain_and_mesh(comm: mpi4py.MPI.Comm):
    """
    Defines the geometry, creates the mesh using gmsh, and converts it
    to DOLFINx Mesh, MeshTags for subdomains, and MeshTags for boundaries.
    """
    gmsh.initialize()
    gmsh.model.add("mesh")

    # points
    p0 = gmsh.model.geo.addPoint(0.0, X, 0.0, mesh_size)
    p1 = gmsh.model.geo.addPoint(L - mu4, X, 0.0, mesh_size)
    p2 = gmsh.model.geo.addPoint(L, X, 0.0, mesh_size)
    p3 = gmsh.model.geo.addPoint(L + mu6 - L_2, H_2 + M, 0.0, mesh_size)
    p4 = gmsh.model.geo.addPoint(L + mu6, H_2, 0.0, mesh_size)
    p5 = gmsh.model.geo.addPoint(L, B, 0.0, mesh_size)
    p6 = gmsh.model.geo.addPoint(L + mu5, H_1, 0.0, mesh_size)
    p7 = gmsh.model.geo.addPoint(L + mu5 - L_1, H_1 + N, 0.0, mesh_size)
    p8 = gmsh.model.geo.addPoint(L, Y, 0.0, mesh_size)
    p9 = gmsh.model.geo.addPoint(L - mu4, Y, 0.0, mesh_size)
    p10 = gmsh.model.geo.addPoint(0.0, Y, 0.0, mesh_size)

    # lines
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p9)
    l9 = gmsh.model.geo.addLine(p9, p10)
    l10 = gmsh.model.geo.addLine(p10, p0)
    l11 = gmsh.model.geo.addLine(p1, p9)
    l12 = gmsh.model.geo.addLine(p2, p5)
    l13 = gmsh.model.geo.addLine(p5, p8)

    # subdomains
    line_loop_rectangle_left = gmsh.model.geo.addCurveLoop([l0, l11, l9, l10])
    line_loop_rectangle_right = gmsh.model.geo.addCurveLoop([l1, l12, l13, l8, -l11])
    line_loop_bifurcation_top = gmsh.model.geo.addCurveLoop([l5, l6, l7, -l13])
    line_loop_bifurcation_bottom = gmsh.model.geo.addCurveLoop([l2, l3, l4, -l12])

    rectangle_left = gmsh.model.geo.addPlaneSurface([line_loop_rectangle_left])
    rectangle_right = gmsh.model.geo.addPlaneSurface([line_loop_rectangle_right])
    bifurcation_top = gmsh.model.geo.addPlaneSurface([line_loop_bifurcation_top])
    bifurcation_bottom = gmsh.model.geo.addPlaneSurface([line_loop_bifurcation_bottom])

    gmsh.model.geo.synchronize()

    # add tags to the boundaries
    gmsh.model.addPhysicalGroup(1, [l10], inlet_tag)
    gmsh.model.setPhysicalName(1, inlet_tag, "Gamma_in")
    gmsh.model.addPhysicalGroup(1, [l0, l1, l2, l4, l5, l7, l8, l9], wall_tag)
    gmsh.model.setPhysicalName(1, wall_tag, "Gamma_walls")
    gmsh.model.addPhysicalGroup(1, [l3, l6], control_tag)
    gmsh.model.setPhysicalName(1, control_tag, "Gamma_control")
    gmsh.model.addPhysicalGroup(1, [l11], obs_tag)
    gmsh.model.setPhysicalName(1, obs_tag, "Gamma_obs")

    # add tags to the subdomains
    gmsh.model.addPhysicalGroup(2, [rectangle_left], 1)
    gmsh.model.addPhysicalGroup(2, [rectangle_right], 2)
    gmsh.model.addPhysicalGroup(2, [bifurcation_top], 3)
    gmsh.model.addPhysicalGroup(2, [bifurcation_bottom], 4)

    gmsh.model.mesh.generate(2)

    ptr = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)

    try:
        mesh, subdomains, boundaries_and_interfaces, *_ = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model,
            comm=comm,
            rank=0,
            gdim=2,
            partitioner=ptr,
        )
    finally:
        gmsh.finalize()

    if subdomains is None or boundaries_and_interfaces is None:
        raise RuntimeError("gmsh to dolfinx conversion failed. Check gmsh physical groups.")

    return mesh, subdomains, boundaries_and_interfaces
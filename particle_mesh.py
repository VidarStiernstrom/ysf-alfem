"""
Functions for creating meshing the particle domain, consisting of a rectangular
domain with a cut-out circle.
"""

__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

import gmsh
from mpi4py import MPI
from numpy import isclose
import pyvista
import dolfinx.plot as plot
from dolfinx.io import ( gmshio )

def create_mesh(rect_dim, circ_dim, n_rect, n_disk):
    """ Creates a gmsh model of the particle domain and creates a DOLFINx mesh
        
        The particle domain consists of a rectangular domain with a cut-out circle 
        (the particle). The origin is placed in the lower-left corner of the rectangle.

        Args:
            rect_dim: Tuple of floats (L,H).
                      L: rectangle length in x-direction.
                      H: rectangle length in y-direction.
            circ_dim: Triplet of floats (c_x,c_y,r).
                      c_x: x-coordinate of circle center.
                      c_y: y-coordinate of circle center.
                      r:   radius of circle.
            n_rect:   int
                      number of nodes along 1 side of the boundary.
                      #TODO n_rect should be a tuple with nodes in the x,y-dir resp.
            n_disk:   int.
                      number of nodes on circle boundary.

        Returns:
            mesh: DOLFINx mesh
            cell_tags: markers for the mesh cells 
            facet_tags: markers for the mesh facets 
    """

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    elem_order = 1 #
    alg = 6 # 

    L, H = rect_dim
    c_x, c_y, r = circ_dim
    gdim = 2
    gmsh.initialize()
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        disk = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        
        rectWithHole = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, disk)])
        gmsh.model.occ.synchronize()
        
        surface_marker = 1
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], surface_marker)
        gmsh.model.setPhysicalName(volumes[0][0], surface_marker, "surf")

        # Get boundary edges
        surface_tag = rectWithHole[0][0][1]
        curve_tags = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)

        # Separate disk and rectangle edges
        disk_edges = []
        rect_edges = []
        for dim, tag in curve_tags:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            on_rect = on_exterior_boundary(com, rect_dim)
            if on_rect:
                rect_edges.append(tag)
            else:
                disk_edges.append(tag)

        for tag in rect_edges:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_rect)

        for tag in disk_edges:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_disk)
        
        gmsh.option.setNumber("Mesh.Algorithm", alg)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(elem_order)
        gmsh.model.mesh.optimize("Netgen")
            
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    
    return mesh, cell_tags, facet_tags

def refine_mesh():
    """ Refines the existing gmsh model uniformely, and returns a new DOLFINX mesh

        Returns:
            mesh: DOLFINx mesh
            cell_tags: markers for the mesh cells 
            facet_tags: markers for the mesh facets 
    """
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gdim = 2
    gmsh.model.mesh.refine()
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    return mesh, cell_tags, facet_tags

def on_exterior_boundary(x, rect_dim):
    """ Checks whether a point is on the exterior (the rectangular) boundary of the domain.

        The check is evaluated numerically using numpy.isclose (with default tolerances.)

        Args:
            x:  array [float,float]
                The coordinates of the point
            rect_dim: Tuple of floats (L,H).
                    L: rectangle length in x-direction.
                    H: rectangle length in y-direction.
        Returns:  True if x on the boundary, False otherwise
    """
    L, H = rect_dim
    return  ( (isclose(x[0], 0) ) |
              (isclose(x[0],  L) ) |
              (isclose(x[1], 0) ) |
              (isclose(x[1],  H) ) )

def on_interior_boundary(x, circ_dim):
    """ Checks whether a point is on the interior (the circular) boundary of the domain.
        
        The check is evaluated numerically using numpy.isclose (with default tolerances.)
        
        Args:
            x:  array [float,float]
                The coordinates of the point
            circ_dim: Triplet of floats (c_x,c_y,r).
                      c_x: x-coordinate of circle center.
                      c_y: y-coordinate of circle center.
                      r:   radius of circle.
        Returns:  True if x on the boundary, False otherwise
    """
    c_x, c_y, r = circ_dim
    return  isclose((x[0] - c_x)**2 + (x[1] - c_y)**2, r**2)

def plot_mesh(msh):
    """ Plots a mesh
        
        Args:
            msh:  DOLFINx mesh
    """
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    topology, cell_types, geometry = plot.vtk_mesh(msh, msh.geometry.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.add_title("Mesh")
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        figure = plotter.screenshot("mesh.png")
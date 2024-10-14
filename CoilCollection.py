import os
import numpy as np
from mayavi import mlab
from magneticalc import API

def display_wire(wire_id, wire_points):
    x = [point[0] for point in wire_points]
    y = [point[1] for point in wire_points]
    z = [point[2] for point in wire_points]

    mlab.figure(f"{wire_id}", bgcolor=(0, 0, 0), size=(1920, 1080))
    mlab.plot3d(x, y, z, tube_radius=0.02, color=(0.3, 0.5, 0.8))
    mlab.savefig(f"Output/{wire_id}.png")

slices = 1000

n_bashar = 20  # Turns
h_bashar = 2  # Height

n_bifilar = 10  # Turns
h_bifilar = 2  # Height
d_bifilar = h_bifilar / n_bifilar / 2  # Displacement

n_bucking = 10  # Turns
h_bucking = 1  # Height
g_bucking = .5  # Gap

n_cwtha = 20  # Turns
r_cwtha = .5  # Poloidal radius
R_cwtha = 1  # Toroidal radius

n_mvg_inner = 30  # Turns of inner coil
R_mvg_inner = 1  # Toroidal radius of inner coil
w_mvg_inner = .1  # Square loop width of inner coil
h_mvg_inner = 1  # Square loop height of inner coil
n_mvg_outer = -100  # Turns of outer coil (reversed winding direction)
R_mvg_outer = 3  # Toroidal radius of outer coil
w_mvg_outer = .1  # Square loop width of outer coil
h_mvg_outer = 1  # Square loop height of outer coil

n_pancake = 10  # Turns

n_rodin = 20  # Turns
r_rodin = .5  # Poloidal radius
R_rodin = 1  # Toroidal radius
f_rodin = 5  # Poloidal frequency
F_rodin = 12  # Toroidal frequency

n_smith = 20  # Turns
h_smith = 5  # Height

n_torus = 20  # Turns
r_torus = .5  # Poloidal radius
R_torus = 1  # Toroidal radius

n_zigzag = 10  # Turns
r_zigzag = .5  # Poloidal radius
R_zigzag = 1  # Toroidal radius

def square_loop_torus_factory(R_mvg, w_mvg, h_mvg, n_mvg):
    def transform_to_torus(square, R_mvg, u):
        transformed_square = []
        for point in square:
            x, y, z = point
            # Displace by toroidal path (R_mvg is the major radius)
            displaced_x = (R_mvg + x) * np.cos(u)
            displaced_y = (R_mvg + x) * np.sin(u)
            # z is invariant, so we just apply translation and rotation
            transformed_square.append([displaced_x, displaced_y, z + y])
        return transformed_square

    # Define the 4 points of the square in local coordinates (invariant in z):
    square = [
        [w_mvg / 2, h_mvg / 2, 0],    # Top-right corner
        [-w_mvg / 2, h_mvg / 2, 0],   # Top-left corner
        [-w_mvg / 2, -h_mvg / 2, 0],  # Bottom-left corner
        [w_mvg / 2, -h_mvg / 2, 0]    # Bottom-right corner
    ]

    # Create the path by rotating and translating each loop along the torus
    wire_path = []
    for u in np.linspace(0, 2 * np.pi, n_mvg, endpoint=False) if n_mvg > 0 else np.linspace(2 * np.pi, 0, -n_mvg, endpoint=False):
        wire_path.extend(transform_to_torus(square, R_mvg, u))

    return wire_path

wires = {
    "Bashar-Anti":
        [
            (r * np.cos(a), r * np.sin(a), r * h_bashar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bashar, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bashar)
            for r, a in [
                (0, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bashar * (1 - r))
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(2 * np.pi * n_bashar, 0, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in [
                (0, 0),
            ]
        ]
        + [],
    "Bashar-Same":
        [
            (r * np.cos(a), r * np.sin(a), r * h_bashar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bashar, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bashar)
            for r, a in [
                (0, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bashar * (1 - r))
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bashar, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in [
                (0, 0),
            ]
        ]
        + [],
    "Bifilar-A":
        [
            (np.cos(a), np.sin(a), r * h_bifilar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bifilar, slices))
        ]
        + [
            (np.cos(a), np.sin(a), d_bifilar + (1 - r) * h_bifilar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(2 * np.pi * n_bifilar, 0, slices))
        ]
        + [
            (np.cos(a), np.sin(a), r * h_bifilar)
            for r, a in [
                (0, 0),
            ]
        ]
        + [],
    "Bifilar-B":
        [
            (np.cos(a), np.sin(a), r * h_bifilar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bifilar, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bifilar)
            for r, a in [
                (1.025, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), d_bifilar)
            for r, a in [
                (1.025, 0),
            ]
        ]
        + [
            (np.cos(a), np.sin(a), d_bifilar + r * h_bifilar)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bifilar, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), d_bifilar + h_bifilar)
            for r, a in [
                (1.05, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in [
                (1.05, 0),
                (1, 0),
            ]
        ]
        + [],
    "Bucking":
        [
            (np.cos(a), np.sin(a), r * h_bucking)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_bucking, slices))
        ]
        + [
            (np.cos(a), np.sin(a), r * h_bucking + (1 + g_bucking))
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(2 * np.pi * n_bucking, 0, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), h_bucking + (1 + g_bucking))
            for r, a in [
                (1.05, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in [
                (1.05, 0),
                (1, 0),
            ]
        ]
        + [],
    "CWTHA":
        [
            ((R_cwtha + r_cwtha * np.cos(v)) * np.cos(u), (R_cwtha + r_cwtha * np.cos(v)) * np.sin(u), r_cwtha * np.sin(v))
            for u, v in zip(np.linspace(0, 2 * np.pi, slices), np.linspace(0, 2 * np.pi * n_cwtha, slices))
        ]
        + [
            ((R_cwtha + r_cwtha * np.cos(v)) * np.cos(u), (R_cwtha + r_cwtha * np.cos(v)) * np.sin(u), r_cwtha * np.sin(v))
            for u, v in zip(np.linspace(0, 2 * np.pi, slices), np.linspace(2 * np.pi * n_cwtha, 0, slices))
        ]
        + [],
    "MVG":
        square_loop_torus_factory(R_mvg_inner, w_mvg_inner, h_mvg_inner, n_mvg_inner)
        + list(reversed(square_loop_torus_factory(R_mvg_outer, w_mvg_outer, h_mvg_outer, n_mvg_outer)))
        + [(R_mvg_inner, 0, h_mvg_inner / 2)]
        + [],
    "Pancake":
        [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_pancake, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), .025)
            for r, a in [
                (1, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), .025)
            for r, a in [
                (.05, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in zip(np.linspace(.05, 1.05, slices), np.linspace(0, 2 * np.pi * n_pancake, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in zip(np.linspace(1.05, 1.05, slices), np.linspace(2 * np.pi * n_pancake, 2 * np.pi * n_pancake, slices))
        ]
        + [
            (r * np.cos(a), r * np.sin(a), .05)
            for r, a in [
                (1.05, 0),
                (0, 0),
            ]
        ]
        + [
            (r * np.cos(a), r * np.sin(a), 0)
            for r, a in [
                (0, 0),
            ]
        ]
        + [],
    "Phasejumping":
        [
            ((R_zigzag + r_zigzag * np.cos(v)) * np.cos(u), (R_zigzag + r_zigzag * np.cos(v)) * np.sin(u), r_zigzag * np.sin(v))
            for u, v in zip(
                np.linspace(0, 2 * np.pi, slices),
                list(list(np.linspace(0, 2 * np.pi, slices // 2 // n_zigzag)) + list(np.linspace(2 * np.pi, 0, slices // 2 // n_zigzag))) * n_zigzag
            )
        ]
        + [],
    "Rodin":
        [
            ((R_rodin + r_rodin * np.cos(v)) * np.cos(u), (R_rodin + r_rodin * np.cos(v)) * np.sin(u), r_rodin * np.sin(v))
            for u, v in zip(np.linspace(0, 2 * np.pi * f_rodin, slices), np.linspace(0, 2 * np.pi * F_rodin, slices))
        ]
        + [],
    "Smith":
        [
            (np.cos(a), np.sin(a), r * h_smith)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_smith, slices))
        ]
        + [
            (np.cos(a), np.sin(a), (1 - r) * h_smith)
            for r, a in zip(np.linspace(0, 1, slices), np.linspace(0, 2 * np.pi * n_smith, slices))
        ]
        + [],
    "Torus":
        [
            ((R_torus + r_torus * np.cos(v)) * np.cos(u), (R_torus + r_torus * np.cos(v)) * np.sin(u), r_torus * np.sin(v))
            for u, v in zip(np.linspace(0, 2 * np.pi * n_torus, slices), np.linspace(0, 2 * np.pi, slices))
        ]
        + [],
}

with open("GALLERY.md", "w") as md_file:
    md_file.write("# Gallery of Coils\n\n")
    for wire_id, wire_points in wires.items():
        API.export_wire(f"Output/{wire_id}.txt", wire_points)
        display_wire(wire_id, wire_points)
        md_file.write(f"## {wire_id}\n\n")
        md_file.write(f"![{wire_id}]({os.path.join('Output', wire_id + '.png')})\n\n")

mlab.show()

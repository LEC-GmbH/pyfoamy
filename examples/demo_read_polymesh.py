"""Demo: reading an ascii and a binary polyMesh."""

from pathlib import Path

import numpy as np

from pyfoamy.parsing import PolyMesh

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def describe(mesh: PolyMesh, label: str) -> None:
    print(f"--- {label} ---")
    print("cells:    ", mesh.n_cells)
    print("points:   ", len(mesh.points))
    print("faces:    ", len(mesh.faces))
    print("owner:    ", len(mesh.owner))
    print("neighbour:", len(mesh.neighbour))

    print("first point:", mesh.points[0])
    print("first face: ", mesh.faces[0], "(point indices)")
    print("  as coordinates:", mesh.points[mesh.faces[0]].tolist())

    print("boundary patches:")
    for name, patch in mesh.patches.items():
        print(
            f"  {name:12s} type={patch.type:10s} "
            f"nFaces={patch.n_faces:4d} startFace={patch.start_face}"
        )
    print()


def main():
    ascii_dir = FIXTURES / "cavity_ascii" / "constant" / "polyMesh"
    binary_dir = FIXTURES / "cavity_binary" / "constant" / "polyMesh"

    ascii_mesh = PolyMesh.read(ascii_dir)
    binary_mesh = PolyMesh.read(binary_dir)

    describe(ascii_mesh, "ascii cavity mesh")
    describe(binary_mesh, "binary cavity mesh")

    # Same connectivity; points differ by ascii writePrecision.
    assert np.array_equal(ascii_mesh.owner, binary_mesh.owner)
    assert np.array_equal(ascii_mesh.faces.offsets, binary_mesh.faces.offsets)
    assert np.array_equal(ascii_mesh.faces.values, binary_mesh.faces.values)
    print("ascii and binary meshes agree on connectivity (owner/faces).")


if __name__ == "__main__":
    main()

/*---------------------------------------------------------------------------*\
    Copyright (C) 2026 LEC - Large Engines Competence Center
-------------------------------------------------------------------------------
License
    This file is part of pyfoamy's OpenFOAM extensions.

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

// pyfoamy's native mesh/field reading backend - a small extern "C"
// surface around OpenFOAM's Time/fvMesh/GeometricField classes, built
// fresh per-install by wmake. Not a pybind11 binding: pyfoamy only
// needs one-shot data extraction (field arithmetic stays in
// Python/numpy), so a hand-designed C ABI called via ctypes is
// simpler and adds no dependency.
//
// A case is opened once (pf_open_case) and kept alive as an opaque
// handle, rather than reconstructed per read, since a mesh can change
// over time (motion, mesh-to-mesh mapping, AMR, NCC stitching).
// `fvMesh::readUpdate()` (called by pf_read_mesh_at before every
// extraction) is OpenFOAM's own general mechanism for staying in sync
// with whatever changed.
//
// Every entry point must catch everything - see enableThrowingErrors()
// below for why that's not optional.
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace
{
	thread_local std::string lastError;

	// Foam::FatalError/FatalIOError call exit()/abort() by default, not
	// a catchable exception - fatal in a long-lived host process on
	// something as ordinary as a missing case directory. Flip their
	// throwExceptions() switch once so a fatal error becomes a normal,
	// catchable C++ exception instead.
	void enableThrowingErrors()
	{
		static const bool done = []
		{
			Foam::FatalError.throwExceptions();
			Foam::FatalIOError.throwExceptions();
			return true;
		}();
		(void)done;
	}

	// Time must outlive fvMesh (fvMesh holds a reference to it) -
	// member declaration order controls construction/destruction order
	// in C++, so runTime is declared first (constructed first, and
	// correspondingly destroyed last, when a PfCase is deleted).
	struct PfCase
	{
		Foam::Time runTime;
		Foam::fvMesh mesh;
		long timeIndex;

		// A mesh's IOobject *name* is its region name -
		// polyMesh::regionDir() inserts it as a path component
		// (constant/<region>/polyMesh) unless it's the default region
		// ("region0"). Anything read against `mesh` afterwards (fields
		// included) inherits the region via its own objectRegistry, so
		// nothing else in this file needs to know about regions.
		PfCase(
		    const Foam::fileName& rootPath,
		    const Foam::fileName& caseName,
		    const Foam::word& regionName
		    )
		:
		    runTime
		    (
		        Foam::Time::controlDictName,
		        rootPath,
		        caseName,
		        false // enableFunctionObjects
		        ),
		    mesh
		    (
		        Foam::IOobject
		        (
		            regionName.empty()
		              ? Foam::fvMesh::defaultRegion
		              : regionName,
		            runTime.name(),
		            runTime,
		            Foam::IOobject::MUST_READ
		            )
		        ),
		    timeIndex(0)
		{}
	};

	// scalar has no .component()/operator[] (it's a plain double, not a
	// VectorSpace) - these two overloads let extractField() below stay
	// generic over both without special-casing scalar internally.
	inline double getComponent(const Foam::scalar& v, int)
	{
		return v;
	}

	inline double getComponent(const Foam::vector& v, int c)
	{
		return v[c];
	}
}

struct PfFieldData
{
	long nComponents;  // 1 for scalar, 3 for vector

	// mass, length, time, temperature, moles, current,
	// luminousIntensity - matching dimensions.py's DimensionSet order.
	double dimensions[7];

	double* internalValues;  // nCells * nComponents, flat
	long nCells;

	char** patchNames;
	char** patchTypes;
	long* patchSizes;      // nFaces per patch
	double** patchValues;  // per patch: patchSizes[i] * nComponents
	long nPatches;
};

namespace
{
	// Every value here is fully evaluated - GeometricField construction
	// has already resolved uniform/nonuniform/field-function forms into
	// concrete per-cell/per-face values, so there's no "uniform"
	// concept left to report (unlike the pure-Python FieldValue, which
	// preserves that distinction by reading raw dictionary text).
	template<class Type, int NComponents>
	void extractField(
	    const Foam::GeometricField<Type, Foam::fvMesh, Foam::Field>& field,
	    PfFieldData* out
	    )
	{
		out->nComponents = NComponents;

		const Foam::dimensionSet& dims = field.dimensions();
		for (int i = 0; i < 7; ++i)
		{
			out->dimensions[i] = dims[i];
		}

		const Foam::Field<Type>& internal = field.primitiveField();
		out->nCells = internal.size();
		out->internalValues = new double[internal.size() * NComponents];
		forAll(internal, i)
		{
			for (int c = 0; c < NComponents; ++c)
			{
				out->internalValues[i * NComponents + c] =
				    getComponent(internal[i], c);
			}
		}

		const auto& bf = field.boundaryField();
		out->nPatches = bf.size();
		out->patchNames = new char*[bf.size()];
		out->patchTypes = new char*[bf.size()];
		out->patchSizes = new long[bf.size()];
		out->patchValues = new double*[bf.size()];
		forAll(bf, patchi)
		{
			const Foam::fvPatchField<Type>& pf = bf[patchi];
			out->patchNames[patchi] = strdup(pf.patch().name().c_str());
			out->patchTypes[patchi] = strdup(pf.type().c_str());
			out->patchSizes[patchi] = pf.size();
			double* values = new double[pf.size() * NComponents];
			forAll(pf, i)
			{
				for (int c = 0; c < NComponents; ++c)
				{
					values[i * NComponents + c] = getComponent(pf[i], c);
				}
			}
			out->patchValues[patchi] = values;
		}
	}
}

extern "C"
{

int pf_open_case(
    const char* rootPath,
    const char* caseName,
    const char* regionName,
    void** outHandle
    )
{
	enableThrowingErrors();
	try
	{
		*outHandle = new PfCase(
		    Foam::fileName(rootPath),
		    Foam::fileName(caseName),
		    Foam::word(regionName)
		    );
		return 0;
	}
	catch (const Foam::error& e)
	{
		lastError = e.message();
		return 1;
	}
	catch (const std::exception& e)
	{
		lastError = e.what();
		return 2;
	}
	catch (...)
	{
		lastError = "unknown error opening case";
		return 3;
	}
}

void pf_close_case(void* handle)
{
	delete static_cast<PfCase*>(handle);
}

struct PfMeshData
{
	double* points;
	long nPoints;

	// Compact face storage - the same "offsets + flat data" shape
	// listfile.py's own compact-list reader uses: face i's vertex
	// indices are faceData[faceOffsets[i] .. faceOffsets[i + 1]).
	long* faceOffsets;   // nFaces + 1 entries
	long* faceData;      // faceOffsets[nFaces] entries
	long nFaces;
	long faceDataCount;

	long* owner;
	long nOwner;
	long* neighbour;
	long nNeighbour;

	char** patchNames;
	char** patchTypes;
	long* patchNFaces;
	long* patchStartFaces;
	long nPatches;

	// Each patch's inGroups, space-separated.
	char** patchGroups;
	long nCells;
};

int pf_read_mesh_at(
    void* handle,
    double timeValue,
    PfMeshData* out
    )
{
	enableThrowingErrors();
	try
	{
		PfCase& c = *static_cast<PfCase*>(handle);
		c.runTime.setTime(timeValue, c.timeIndex++);
		// Change-type-agnostic: handles POINTS_MOVED (moving meshes),
		// TOPO_CHANGE (mesh-to-mesh mapping, AMR), and STITCHED (NCC)
		// the same way - re-reading whatever changed from this time
		// directory. Safe (and cheap) to call even when nothing did.
		c.mesh.readUpdate();
		Foam::fvMesh& mesh = c.mesh;

		// Gather everything into local containers first, so a failure
		// partway through (unlikely once readUpdate() itself succeeds,
		// but not impossible under e.g. std::bad_alloc) leaves *out
		// completely untouched rather than half-filled - only the
		// final copy below writes through the caller's pointer.
		const Foam::pointField& points = mesh.points();
		std::vector<double> pointsBuf(3 * points.size());
		forAll(points, i)
		{
			pointsBuf[3 * i] = points[i].x();
			pointsBuf[3 * i + 1] = points[i].y();
			pointsBuf[3 * i + 2] = points[i].z();
		}

		const Foam::faceList& faces = mesh.faces();
		std::vector<long> faceOffsets(faces.size() + 1);
		faceOffsets[0] = 0;
		forAll(faces, i)
		{
			faceOffsets[i + 1] = faceOffsets[i] + faces[i].size();
		}
		std::vector<long> faceData(faceOffsets[faces.size()]);
		{
			long pos = 0;
			forAll(faces, i)
			{
				forAll(faces[i], j)
				{
					faceData[pos++] = faces[i][j];
				}
			}
		}

		const Foam::labelList& owner = mesh.faceOwner();
		std::vector<long> ownerBuf(owner.begin(), owner.end());

		const Foam::labelList& neighbour = mesh.faceNeighbour();
		std::vector<long> neighbourBuf(neighbour.begin(), neighbour.end());

		// fvMesh::boundary() hides polyMesh::boundary() (returns the
		// fvPatch-based fvBoundaryMesh instead) - the patch name/type/
		// nFaces/startFace this function reports come from the
		// mesh-level polyBoundaryMesh, so call it explicitly qualified.
		const Foam::polyBoundaryMesh& boundary = mesh.polyMesh::boundary();
		std::vector<std::string> patchNames(boundary.size());
		std::vector<std::string> patchTypes(boundary.size());
		std::vector<long> patchNFaces(boundary.size());
		std::vector<long> patchStartFaces(boundary.size());
		std::vector<std::string> patchGroups(boundary.size());
		forAll(boundary, i)
		{
			const Foam::polyPatch& patch = boundary[i];
			patchNames[i] = patch.name();
			patchTypes[i] = patch.type();
			patchNFaces[i] = patch.size();
			patchStartFaces[i] = patch.start();
			forAll(patch.inGroups(), j)
			{
				patchGroups[i] += (j ? " " : "") + patch.inGroups()[j];
			}
		}

		// Only now, with every OpenFOAM read already having succeeded,
		// allocate the caller-owned buffers and copy in.
		out->nPoints = points.size();
		out->points = new double[pointsBuf.size()];
		std::copy(pointsBuf.begin(), pointsBuf.end(), out->points);

		out->nFaces = faces.size();
		out->faceOffsets = new long[faceOffsets.size()];
		std::copy(faceOffsets.begin(), faceOffsets.end(), out->faceOffsets);
		out->faceDataCount = faceData.size();
		out->faceData = new long[faceData.size()];
		std::copy(faceData.begin(), faceData.end(), out->faceData);

		out->nOwner = ownerBuf.size();
		out->owner = new long[ownerBuf.size()];
		std::copy(ownerBuf.begin(), ownerBuf.end(), out->owner);

		out->nNeighbour = neighbourBuf.size();
		out->neighbour = new long[neighbourBuf.size()];
		std::copy(neighbourBuf.begin(), neighbourBuf.end(), out->neighbour);

		out->nPatches = boundary.size();
		out->patchNames = new char*[boundary.size()];
		out->patchTypes = new char*[boundary.size()];
		out->patchNFaces = new long[boundary.size()];
		out->patchStartFaces = new long[boundary.size()];
		out->patchGroups = new char*[boundary.size()];
		forAll(boundary, i)
		{
			out->patchNames[i] = strdup(patchNames[i].c_str());
			out->patchTypes[i] = strdup(patchTypes[i].c_str());
			out->patchGroups[i] = strdup(patchGroups[i].c_str());
		}
		out->nCells = mesh.nCells();
		std::copy(
		    patchNFaces.begin(), patchNFaces.end(), out->patchNFaces
		    );
		std::copy(
		    patchStartFaces.begin(),
		    patchStartFaces.end(),
		    out->patchStartFaces
		    );

		return 0;
	}
	catch (const Foam::error& e)
	{
		lastError = e.message();
		return 1;
	}
	catch (const std::exception& e)
	{
		lastError = e.what();
		return 2;
	}
	catch (...)
	{
		lastError = "unknown error reading mesh";
		return 3;
	}
}

void pf_free_mesh(PfMeshData* data)
{
	delete[] data->points;
	delete[] data->faceOffsets;
	delete[] data->faceData;
	delete[] data->owner;
	delete[] data->neighbour;
	for (long i = 0; i < data->nPatches; ++i)
	{
		std::free(data->patchNames[i]);
		std::free(data->patchTypes[i]);
		std::free(data->patchGroups[i]);
	}
	delete[] data->patchNames;
	delete[] data->patchTypes;
	delete[] data->patchGroups;
	delete[] data->patchNFaces;
	delete[] data->patchStartFaces;
}

int pf_read_field(
    void* handle,
    double timeValue,
    const char* fieldName,
    PfFieldData* out
    )
{
	enableThrowingErrors();
	try
	{
		PfCase& c = *static_cast<PfCase*>(handle);
		c.runTime.setTime(timeValue, c.timeIndex++);
		c.mesh.readUpdate();

		Foam::IOobject header
		(
		    Foam::word(fieldName),
		    c.runTime.name(),
		    c.mesh,
		    Foam::IOobject::MUST_READ
		    );

		// Peeking the header (rather than guessing the type up front)
		// is the only way to know which templated GeometricField<Type>
		// to construct - C++ needs that decided at compile time, so
		// there's one branch per type this backend supports, not a
		// generic read.
		if (!header.headerOk())
		{
			lastError = std::string("field not found or unreadable: ")
			    + fieldName;
			return 4;
		}

		const Foam::word headerClass = header.headerClassName();
		if (headerClass == Foam::volScalarField::typeName)
		{
			Foam::volScalarField field(header, c.mesh);
			extractField<Foam::scalar, 1>(field, out);
		}
		else if (headerClass == Foam::volVectorField::typeName)
		{
			Foam::volVectorField field(header, c.mesh);
			extractField<Foam::vector, 3>(field, out);
		}
		else
		{
			lastError = "unsupported field class (not yet handled by "
			    "pyfoamy's native backend): " + headerClass;
			return 5;
		}

		return 0;
	}
	catch (const Foam::error& e)
	{
		lastError = e.message();
		return 1;
	}
	catch (const std::exception& e)
	{
		lastError = e.what();
		return 2;
	}
	catch (...)
	{
		lastError = "unknown error reading field";
		return 3;
	}
}

void pf_free_field(PfFieldData* data)
{
	delete[] data->internalValues;
	for (long i = 0; i < data->nPatches; ++i)
	{
		std::free(data->patchNames[i]);
		std::free(data->patchTypes[i]);
		delete[] data->patchValues[i];
	}
	delete[] data->patchNames;
	delete[] data->patchTypes;
	delete[] data->patchSizes;
	delete[] data->patchValues;
}

const char* pf_last_error()
{
	return lastError.c_str();
}

} // extern "C"

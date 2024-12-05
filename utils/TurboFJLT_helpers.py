from tqdm import trange
import numpy as np
import h5py as h5
import time
from TurboFJLT import fjlt_Matrices


class FJLT:
    def __init__(self, vec_dim, snapshots_dim, max_distortion):
        self.vec_dim = vec_dim
        self.snapshots_dim = snapshots_dim
        self.max_distortion = max_distortion
        self.embedding_dim = 8 * int(
            np.log(self.snapshots_dim) / self.max_distortion / self.max_distortion
        )
        self.sparsity = np.log(self.snapshots_dim) ** 2 / self.vec_dim
        self.__gen_matrices()

    def __str__(self):
        return (
            "State vector dimensions:        {vec_dim}\n"
            "Number of embedded snapshots:   {snapshots_dim}\n"
            "Maximum distortion:             {max_distortion}\n"
            "Embedding dimension:            {embedding_dim}\n"
            "Sparsity:                       {sparsity:.5e}\n"
            "Generating matrices took:       {perf_counter_matrices:.5f} seconds"
        ).format(**self.__dict__)

    def __gen_matrices(self):
        t0 = time.time()
        self.P, self.s, self.D = fjlt_Matrices(
            self.vec_dim, self.snapshots_dim, self.embedding_dim, self.sparsity
        )
        t1 = time.time()
        self.perf_counter_matrices = t1 - t0


# Class to read in data for turbomachinery problem
class TurboHDF5Reader:
    def __init__(self, file):
        self.file = file
        self.__load_parameters()
        return None

    def __str__(self):
        return (
            "Extracting cascade data from: {file}\n"
            "TEMPORAL\n"
            "Number of snapshots:          {num_snapshots}\n"
            "Timestep:                     {timestep:.5e}\n"
            "SPATIAL\n"
            "Number of passages:           {num_passages}\n"
            "Number of regions:            {num_regions}\n"
            "Region parameters:            {regions_params}\n"
            "State vector dimension:       {state_dim}"
        ).format(**self.__dict__)

    def __load_parameters(self):
        # The number of degrees of freedom is 4 in this case
        dofs = 4
        with h5.File(self.file, "r") as f:
            self.keys = list(f.keys())

            # Temporal parameters
            self.num_snapshots = len(self.keys) - 2
            self.timestep = (
                f[self.keys[1]].attrs["t"][0] - f[self.keys[0]].attrs["t"][0]
            )

            # Spatial parameters
            self.num_regions = len(f["/{}/field".format(self.keys[0])])
            self.num_passages = self.num_regions - 2
            self.regions_params = f["/{}/field".format(self.keys[0])].attrs["param"]
            self.state_dim = (
                (
                    (self.regions_params[0] + self.regions_params[2])
                    * (self.regions_params[3] - 1)
                    * self.num_passages
                    + self.regions_params[1] * self.regions_params[3]
                )
                * self.num_passages
                * dofs
            )

        us_region_shape = (self.regions_params[3] - 1, self.regions_params[0], 4)
        passage_region_shape = (self.regions_params[3], self.regions_params[1], 4)
        ds_region_shape = (self.regions_params[3] - 1, self.regions_params[2], 4)
        self.region_shapes = [us_region_shape]
        for p in range(self.num_passages):
            self.region_shapes.append(passage_region_shape)
        self.region_shapes.append(ds_region_shape)

        return None

    def __load_snapshot(self, h5_file, snapshot_id):
        qs = []
        for region in range(self.num_regions):
            qs.append(
                h5_file["/{}/field/{}".format(self.keys[snapshot_id], region)][
                    ()
                ].flatten()
            )
        Q = np.hstack(qs)
        return Q

    def __load_snapshot_chunk(self, snap_chunk_list):
        with h5.File(self.file, "r") as f:
            snapshots = [
                self.__load_snapshot(f, snap_ind) for snap_ind in snap_chunk_list
            ]
        return snapshots

    def __setup_chunking(self, snapshot_list, chunk_dim):
        num_full_chunks = len(snapshot_list) // chunk_dim
        self.snapshot_chunks_inds = [
            snapshot_list[i * chunk_dim : (i + 1) * chunk_dim]
            for i in range(num_full_chunks)
        ]
        if len(snapshot_list) % chunk_dim != 0:
            self.snapshot_chunks_inds.append(
                snapshot_list[num_full_chunks * chunk_dim :]
            )
        # Extra params for data
        self.chunk_dim = chunk_dim
        self.num_chunks = len(self.snapshot_chunks_inds)
        return None

    def reset_chunked_loading(self, snapshot_list, chunks_dim):
        self.q_chunk = None
        assert np.all(np.array(snapshot_list) >= 0) and np.all(
            np.array(snapshot_list) < self.num_snapshots
        ), "Index out of range in snapshot list"

        chunks_dim = (
            len(snapshot_list) if chunks_dim > len(snapshot_list) else chunks_dim
        )
        self.__setup_chunking(snapshot_list, chunks_dim)
        self.__current_index = -1
        return None

    def load_next(self):
        self.__current_index += 1

        in_chunk_index = self.__current_index % self.chunk_dim
        chunk_index = self.__current_index // self.chunk_dim

        # Load only when we move to a new chunk
        if in_chunk_index == 0:
            self.q_chunk = self.__load_snapshot_chunk(
                self.snapshot_chunks_inds[chunk_index]
            )
        return self.q_chunk[in_chunk_index]

    def load_full(self, snapshot_list):
        return np.asarray(self.__load_snapshot_chunk(snapshot_list)).T

    def load_meanflow(self, key_index=-3):
        q_mf = []
        with h5.File(self.file, "r") as f:
            for region in range(self.num_regions):
                key = list(f.keys())[key_index]
                q_mf.append(f[f"/{key}/average/{region}"][()].flatten())
            q_mf = np.hstack(q_mf)
        return q_mf

    def load_grid(self):
        grid = []
        with h5.File(self.file, "r") as f:
            for region in range(self.num_regions):
                grid.append(f["/grid/point/{}".format(region)][()])
        return grid

    def reconstruct_field(self, U_r):
        # Set to standard shape if just have column vector
        single_snapshot = False
        if len(U_r.shape) == 1:
            U_r = np.expand_dims(U_r, axis=1)
            single_snapshot = True

        # Set up array:
        Q = []
        for sp in range(U_r.shape[1]):
            Q.append([])

        def col_to_regions(u_col):
            u = []
            offset = 0
            for i, reg in enumerate(self.region_shapes):
                region_linear_dimension = reg[0] * reg[1] * reg[2]
                u.append(u_col[offset : offset + region_linear_dimension].reshape(reg))
                offset = offset + region_linear_dimension
            return u

        # Receiving array
        for sp in trange(U_r.shape[1]):
            u_col = U_r[:, sp]
            Q[sp] = col_to_regions(u_col)

        # Rectify back to single snapshot if we only supplied one
        if single_snapshot:
            U_r = U_r[:, 0]
            Q = Q[0]

        return Q


class TurboVisual:
    def __init__(self, reader):
        self.reader = reader
        self.grid = self.load_grid()
        return None

    def load_grid(self):
        return self.reader.load_grid()

    def extract_plotting_field(self, snapshot, variable):
        field = [region[:, :, variable] for region in snapshot]
        return field

    def colourmap_limits(self, field, region):
        if region not in np.arange(len(field)):
            raise ValueError("Adjust region to be 0<region<{}".format(len(field)))
        vmin = np.min(np.min(field[region]))
        vmax = np.max(np.max(field[region]))
        return vmin, vmax

    def process_complex_field(self, field, plot_type):
        if plot_type not in ["abs", "real", "imag"]:
            raise ValueError(
                "Complex plot type needs to be one of 'abs', 'real' or 'imag'"
            )
        if plot_type == "abs":
            return [np.abs(region) for region in field]
        if plot_type == "real":
            return [region.real for region in field]
        if plot_type == "imag":
            return [region.imag for region in field]
        return None

    def plot_field(
        self,
        ax,
        snapshot,
        variable,
        limits=None,
        region=None,
        plot_type=None,
        centre_colourmap=False,
        grid_offset=0,
        **kwargs,
    ):
        field = self.extract_plotting_field(snapshot, variable)

        # If the field is complex, we need to figure out which part to plot:
        if field[0][0, 0].dtype is not float:
            if plot_type is None:
                plot_type = "abs"
            print("Plotting ", plot_type, " values")
            field = self.process_complex_field(field, plot_type)

        # Determine the limits
        if limits is None:
            vmin, vmax = self.colourmap_limits(field, region if region else 0)
        else:
            vmin, vmax = limits

        # make abs(vmin)=abs(vmax)
        if centre_colourmap:
            vmax = np.max([np.abs(vmin), np.abs(vmax)])
            vmin = -vmax

        for i in range(self.reader.num_regions):
            ax.pcolormesh(
                self.grid[i][:, :, 0],
                self.grid[i][:, :, 1] + grid_offset,
                field[i],
                vmin=vmin,
                vmax=vmax,
                shading="gouraud",
                **kwargs,
            )
        ax.set_aspect("equal")

        return vmin, vmax

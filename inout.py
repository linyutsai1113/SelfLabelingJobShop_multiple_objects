import torch
import os


def standardize(input: torch.Tensor, dim: int = 0, eps: float = 1e-6):
    """
    Remove mean and divide for standard deviation (z-score).

    Args:
        input: Input tensor to be standardized.
        dim: Dimension of the standardization.
        eps: A value added to the denominator for numerical stability.
    """
    means = input.mean(dim=dim, keepdim=True)
    stds = input.std(dim=dim, keepdim=True) + eps
    return (input - means) / stds


def read_basic(f_path: str, sep: str = ' ', device: str = 'cpu'):
    """
    Load the basic information about a JSP instance.

    Structure of the file:
        1. num_jobs num_machines
        2. instance matrix (each row is a job)
        3. makespan
    """
    with open(f_path) as f:
        # Load the shape
        shape = next(f).split(sep)
        n = int(shape[0])
        m = int(shape[1])
        name = os.path.splitext(os.path.basename(f_path))[0]

        # Load the instance
        instance = torch.empty((n, 2 * m), dtype=torch.float32, device=device)
        for j in range(n):
            instance[j] = torch.tensor(
                [float(x) for x in next(f).split(sep) if x and not x.isspace()],
                device=device
            )

        # After the instance matrix there may be extra blocks:
        # - a line with "-1" followed by stage_splits (one line per job)
        # - a line with "-2" followed by due_dates (one line per job)
        # - optionally a makespan line
        stage_splits = None
        due_dates = None
        ms = 1.0

        # Read remaining lines into a list for easier parsing
        rem = [ln for ln in f]
        rem = [ln.strip() for ln in rem if ln is not None]
        idx = 0
        # Look for "-1" marker
        if idx < len(rem) and rem[idx] == '-1':
            idx += 1
            stage_splits = []
            for j in range(n):
                if idx >= len(rem):
                    raise ValueError(f"Missing stage splits for job {j} in {f_path}")
                parts = [int(x) for x in rem[idx].split() if x]
                stage_splits.append(parts)
                idx += 1

        # Look for "-2" marker (due dates)
        if idx < len(rem) and rem[idx] == '-2':
            idx += 1
            due_dates = []
            for j in range(n):
                if idx >= len(rem):
                    raise ValueError(f"Missing due dates for job {j} in {f_path}")
                parts = [int(x) for x in rem[idx].split() if x]
                due_dates.append(parts)
                idx += 1

        # If any extra non-empty line remains, try parse it as makespan or a pair "makespan, tardy".
        tardy_ref = None
        while idx < len(rem) and rem[idx] == '':
            idx += 1
        if idx < len(rem):
            s = rem[idx]
            # normalize separators (comma or whitespace)
            parts = [p for p in s.replace(',', ' ').split() if p]
            try:
                if len(parts) == 1:
                    ms = float(parts[0])
                elif len(parts) >= 2:
                    ms = float(parts[0])
                    try:
                        tardy_ref = int(float(parts[1]))
                    except Exception:
                        tardy_ref = None
            except Exception:
                # not a parsable makespan, ignore
                pass

    return name, n, m, instance, ms, stage_splits, due_dates, tardy_ref


def graph_edges(num_j: int, num_m: int, machines: torch.Tensor,
                device: str = 'cpu'):
    """
    Make the disjunctive graph as an edge list.

    Args:
        num_j: number of jobs
        num_m: number of machines
        machines_t: machine of each operation
    :return:
        The edges of the graph.
    """
    # Conjunctive arcs
    edges = [(j * num_m + m, j * num_m + m + 1) for j in range(num_j)
             for m in range(num_m - 1)]

    # Disjunctive arcs and solution
    for m in range(num_m):
        # Get the operations on the machine
        machine = (machines == m).view(-1).nonzero().squeeze(-1)

        for src_i in range(len(machine) - 1):
            for dst_i in range(src_i + 1, len(machine)):
                #
                edges.append((machine[src_i].item(), machine[dst_i].item()))
                edges.append((machine[dst_i].item(), machine[src_i].item()))

    return torch.tensor(edges, dtype=torch.long, device=device)


def extract_features(num_j: int, num_m: int, costs_t: torch.Tensor,
                     machines_t: torch.Tensor, device: str = 'cpu'):
    """
    Compute the base set of features from the instance information.

    Args:
        num_j: number of jobs
        num_m: number of machines
        costs_t: cost of each operation
        machines_t: machine of each operation
    :return:
        The set of features
    """
    q = torch.tensor([0.25, 0.5, 0.75], device=device)
    _max = costs_t.max()
    costs = costs_t / _max
    # Job-related
    feat_j = torch.quantile(costs, q, dim=1).T

    # Machine-related
    _costs = torch.empty((num_m, num_j), dtype=torch.float32, device=device)
    for m in range(num_m):
        _costs[m] = costs[machines_t == m]
    m_costs = _costs / _max
    feat_m = torch.quantile(m_costs, q, dim=1).T

    # Operation-related
    j_sum = costs.sum(dim=1, keepdims=True)
    cumsum = costs.cumsum(dim=1)
    completion = cumsum / j_sum
    remaining = (j_sum - cumsum + costs) / j_sum
    pos_j = costs.unsqueeze(-1) - feat_j.unsqueeze(1)
    pos_m = costs.unsqueeze(-1) - feat_m[machines_t]

    #
    features = torch.cat([
        feat_j.repeat_interleave(num_m, 0),
        feat_m[machines_t.view(-1)],
        costs.view(-1, 1),
        completion.view(-1, 1),
        remaining.view(-1, 1),
        pos_j.view(-1, 3),
        pos_m.view(-1, 3),
    ], dim=1)
    return features


def load_data(path, device: str = 'cpu', sep: str = ' '):
    """
    Load a JSP instance from path and return a PyTorch Data object.
    Note that the instance is loaded as a DiGraph.

    Args:
        path: The path to the input instance.
        sep: The separator between values in the input file.
        device: Either cpu or cuda.
    Return:
        Dict containing the information about the instance
    """
    # Load the instance from the instance.jsp file
    name, num_j, num_m, instance, ms, stage_splits, due_dates, tardy_ref = read_basic(path, sep, device)
    costs = instance[:, 1::2]
    machines = instance[:, :-1:2].long()

    # Make the disjunctive graph
    edges_t = graph_edges(num_j, num_m, machines, device=device)

    # Prepare the features
    x = extract_features(num_j, num_m, costs, machines, device=device)
    x = standardize(x, dim=0)

    # Make the data object of the loaded instance
    data = dict(
        name=name, path=path,
        j=num_j, m=num_m, shape=f"{num_j}x{num_m}",
        x=x.to(device),     # Features
        edge_index=edges_t.t().contiguous(),
        costs=costs,        # Rows are jobs
        machines=machines,  # Rows are jobs
        makespan=ms,        # Optional
        stage_splits=stage_splits,
        due_dates=due_dates,
        tardy_ref=tardy_ref
    )
    # If makespan is missing or left at default 1.0, compute a simple deterministic baseline
    # by scheduling jobs in numerical order (job 0 then job1 ...) to produce a reference makespan.
    try:
        if data.get('makespan', None) is None or float(data['makespan']) <= 1.0:
            num_jobs = data['j']
            num_m = data['m']
            job_ops = []
            # reconstruct job_ops as list of (machine,ptime) from costs and machines
            costs_mat = data['costs']
            machines_mat = data['machines']
            for j in range(num_jobs):
                ops = []
                for k in range(num_m):
                    m = int(machines_mat[j, k].item())
                    p = float(costs_mat[j, k].item())
                    ops.append((m, p))
                job_ops.append(ops)

            # simple schedule: process jobs in order 0..n-1, each job's ops in order
            machine_available = [0.0] * num_m
            job_finish = [0.0] * num_jobs
            for j in range(num_jobs):
                for op_idx, (m, p) in enumerate(job_ops[j]):
                    prev = job_finish[j]
                    start = max(machine_available[m], prev)
                    finish = start + p
                    job_finish[j] = finish
                    machine_available[m] = finish

            baseline_ms = max(machine_available)
            data['makespan'] = float(baseline_ms)
    except Exception:
        # If anything fails, leave makespan as-is
        pass
    return data


def load_dataset(path: str = './dataset/',
                 use_cached: bool = True,
                 shape: str = "",
                 device: str = 'cpu',
                 sep: str = ' '):
    """
    Load the dataset.

    Args:
        path: Path to the folder that contains the instances.
        use_cached: Whether to use the cached dataset.
        shape: Shape of instances to load.
        device: Either cpu or cuda.
        sep: The separator between values in the input file.
    Returns:
        instances: (list)
    """
    print(f"Loading {path} ...")
    c_path = os.path.join(path, f"cached_{shape}.pt"
                                if shape else 'cached.pt')
    if use_cached and os.path.exists(c_path):
        print(f'\tUsing {c_path} ...')
        instances = torch.load(c_path, map_location=device)
    else:
        print('\tExtracting features ...')
        instances = []
        for file in os.listdir(path):
            if file.startswith('.') or file.startswith('cached') or \
                    shape not in file:
                continue
            instances.append(load_data(os.path.join(path, file),
                                       device=device, sep=sep))
        torch.save(instances, c_path)
    print(f"Number of dataset instances = {len(instances)}")
    return instances


def load_raw(path: str = './bachmarks/'):
    """
    Read raw instances without generating features.

    Args:
        path: path to the benchmark to load
    Returns:
        The benchmark instances.
    """
    instances = []
    for file in os.listdir(path):
        if file.startswith('.') or file.startswith('cached'):
            continue

        # Load the instance from the instance.jsp file
        f_path = os.path.join(path, file)
        # read_basic may return extra fields (stage_splits, due_dates, tardy_ref)
        res = read_basic(f_path, device='cpu')
        name, num_j, num_m, instance, ms = res[0], res[1], res[2], res[3], res[4]
        costs = instance[:, 1::2]
        machines = instance[:, :-1:2].long()

        # Make the data object of the loaded instance
        instances.append(dict(
            name=name, path=f_path,
            j=num_j, m=num_m, shape=f"{num_j}x{num_m}",
            costs=costs,  # Rows are jobs
            machines=machines,  # Rows are jobs
            makespan=ms  # Optional
        ))
    return instances

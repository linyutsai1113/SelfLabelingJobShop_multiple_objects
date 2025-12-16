import torch
import random
import sampling as stg
from PointerNet import GATEncoder, MHADecoder
from argparse import ArgumentParser
from inout import load_dataset
from tqdm import tqdm
from utils import *
import os
import csv

# Training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of steps to wait before probing for improvements
PROBE_EVERY = 2500


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set: list,
               num_sols: int = 16,
               seed: int = 12345):
    """
    Test the model at the end of each epoch.

    Args:
        encoder: Encoder.
        decoder: Decoder.
        val_set: Validation set.
        num_sols: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()
    gaps_ms = ObjMeter()
    gaps_td = ObjMeter()

    # For each instance in the benchmark
    for ins in val_set:
        # Sample multiple solutions
        s, mss, tardy = stg.sampling(ins, encoder, decoder, bs=num_sols, device=device)

        # Log info: report makespan gap
        if 'makespan' in ins and ins['makespan'] is not None:
            min_ms = mss.min().item()
            min_gap = (min_ms / ins['makespan'] - 1) * 100
            gaps_ms.update(ins, min_gap)
        else:
            # No reference makespan available
            gaps_ms.update(ins, 0.0)
        # If tardy info exists, track gap (no per-instance printing)
        if tardy is not None and ins.get('tardy_ref') is not None:
            min_tardy_val = int(tardy.min().item())
            ref_tardy_val = ins.get('tardy_ref')
            gap_tardy = ((min_tardy_val - ref_tardy_val) / ref_tardy_val * 100) if ref_tardy_val > 0 else 0
            gaps_td.update(ins, gap_tardy)

    # Print stats
    avg_gap_ms = gaps_ms.avg
    avg_gap_td = gaps_td.avg
    print(f"\t\tValidation: AVG Gap (makespan)={avg_gap_ms:.3f}")
    print(gaps_ms)
    if avg_gap_td > -9999:  # Only print if tardy gaps were tracked
        print(f"\t\tValidation: AVG Gap (tardy)={avg_gap_td:.2f}")
        print(gaps_td)
    return avg_gap_ms

def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_set: list,
          val_set: list,
          epochs: int = 50,
          virtual_bs: int = 128,
          num_sols: int = 128,
          model_path: str = 'checkpoints/PointerNet.pt',
          log_csv: str = None):
    """
    Train the Pointer Network.

    Args:
        encoder: Encoder to train.
        decoder: Decoder to train.
        train_set: Training set.
        val_set: Validation set.
        epochs: Number of epochs.
        virtual_bs: Virtual batch size that gives the number of instances
            predicted before back-propagation.
        num_sols: Number of solutions to use in back-propagation.
        model_path:
    """
    frac, _best = 1. / virtual_bs, None
    size = len(train_set)
    indices = list(range(size))
    ### OPTIMIZER
    opti = torch.optim.Adam(list(_enc.parameters()) +
                            list(_dec.parameters()), lr=args.lr)
    c = torch.nn.CrossEntropyLoss(reduction='mean')
    #
    print("Training ...")
    # Prepare CSV logging if requested
    csv_file = None
    csv_writer = None
    if log_csv is not None:
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        new_file = not os.path.exists(log_csv)
        csv_file = open(log_csv, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(['epoch', 'step', 'phase', 'instance', 'selected_makespan', 'selected_tardy', 'ref_makespan', 'ref_tardy'])
    for epoch in range(epochs):
        losses = AverageMeter()
        gaps_ms = ObjMeter()  # makespan gap
        gaps_td = ObjMeter()  # tardy gap 
        random.shuffle(indices)
        cnt = 0
        # For each instance in the training set
        for idx, i in tqdm(enumerate(indices)):
            ins = train_set[i]
            cnt += 1
            # Training step (sample solutions)
            trajs, logits, mss, tardy = stg.sample_training(ins, encoder, decoder, bs=num_sols, device=device)

            # Select pseudo-label among sampled solutions
            if tardy is None:
                # Fallback to single-objective makespan selection
                ms, argmin = mss.min(-1)
                argmin = int(argmin.item())
            else:
                # Pareto-based selection (non-dominated set), tie-break by makespan
                bs_samp = mss.shape[0]
                nondom = []
                for p in range(bs_samp):
                    dominated = False
                    for q in range(bs_samp):
                        if p == q:
                            continue
                        # q dominates p?
                        if (mss[q] <= mss[p] and tardy[q] <= tardy[p]) and (mss[q] < mss[p] or tardy[q] < tardy[p]):
                            dominated = True
                            break
                    if not dominated:
                        nondom.append(p)
                if len(nondom) == 0:
                    # numerical fallback
                    ms, argmin = mss.min(-1)
                    argmin = int(argmin.item())
                elif len(nondom) == 1:
                    argmin = nondom[0]
                    ms = mss[argmin]
                else:
                    # choose one with minimal makespan among nondominated
                    nd_mss = torch.stack([mss[p] for p in nondom])
                    rel = torch.argmin(nd_mss)
                    argmin = int(nondom[int(rel.item())])
                    ms = mss[argmin]

            loss = c(logits[argmin], trajs[argmin])

            # log info
            losses.update(loss.item())
            ms_val = ms.item()
            ref_ms = ins.get('makespan', 1.0)
            gap_ms = (ms_val / ref_ms - 1) * 100
            gaps_ms.update(ins, gap_ms)  # Track makespan gap per instance
            # Track tardy gap if reference tardy available
            if tardy is not None and ins.get('tardy_ref') is not None:
                sel_tardy_val = int(tardy[argmin].item())
                ref_tardy_val = ins.get('tardy_ref')
                gap_tardy = ((sel_tardy_val - ref_tardy_val) / ref_tardy_val * 100) if ref_tardy_val > 0 else 0
                gaps_td.update(ins, gap_tardy)


            # Virtual batching for managing without masking different sizes
            loss *= frac
            loss.backward()
            # Log selected solution to CSV (training step)
            if csv_writer is not None:
                try:
                    sel_m = float(ms.item()) if hasattr(ms, 'item') else float(ms)
                except Exception:
                    sel_m = float(ms)
                sel_t = int(tardy[argmin].item()) if (tardy is not None) else ''
                ref_m = float(ins.get('makespan')) if ins.get('makespan') is not None else ''
                ref_t = int(ins.get('tardy_ref')) if ins.get('tardy_ref') is not None else ''
                csv_writer.writerow([epoch, idx, 'train', ins.get('name', ins.get('path', '')), sel_m, sel_t, ref_m, ref_t])
            if cnt == virtual_bs or idx + 1 == size:
                opti.step()
                opti.zero_grad()
                cnt = 0

            # Probe model
            if idx > 0 and idx % PROBE_EVERY == 0:
                val_gap = validation(encoder, decoder, val_set, num_sols=128)
                if _best is None or val_gap < _best:
                    _best = val_gap
                    torch.save((encoder.state_dict(), decoder), model_path)

        # ...log the running loss and avg gaps
        avg_loss = losses.avg
        avg_gap_ms = gaps_ms.avg
        avg_gap_td = gaps_td.avg
        logger.train(epoch, avg_loss, avg_gap_ms)
        print(f'\tEPOCH {epoch:02}: avg loss={losses.avg:.4f}')
        print(f"\t\tTrain: AVG Gap (makespan)={avg_gap_ms:2.3f}")
        print(gaps_ms)
        if avg_gap_td > -9999:  # Only print if tardy gaps were tracked
            print(f"\t\tTrain: AVG Gap (tardy)={avg_gap_td:.2f}")
            print(gaps_td)
        # Test model and save
        val_gap = validation(encoder, decoder, val_set, num_sols=128)
        # Record validation multi-objective stats to CSV (no per-instance printing)
        if csv_writer is not None:
            for ins in val_set:
                s, mss, tardy = stg.sampling(ins, encoder, decoder, bs=8, device=device)
                try:
                    min_idx = int(torch.argmin(mss).item())
                except Exception:
                    min_idx = 0
                sel_m = float(mss[min_idx].item())
                sel_t = int(tardy[min_idx].item()) if tardy is not None else ''
                ref_m = float(ins.get('makespan')) if ins.get('makespan') is not None else ''
                ref_t = int(ins.get('tardy_ref')) if ins.get('tardy_ref') is not None else ''
                csv_writer.writerow([epoch, 'validation', 'val', ins.get('name', ins.get('path', '')), sel_m, sel_t, ref_m, ref_t])
        logger.validation(val_gap)
        if _best is None or val_gap < _best:
            _best = val_gap
            torch.save((encoder.state_dict(), decoder), model_path)
        #
        logger.flush()
    # Close CSV file if opened
    if csv_file is not None:
        csv_file.close()

#
parser = ArgumentParser(description='PointerNet arguments for the JSP')
parser.add_argument("-data_path", type=str, default="./dataset5k",
                    required=False, help="Path to the training data.")
parser.add_argument("-model_path", type=str, required=False,
                    default=None, help="Path to the model.")
parser.add_argument("-enc_hidden", type=int, default=64, required=False,
                    help="Hidden size of the encoder.")
parser.add_argument("-enc_out", type=int, default=128, required=False,
                    help="Output size of the encoder.")
parser.add_argument("-mem_hidden", type=int, default=64, required=False,
                    help="Hidden size of the memory network.")
parser.add_argument("-mem_out", type=int, default=128, required=False,
                    help="Output size of the memory network.")
parser.add_argument("-clf_hidden", type=int, default=128, required=False,
                    help="Hidden size of the classifier.")
parser.add_argument("-lr", type=float, default=0.00002, required=False,
                    help="Learning rate in the first checkpoint.")
parser.add_argument("-epochs", type=int, default=20, required=False,
                    help="Number of epochs.")
parser.add_argument("-bs", type=int, default=16, required=False,
                    help="Virtual batch size.")
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions.")
args = parser.parse_args()
print(args)
#
run_name = f"PtrNet-BS{args.bs}-B{args.beta}"
logger = Logger(run_name)

if __name__ == '__main__':
    print(f"Using device: {device}")

    ### TRAINING and VALIDATION
    # Load train set directly on the chosen device to avoid repeated .to(device) calls
    #train_set = load_dataset(args.data_path, device=device)
    train_set = load_dataset('./training_data/5x5', device=device)
    #val_set = load_dataset('./benchmarks/validation', device=device)
    val_set = load_dataset('./validation_data/5x5', device=device)
    ### MAKE MODEL
    _enc = GATEncoder(train_set[0]['x'].shape[1],
                      hidden_size=args.enc_hidden,
                      embed_size=args.enc_out).to(device)
    _dec = MHADecoder(encoder_size=_enc.out_size,
                      context_size=stg.JobShopStates.size,
                      hidden_size=args.mem_hidden,
                      mem_size=args.mem_out,
                      clf_size=args.clf_hidden).to(device)
    # Load model if necessary
    if args.model_path is not None:
        print(f"Loading {args.model_path}.")
        m_path = f"{args.model_path}"
        _enc_w, _dec = torch.load(args.model_path, map_location=device)
        _enc.load_state_dict(_enc_w)
    else:
        m_path = f"checkpoints/{run_name}.pt"
    print(_enc)
    print(_dec)

    log_path = f"logs/{run_name}_multiobj.csv"
    train(_enc, _dec, train_set, val_set,
          epochs=args.epochs,
          virtual_bs=args.bs,
          num_sols=args.beta,
          model_path=m_path,
          log_csv=log_path)
    

from pathlib import Path

configurations = [
    (
        "train_soup3.py",
        "train_stability_clam.py",
        "[1, 1, 1, 1, 1]",
    ),
    (
        "train_soup3_dsmil.py",
        "train_stability_dsmil.py",
        "[0, 0, 0, 1, 1]",
    ),
]

for source_name, output_name, weights in configurations:
    source = Path(source_name).read_text()
    output = Path(output_name)

    if output.exists():
        raise FileExistsError(
            f"{output_name} already exists; refusing to overwrite it"
        )

    # Add the stability penalty as a command-line parameter.
    anchor = "    args = parser.parse_args()"

    addition = (
        "    parser.add_argument("
        "'--stability_lambda', type=float, default=0.0)\n"
        + anchor
    )

    assert source.count(anchor) == 1
    source = source.replace(anchor, addition, 1)

    # Make candidate shuffle seeds reproducible across runs.
    old_seeds = "seeds = np.random.randint(1, 1000, size=10)"
    new_seeds = (
        "seeds = np.random.RandomState(args.init_seed).randint("
        "1, 1000, size=args.nb_models)"
    )

    assert source.count(old_seeds) == 1
    source = source.replace(old_seeds, new_seeds, 1)

    # Replace the original candidate selection logic.
    start = "    early_aucs = np.array(early_aucs)"
    end = "    print('############################ BEST MODEL SELECTION"

    assert source.count(start) == 1
    assert source.count(end) == 1

    first = source.index(start)
    last = source.index(end, first)

    replacement = f"""    early_aucs = np.asarray(early_aucs, dtype=float)
    checkpoint_weights = np.asarray({weights}, dtype=float)
    checkpoint_weights /= checkpoint_weights.sum()

    if args.stability_lambda < 0:
        raise ValueError("stability_lambda must be nonnegative")

    mean_auc = np.average(
        early_aucs, weights=checkpoint_weights, axis=1
    )
    variance = np.average(
        (early_aucs - mean_auc[:, None]) ** 2,
        weights=checkpoint_weights,
        axis=1
    )
    std_auc = np.sqrt(variance)

    stability_scores = (
        mean_auc - args.stability_lambda * std_auc
    )

    bests = np.argsort(stability_scores)[-3:]
    best = np.argsort(stability_scores)[-1]

    np.savez(
        os.path.join(save_path, "candidate_selection.npz"),
        early_aucs=early_aucs,
        mean_auc=mean_auc,
        std_auc=std_auc,
        scores=stability_scores,
        selected=bests,
        lambda_value=args.stability_lambda,
        seed=args.init_seed,
    )

    print("STABILITY LAMBDA:", args.stability_lambda)
    print("MEAN AUC:", mean_auc)
    print("AUC STD:", std_auc)
    print("STABILITY SCORES:", stability_scores)
    print("SELECTED CANDIDATES:", bests)
"""

    source = source[:first] + replacement + source[last:]
    output.write_text(source)

    print(f"Created {output_name}")

print("Both stability-aware training scripts created.")

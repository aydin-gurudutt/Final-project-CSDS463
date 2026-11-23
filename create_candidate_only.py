from pathlib import Path

pairs = [
    ("train_repro_clam.py", "train_candidates_clam.py"),
    ("train_repro_dsmil.py", "train_candidates_dsmil.py"),
]

for source_name, output_name in pairs:
    source = Path(source_name).read_text()
    output = Path(output_name)

    if output.exists():
        raise FileExistsError(
            f"{output_name} already exists; refusing to overwrite"
        )

    # Add command-line switch.
    anchor = "    args = parser.parse_args()"
    assert source.count(anchor) == 1, source_name

    source = source.replace(
        anchor,
        "    parser.add_argument('--candidate_only', action='store_true')\n"
        + anchor,
        1
    )

    # Save the frozen initial weights once the output directory exists.
    anchor = "    os.makedirs(save_path, exist_ok=True)"
    assert source.count(anchor) == 1, source_name

    source = source.replace(
        anchor,
        anchor + "\n"
        "    torch.save(\n"
        "        {k: v.detach().cpu().clone() "
        "for k, v in init_dict.items()},\n"
        "        os.path.join(save_path, 'initial_weights.pth')\n"
        "    )\n"
        "    print('Saved initial weights')",
        1
    )

    # Save validation histories and exit before selection/fusion.
    anchor = "    print('Saved candidate checkpoints:', len(mils))"
    assert source.count(anchor) == 1, source_name

    source = source.replace(
        anchor,
        anchor + "\n\n"
        "    if args.candidate_only:\n"
        "        np.savez(\n"
        "            os.path.join(save_path, 'candidate_histories.npz'),\n"
        "            early_aucs=np.asarray(early_aucs, dtype=float),\n"
        "            seed=args.init_seed,\n"
        "            nb_models=args.nb_models,\n"
        "            nb_epochs=args.nb_epochs,\n"
        "        )\n"
        "        print('CANDIDATE GENERATION COMPLETE')\n"
        "        print('Candidate directory:', save_path)\n"
        "        return",
        1
    )

    output.write_text(source)
    print("Created:", output_name)

print("Candidate-only scripts created successfully.")

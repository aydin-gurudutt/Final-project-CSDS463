from pathlib import Path

pairs = [
    ("train_stability_clam.py", "train_repro_clam.py"),
    ("train_stability_dsmil.py", "train_repro_dsmil.py"),
]

for source_name, output_name in pairs:
    source = Path(source_name).read_text()
    output = Path(output_name)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    # Freeze the original initialization weights.
    old = "    init_dict = milnet.state_dict()"
    new = (
        "    init_dict = {\n"
        "        key: value.detach().clone()\n"
        "        for key, value in milnet.state_dict().items()\n"
        "    }"
    )

    assert source.count(old) == 1, source_name
    source = source.replace(old, new, 1)

    # Save every candidate after early training.
    anchor = "    early_aucs = np.asarray(early_aucs, dtype=float)"

    addition = (
        "    candidate_dir = os.path.join(save_path, 'candidates')\n"
        "    os.makedirs(candidate_dir, exist_ok=True)\n"
        "    for candidate_index, candidate_model in enumerate(mils):\n"
        "        torch.save(\n"
        "            candidate_model.state_dict(),\n"
        "            os.path.join(candidate_dir, "
        "f'candidate_{candidate_index}.pth')\n"
        "        )\n"
        "    print('Saved candidate checkpoints:', len(mils))\n\n"
        + anchor
    )

    assert source.count(anchor) == 1, source_name
    source = source.replace(anchor, addition, 1)

    # Allow DSMIL to use the requested number of candidate epochs.
    if "dsmil" in source_name:
        old = "    M, N = len(mil_ids), 5"
        new = "    M, N = len(mil_ids), args.nb_epochs"
        assert source.count(old) == 1
        source = source.replace(old, new, 1)

    output.write_text(source)
    print(f"Created: {output_name}")

print("Pipeline patch completed.")

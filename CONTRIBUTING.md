# Contributing

Contributions are welcome through focused issues and pull requests.

Before opening a pull request:

1. Run `python -m unittest discover -s tests -v`.
2. Run `node --check viewer/server.js` and `node --check viewer/main.js`.
3. Do not commit room captures, personal scans, model checkpoints, or generated training data.
4. Describe which camera, mask, or coordinate-system contract a pipeline change affects.
5. Preserve unknown/uncertain semantic assignments instead of silently forcing labels.

Experiments intended to support a performance claim must follow
`research/EXPERIMENT_PROTOCOL.md` and add an immutable row to
`research/experiment_registry.csv`.


# tools

Scripts for finding MPICH bugs that could become tests here. They live on the
`tools` branch only, never on `main`, so `main` stays a clean mirror of
upstream and a PR branched from it carries no tooling commits.

Nothing here is wired to run automatically, and nothing has produced a test
that has been run on Aurora yet.

## watch.py

Dumps aurora-labelled `pmodels/mpich` issues as JSON. No filtering, no state,
no credentials: it reads public issues anonymously.

    ./tools/watch.py --state open -o issues.json
    ./tools/watch.py --since 2026-07-01

## classify.py

Asks a model one question per issue: could someone write a 1- or 2-node test
of the shape in this suite from this report? Shows the model a spread of real
tests first, and requires it to quote the issue verbatim for whatever it
claims. Quotes that are not in the issue body are flagged -- that is the check
on a confident wrong answer.

    ./tools/classify.py issues.json . --dry-run --only 7880   # see the prompt
    ./tools/classify.py issues.json . --eval                  # score itself

`--eval` scores against this suite: an issue that already has a test should
come back usable, since someone demonstrably wrote one.

## propose.py

Drafts a reproducer and a bats file for one issue, writes them into a new
`1node/` or `2node/` directory, and opens a DRAFT PR on the fork.

    ./tools/propose.py issues.json . --issue 7880 --write-only   # files only
    ./tools/propose.py issues.json . --issue 7880                # branch + PR

`--write-only` first, always. The drafted test is unrun: a machine without
Aurora MPICH cannot compile it, let alone check that it reproduces anything.
The PR body says so in bold.

## Model

Both `classify.py` and `propose.py` speak OpenAI-compatible
`/v1/chat/completions`, so vLLM and llama.cpp's server work unchanged:

    vllm serve <model> --port 8000
    llama-server -m <model>.gguf --port 8000
    export MPICH_WATCH_API=http://localhost:8000/v1
    export MPICH_WATCH_MODEL=<name the server reports>

`propose.py` needs `GITHUB_TOKEN` for the PR step alone; `--no-pr` commits and
pushes over SSH without one.

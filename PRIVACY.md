# Privacy

This repository is public, and so is every commit in its history. It holds a method, templates
and a generic tool. It never holds anything that identifies a person, a machine or a network.

## What never goes in

- IP and MAC addresses, hostnames, tunnel names, private ports.
- Emails, usernames and account handles other than the repository's own, home-directory paths.
- API keys, tokens, cookies, passwords, anything token-shaped.
- Names of private repositories, real task titles, real session transcripts.
- A real config file. `config.example.toml` and the test fixtures are synthetic.

## The guard

Three layers. The first one prevents a leak; the other two notice when the first was off or
was not enough.

1. Local hooks. `tools/privacy_scan.py` is a standard-library scanner. It prints the rule
   that fired and where, never the matched value, and it fails closed: a file it cannot read
   is a finding. The hooks in `.githooks/` run it on staged files and the commit identity
   before a commit, on the commit message, and on the whole history before a push. Nothing
   else stops private content before it leaves the machine, so turn the hooks on once per
   clone:

   ```bash
   git config core.hooksPath .githooks
   ```

2. Reconcile. A fresh clone starts without the hooks, and nothing in git says so.
   `oversight reconcile` reports `guard_off` for a repository marked `public` in its config
   whose local clone has no executable pre-push hook.

3. CI. `.github/workflows/ci.yml` has two jobs: `gates` runs the same scan over the tree and
   every branch the checkout fetched (`rev-list --all`, not only the head commit's history) on
   every push and pull request, after the commit check of the change, and `tests` runs the
   tests. On a pull request `gates` checks out the head commit, not the merge commit
   GitHub makes for the run: that commit is written with the web identity of the account that
   opened the pull request, whose address is not on the allowed list. The scanner and its rules
   are taken from the base branch on a pull request, so a change cannot loosen the rules it is
   judged by (the branch is named by its full ref, so a tag of the same name cannot stand in for
   it); on a push the scan runs the scanner of the pushed commit, and the commit check,
   which runs first, reports a change to it. A change can still edit the workflow file, and
   GitHub runs the changed copy of that, on the pull request and again after the merge. The
   commit check reports such a change while the change leaves that step working; one that
   rewrites or removes the step is reported by nothing in CI, so read every change under
   `.github/` before it merges. CI runs after the content has
   reached GitHub: it catches what slipped past the hooks and what other people send, and it
   cannot undo a push.

On GitHub, also turn on secret scanning with push protection for the repository. It refuses
pushes that contain known token formats, before they land.

## Private names

The built-in rules are in `tools/privacy_rules.json`. Names that are private to you (machines,
projects, people) go in a rules file outside the repository, in the same format:
`~/.config/agent-oversight/private_rules.json`, or the path in the
`AGENT_OVERSIGHT_PRIVATE_RULES` environment variable. Set `AGENT_OVERSIGHT_REQUIRE_PRIVATE=1`
to make the scan fail when that file is missing. CI knows only the built-in rules: your
private names stay on your machine, which is one more reason the local hooks have to be on.

A clean scan is necessary and not sufficient. Read the whole diff before you commit.

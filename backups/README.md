# backups/

**Empty on purpose.**

Source code is backed up with git — commits, branches and tags. Files named `*_old.py`,
`*_new.py`, `*_final2.py` or `*_backup.py` are prohibited (see `docs/DECISIONS.md` D005).

This directory exists only for large experiment snapshots that genuinely cannot live in
git: recorded datasets, trained checkpoints, long video captures. It is git-ignored except
for this file.

If you put something here, add a row:

| Directory / file | Why it was kept | Date | Corresponding commit | Safe to delete after |
|---|---|---|---|---|
| _(nothing yet)_ | | | | |

"Safe to delete after" is not optional. A snapshot with no expiry is how a repository
turns into a junk drawer.

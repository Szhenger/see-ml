# runtime/host — the package entry point

The role where an operating system's signals and a user's intent enter the
runtime (`docs/next-project/seeai.md` §6): the generated `main`, flag
parsing against the recipe's rules (S4 #150), exit codes (0 commit, 3
reject, 1 error, 2 usage), cooperative cancellation polled once per step,
and thread-count and affinity policy. Today the entry point is
`update_main.cc`, written by the compiler's packaging phase into every
package; on a phone it becomes a framework call. S7 (#153) moves the
contract here; until then this README is the contract.

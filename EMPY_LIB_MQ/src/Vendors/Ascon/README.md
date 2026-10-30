# Ascon-AEAD128 reference

Unmodified portable reference files from the Ascon team's
[ascon-c](https://github.com/ascon/ascon-c/tree/446347f21b209f3921c65ece70027c366cbe1693/crypto_aead/asconaead128/ref),
commit `446347f21b209f3921c65ece70027c366cbe1693`, plus `tests/crypto_aead.h`.
The upstream CC0 license and implementors list are retained. The matching 1,089
known-answer vectors are in `EMPY_EXE_TESTS/data/Ascon-AEAD128.txt`.

This implements NIST SP 800-232 Ascon-AEAD128: 16-byte key, nonce and tag.
Only `aead.c` is compiled, exclusively into the client-facing EmpyMQ library.
`ASCON_PRINT_STATE` must remain undefined. These files are excluded from project
formatting and naming changes. Update source and vectors together at a pinned
revision and run the Security suite after any update.

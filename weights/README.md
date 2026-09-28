# Model weights

The trained weights for every method compared in the paper are hosted externally:

**https://www.dropbox.com/scl/fo/x7s7gckh90pung1j7zyxh/ABofwPOVd7Bl_vKC9Ld5P0E?rlkey=5y5s7kyzhe4to297s182h2q68&dl=0**

Download them into this directory. `CHECKSUMS.tsv` lists, for every file, its SHA-256, its size in
bytes, the method it belongs to and the epoch it is scored at; verify a download with

    sha256sum -c <(awk -F'\t' 'NR>1 {print $3"  "$6}' CHECKSUMS.tsv)

`eval/reproduce_comparison.py` expects the files in this directory under the names given in
`CHECKSUMS.tsv`.

## Licensing

The weights of the proposed model are derivatives of Stable Diffusion v1-5 and are distributed
under the CreativeML Open RAIL-M licence, whose use restrictions apply to them. The baseline
weights were trained from scratch and are released under the same licence as the code; see
`../LICENSE`.

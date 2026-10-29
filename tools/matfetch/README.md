# matfetch

Mirrors CC0 PBR materials and stamp brushes into a local library that zestpaint
can read directly.

Sources:

| Source                             | Assets | What                                        |
|------------------------------------|-------:|---------------------------------------------|
| [Poly Haven](https://polyhaven.com)   |    790 | Textures                                    |
| [ambientCG](https://ambientcg.com)    |  2 006 | Materials                                   |
| ambientCG                          |    126 | Decals                                      |
| ambientCG                          |     60 | Atlases                                     |
| ambientCG                          |      5 | Brushes                                     |
| procedural                         |     18 | Stitch / seam stamps (generated, see below) |
| procedural                         |      5 | Lace / net materials (generated, see below)  |

Everything is CC0. At 2K JPG the full mirror is **~55 GiB**.

## Quick start

```sh
cargo build -p matfetch --release

./target/release/matfetch sync          # build the catalog index (~1 min)
./target/release/matfetch brushes       # generate the procedural stitch/seam stamps
./target/release/matfetch lace          # generate the procedural lace materials
./target/release/matfetch link          # point ~/.zestpaint/library at the mirror
./target/release/matfetch serve         # browse and download interactively
```

`serve` opens a gallery on <http://127.0.0.1:7878> with thumbnails, search,
source/kind filters and per-asset or bulk download buttons with live progress.

To pull everything headlessly instead:

```sh
./target/release/matfetch download --all --yes
```

## Layout on disk

The library root defaults to `~/storage/materials` (`--out` to change it):

```
<root>/
  materials/<Name>/basecolor.jpg normal.jpg roughness.jpg
                   metallic.jpg height.jpg ao.jpg material.json
  brushes/<Name>/...              # decals, atlases, brushes, procedural stamps
  CREDITS.md
  .matfetch/index.json
```

Map filenames are **normalised to canonical slot names** on the way in. This
matters: `src/library.rs` matches maps by filename keyword, and neither source's
native names fully match it — Poly Haven ships `_diff_`, `_nor_gl_` and `_disp_`,
which the scanner's `diffuse` / `normal` / `displacement` checks all miss.
Normalising at download time means every folder loads in the app regardless of
where it came from.

Where a site offers several conventions for one slot, the better one wins:
OpenGL normals over DirectX, true displacement over bump.

`material.json` in each folder records the source, author, licence and page URL,
and doubles as the "already downloaded, at this resolution" marker — so runs are
resumable and re-running skips what you already have.

## Options

```
--out <dir>           Library root (default: ~/storage/materials)
--res <1k|2k|4k|8k>   Resolution (default: 2k)
--format <jpg|png>    Format (default: jpg)
--jobs <n>            Parallel downloads (default: 4)

download:
  --all                 Everything in the index
  --source <name>       polyhaven | ambientcg
  --kind <name>         material | decal | atlas | brush
  --limit <n>           Stop after n assets (trial run)
  --yes                 Skip the size confirmation
  <id>...               Explicit ids: ph:aerial_asphalt_01 acg:Ground107
```

The index stores every resolution/format combination, so switching `--res` needs
no re-sync.

## Procedural stitch stamps

Neither site publishes stitch or seam alphas — searching ambientCG for
*stitch*, *seam*, *thread*, *zipper* or *rivet* returns nothing, and Poly Haven
has no brush category. Since those are exactly the stamps you want for painting
seams onto a garment, `matfetch brushes` generates them instead:

Stitches: `stitch_straight`, `stitch_double`, `stitch_running`, `stitch_zigzag`,
`stitch_cross`, `stitch_chain`, `stitch_blanket`, `stitch_serge`, `stitch_ladder`

Seams and hardware: `seam_hem`, `seam_piping`, `seam_rivets`, `seam_grommet`,
`seam_button`, `seam_snap`, `seam_zipper`, `seam_zipper_coil`, `seam_velcro`

Each is rendered from rounded-capsule distance fields into an `opacity.png`
coverage mask and a domed `height.png`, at any `--size` (default 2048), and
tiles horizontally so a stroke dragged along a seam has no visible join.

Load one with **Brush → Tip → Load…** in the app.

## Procedural lace

Same reasoning, same answer: ambientCG returns nothing for lace, tulle,
chiffon, organza, embroidery, doily or filigree, and Poly Haven has no lace.
The hard part of lace is the *cutout*, and `matfetch lace` generates it:

`lace_floral`, `lace_geometric`, `lace_scallop`, `lace_tulle`, `lace_eyelet`

Each is a full material — `basecolor`, `opacity`, `height`, `normal`,
`roughness` — evaluated analytically per texel, so it tiles seamlessly (net
lines are integer-frequency and motifs sit on a lattice that divides the tile).
The `opacity` map is the point: it gives real holes, which is what the app's
cutout path needs.

Threads are deliberately about one texel wide at 1K, so generate at `--size
2048` or above; below roughly 512 the finer patterns wash out.

## Where else to look for brush alphas

Searched, for the record:

- **ambientCG** — 5 brush assets (wrinkles, rock). Mirrored by `download --kind brush`.
- **[3dtextures.me](https://3dtextures.me/category/alpha-brushes/)** — CC0-tagged, but
  only three relevant assets (Stitches 001, Fabric Seam 001/002), the licence page
  404s, and downloads go through Google Drive. Fine to grab by hand; not
  something to mirror automatically.
- **ArtStation / FlippedNormals / JROTools / ZBrushCentral / Pixologic** — large
  seam-and-stitch packs, free as in price but *not* CC0. They need an account and
  their terms forbid redistribution, so they cannot live in this repo. Download
  them yourself and drop them in `~/.zestpaint/library/brushes/<name>/`.

That scarcity is why the generator exists. Being SDF-based it is effectively
vector: regenerate at any `--size` and the edges stay clean.

## Notes on the sources

- Poly Haven's API rejects requests without a `User-Agent`; matfetch sets one
  identifying itself.
- ambientCG's `downloadFolders` JSON nests dicts inside lists inconsistently
  between asset types, so download descriptors are collected by walking the
  whole subtree rather than by a fixed path.
- ambientCG Brush assets are not PBR sets: some are a single unsuffixed PNG,
  others a `_masked`/`_raw` TIFF pair. The masked cut-out is preferred and
  transcoded to PNG, since nothing downstream reads TIFF.
- Concurrency defaults to 4. These are free hosts paying the bandwidth bill.

//! matfetch — mirror CC0 PBR materials and stamp brushes for zestpaint.
//!
//! Sources: Poly Haven (textures) and ambientCG (materials, decals, atlases,
//! brushes), plus a set of procedurally generated stitch/seam stamps.

mod serve;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use matfetch::{brushes, catalog, download, http, lace, load_index, save_index, splatter};

const USAGE: &str = "\
matfetch — CC0 material & brush mirror for zestpaint

USAGE:
  matfetch <command> [options]

COMMANDS:
  sync                  Refresh the catalog index from both sites
  serve                 Open the local browser gallery (default port 7878)
  download              Download assets headlessly
  brushes               Generate the procedural stitch/seam stamp brushes
  lace                  Generate procedural lace and net materials
  splatter              Generate procedural splatter stamps (splat/drip/spray/droplets/smear)
  status                Show what is on disk
  link                  Point ~/.zestpaint/library at the library root

OPTIONS:
  --out <dir>           Library root (default: ~/storage/materials)
  --res <1k|2k|4k|8k>   Resolution (default: 2k)
  --format <jpg|png>    Format (default: jpg)
  --jobs <n>            Parallel downloads (default: 4)
  --port <n>            Port for `serve` (default: 7878)

  download options:
  --all                 Every asset in the index
  --source <name>       polyhaven | ambientcg
  --kind <name>         material | decal | atlas | brush
  --limit <n>           Stop after n assets (useful for a trial run)
  --yes                 Skip the size confirmation
  <id>...               Explicit ids, e.g. ph:aerial_asphalt_01 acg:Ground107

  brushes / lace / splatter options:
  --size <px>           Output resolution (default: 2048)

EXAMPLES:
  matfetch sync
  matfetch serve
  matfetch download --all --yes
  matfetch download --kind brush --yes
  matfetch download acg:Ground107 ph:aerial_asphalt_01
";

struct Args {
    command: String,
    out: PathBuf,
    res: String,
    format: String,
    jobs: usize,
    port: u16,
    all: bool,
    yes: bool,
    source: Option<String>,
    kind: Option<String>,
    limit: Option<usize>,
    size: u32,
    ids: Vec<String>,
}

fn default_out() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("storage/materials")
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        command: String::new(),
        out: default_out(),
        res: "2k".into(),
        format: "jpg".into(),
        jobs: 4,
        port: 7878,
        all: false,
        yes: false,
        source: None,
        kind: None,
        limit: None,
        size: 2048,
        ids: Vec::new(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        let mut next = |what: &str| -> Result<String> {
            i += 1;
            argv.get(i).cloned().with_context(|| format!("{what} needs a value"))
        };
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(next("--out")?),
            "--res" => a.res = next("--res")?.to_lowercase(),
            "--format" => a.format = next("--format")?.to_lowercase(),
            "--jobs" => a.jobs = next("--jobs")?.parse()?,
            "--port" => a.port = next("--port")?.parse()?,
            "--source" => a.source = Some(next("--source")?.to_lowercase()),
            "--kind" => a.kind = Some(next("--kind")?.to_lowercase()),
            "--limit" => a.limit = Some(next("--limit")?.parse()?),
            "--size" => a.size = next("--size")?.parse()?,
            "--all" => a.all = true,
            "--yes" | "-y" => a.yes = true,
            "-h" | "--help" | "help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => bail!("unknown option {other}\n\n{USAGE}"),
            other => {
                if a.command.is_empty() {
                    a.command = other.to_string();
                } else {
                    a.ids.push(other.to_string());
                }
            }
        }
        i += 1;
    }
    if a.command.is_empty() {
        print!("{USAGE}");
        std::process::exit(0);
    }
    if !catalog::RESOLUTIONS.contains(&a.res.as_str()) {
        bail!("--res must be one of {:?}", catalog::RESOLUTIONS);
    }
    if !catalog::FORMATS.contains(&a.format.as_str()) {
        bail!("--format must be one of {:?}", catalog::FORMATS);
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let agent = http::agent();
    let variant = format!("{}-{}", args.res, args.format);
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create library root {}", args.out.display()))?;

    match args.command.as_str() {
        "sync" => {
            let index = catalog::sync(&agent)?;
            save_index(&args.out, &index)?;
        }
        "serve" => {
            let index = Arc::new(load_index(&args.out, &agent)?);
            serve::serve(args.out.clone(), index, args.port, args.jobs)?;
        }
        "brushes" => {
            let n = brushes::generate(&args.out, args.size)?;
            println!("generated {n} procedural stamps in {}/brushes", args.out.display());
        }
        "splatter" => {
            let n = splatter::generate(&args.out, args.size)?;
            println!("generated {n} splatter stamps in {}/brushes", args.out.display());
        }
        "lace" => {
            let n = lace::generate(&args.out, args.size)?;
            println!("generated {n} lace materials in {}/materials", args.out.display());
        }
        "status" => cmd_status(&args, &agent)?,
        "link" => cmd_link(&args)?,
        "download" => cmd_download(&args, &agent, &variant)?,
        other => bail!("unknown command {other}\n\n{USAGE}"),
    }
    Ok(())
}

fn cmd_status(args: &Args, agent: &ureq::Agent) -> Result<()> {
    let index = load_index(&args.out, agent)?;
    let mut have = 0usize;
    let mut bytes = 0u64;
    for a in &index.assets {
        if let Some(v) = download::installed(&args.out, a) {
            have += 1;
            bytes += a.variants.get(&v).map(|x| x.bytes).unwrap_or(0);
        }
    }
    println!("library: {}", args.out.display());
    println!("indexed: {} assets", index.assets.len());
    println!("on disk: {have} assets, ~{}", download::human(bytes));
    Ok(())
}

fn cmd_link(args: &Args) -> Result<()> {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let lib = home.join(".zestpaint/library");
    std::fs::create_dir_all(&lib)?;
    for sub in ["materials", "brushes"] {
        let link = lib.join(sub);
        let target = args.out.join(sub);
        std::fs::create_dir_all(&target)?;
        match std::fs::symlink_metadata(&link) {
            Ok(meta) if meta.is_symlink() => {
                std::fs::remove_file(&link)?;
            }
            Ok(_) => {
                eprintln!("! {} already exists and is not a symlink — left alone", link.display());
                continue;
            }
            Err(_) => {}
        }
        std::os::unix::fs::symlink(&target, &link)?;
        println!("{} -> {}", link.display(), target.display());
    }
    Ok(())
}

fn cmd_download(args: &Args, agent: &ureq::Agent, variant: &str) -> Result<()> {
    let index = load_index(&args.out, agent)?;

    let mut todo: Vec<&catalog::Asset> = if args.all || !args.ids.is_empty() {
        if args.ids.is_empty() {
            index.assets.iter().collect()
        } else {
            let mut v = Vec::new();
            for id in &args.ids {
                v.push(index.get(id).with_context(|| format!("unknown id {id}"))?);
            }
            v
        }
    } else {
        // No --all and no ids: still allow filtering down by source/kind.
        if args.source.is_none() && args.kind.is_none() {
            bail!("nothing to do: pass --all, --source, --kind, or explicit ids");
        }
        index.assets.iter().collect()
    };

    if let Some(src) = &args.source {
        todo.retain(|a| &a.source == src);
    }
    if let Some(kind) = &args.kind {
        todo.retain(|a| &a.kind == kind);
    }
    todo.retain(|a| a.variants.contains_key(variant));
    todo.retain(|a| download::installed(&args.out, a).as_deref() != Some(variant));
    if let Some(limit) = args.limit {
        todo.truncate(limit);
    }

    if todo.is_empty() {
        println!("nothing to download — everything matching is already at {variant}");
        return Ok(());
    }

    let bytes: u64 = todo.iter().filter_map(|a| a.variants.get(variant)).map(|v| v.bytes).sum();
    println!("{} assets at {variant}, {} to download", todo.len(), download::human(bytes));
    println!("into {}", args.out.display());
    if !args.yes {
        print!("proceed? [y/N] ");
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if !matches!(line.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("aborted");
            return Ok(());
        }
    }

    let ids: Vec<String> = todo.iter().map(|a| a.id.clone()).collect();
    let total = ids.len();
    drop(todo);

    let index = Arc::new(index);
    let pool = download::Pool::new(args.out.clone(), Arc::clone(&index), args.jobs);
    for id in &ids {
        pool.enqueue(id, variant);
    }
    let mut last_report = 0usize;
    while pool.pending() > 0 {
        std::thread::sleep(std::time::Duration::from_millis(700));
        let st = pool.shared.lock().unwrap();
        let done = st.values().filter(|p| p.state == download::State::Done).count();
        let failed = st.values().filter(|p| p.state == download::State::Failed).count();
        let got: u64 = st.values().map(|p| p.done_bytes).sum();
        drop(st);
        if done + failed != last_report {
            last_report = done + failed;
            eprintln!(
                "[{}/{total}] {} downloaded{}",
                done + failed,
                download::human(got),
                if failed > 0 { format!(", {failed} failed") } else { String::new() }
            );
        }
    }

    let st = pool.shared.lock().unwrap();
    let failed: Vec<_> = st
        .iter()
        .filter(|(_, p)| p.state == download::State::Failed)
        .map(|(id, p)| (id.clone(), p.error.clone().unwrap_or_default()))
        .collect();
    let done = st.values().filter(|p| p.state == download::State::Done).count();
    drop(st);

    println!("\ndone: {done}/{total}");
    if !failed.is_empty() {
        println!("failed: {}", failed.len());
        for (id, err) in failed.iter().take(20) {
            println!("  {id}: {err}");
        }
        if failed.len() > 20 {
            println!("  ... and {} more", failed.len() - 20);
        }
    }
    download::write_credits(&args.out, &index)?;
    Ok(())
}

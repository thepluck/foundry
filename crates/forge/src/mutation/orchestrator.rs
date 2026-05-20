//! Mutation testing orchestrator.
//!
//! This module coordinates the mutation testing workflow, including:
//! - Filtering source files for mutation
//! - Managing mutation handlers per file
//! - Running mutations in parallel with caching
//! - Aggregating results and reporting

use std::{path::PathBuf, sync::Arc, time::Instant};

use eyre::{Result, WrapErr};
use foundry_cli::utils::FoundryPathExt;
use foundry_common::sh_println;
use foundry_compilers::{
    Language, ProjectCompileOutput,
    compilers::multi::{MultiCompiler, MultiCompilerLanguage},
    utils::source_files_iter,
};
use foundry_config::{Config, filter::GlobMatcher};
use foundry_evm::opts::EvmOpts;

use crate::{
    cmd::test::FilterArgs,
    mutation::{
        MutationHandler, MutationProgress, MutationReporter, MutationsSummary,
        mutant::MutationResult, runner::run_mutations_parallel_with_progress,
    },
};

/// Configuration for mutation testing run.
pub struct MutationRunConfig {
    /// Paths to mutate (if empty, use all source files).
    pub mutate_paths: Vec<PathBuf>,
    /// Optional glob pattern to filter paths.
    pub mutate_path_pattern: Option<GlobMatcher>,
    /// Optional contract regex pattern to filter contracts.
    pub mutate_contract_pattern: Option<regex::Regex>,
    /// Number of parallel workers (0 = auto-detect).
    pub num_workers: usize,
    /// Whether to show progress display.
    pub show_progress: bool,
    /// Whether to output JSON (suppress all other output).
    pub json_output: bool,
    /// Test filter (`--match-test`, `--match-contract`, `--match-path`, ...)
    /// applied identically to baseline and every mutant run so they exercise
    /// the same test set.
    pub filter_args: FilterArgs,
    /// EVM isolation flag — mirrors the canonical `forge test` runner so
    /// baseline and mutant runs use the same execution model.
    pub isolate: bool,
}

impl MutationRunConfig {
    /// Determine number of workers, using auto-detection if 0.
    pub fn effective_workers(&self) -> usize {
        if self.num_workers == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            self.num_workers
        }
    }
}

/// Result of a mutation testing run.
pub struct MutationRunResult {
    /// Summary of all mutations across all files.
    pub summary: MutationsSummary,
    /// Whether the run was cancelled (e.g., Ctrl+C).
    pub cancelled: bool,
    /// Duration of the mutation testing run in seconds.
    pub duration_secs: f64,
}

/// Run mutation testing on the project.
///
/// This function encapsulates the mutation testing logic that was previously
/// in the test command. It handles:
/// - Filtering source files based on patterns
/// - Per-file mutation handling with caching
/// - Parallel mutation execution
/// - Result aggregation and reporting
pub async fn run_mutation_testing(
    config: Arc<Config>,
    output: &ProjectCompileOutput<MultiCompiler>,
    evm_opts: EvmOpts,
    mutation_config: MutationRunConfig,
) -> Result<MutationRunResult> {
    let num_workers = mutation_config.effective_workers();
    let json_output = mutation_config.json_output;

    // Compute a single digest of execution-affecting inputs (test filter,
    // isolation, fork URL/block, sender, ...) that aren't covered by the
    // source/build hash. Folded into every per-file cache key below so a
    // re-run with different `--match-test`, `--isolate`, or fork settings
    // does not silently reuse stale mutant outcomes from a previous run.
    let runtime_context_digest = runtime_context_digest(&mutation_config, &evm_opts);

    // Determine which paths to mutate
    let mutate_paths = resolve_mutate_paths(&config, output, &mutation_config)?;

    if !mutation_config.show_progress && !json_output {
        sh_println!("Running mutation tests with {} parallel workers...", num_workers)?;
    }

    let mut mutation_summary = MutationsSummary::new();
    let mut cancelled = false;
    let start_time = Instant::now();

    for path in mutate_paths {
        if !mutation_config.show_progress && !json_output {
            sh_println!("Running mutation tests for {}", path.display())?;
        }

        // Create handler for this file, optionally restricting to a subset of
        // contracts by name when --mutate-contract is provided.
        let mut handler = MutationHandler::new(path.clone(), config.clone())
            .with_runtime_context_digest(runtime_context_digest);
        if let Some(filter) = &mutation_config.mutate_contract_pattern {
            handler = handler.with_contract_filter(filter.clone());
        }
        handler.read_source_contract()?;

        // Get build ID for caching
        let build_id = output
            .artifact_ids()
            .find_map(|(id, _)| (id.source == path).then_some(id.build_id))
            .unwrap_or_default();

        // Check for cached results
        if let Some(prior) = handler.retrieve_cached_mutant_results(&build_id) {
            if !mutation_config.show_progress && !json_output {
                sh_println!("  Using cached results for {} mutants", prior.len())?;
            }
            for (mutant, status) in prior {
                match status {
                    MutationResult::Dead => handler.add_dead_mutant(mutant),
                    MutationResult::Alive => handler.add_survived_mutant(mutant),
                    MutationResult::Invalid => handler.add_invalid_mutant(mutant),
                    MutationResult::Skipped => handler.add_skipped_mutant(mutant),
                    MutationResult::TimedOut => handler.add_timed_out_mutant(mutant),
                }
            }
            mutation_summary.merge(handler.get_report());
            continue;
        }

        // Load persisted survived spans *before* generating/loading mutants so
        // they can actually steer the adaptive skip — both at AST generation
        // time (via the span filter inside `generate_ast`) and when re-using
        // a cached mutant list from a prior partial run.
        handler.retrieve_survived_spans(&build_id);

        // Generate or load cached mutants
        let mut mutants = if let Some(ms) = handler.retrieve_cached_mutants(&build_id) {
            // When loading from cache, filter out mutants whose span already
            // had a survivor in a previous run. Without this, a resumed run
            // would re-test mutations the adaptive heuristic already knows
            // are uninformative.
            ms.into_iter().filter(|m| !handler.should_skip_span(m.span)).collect()
        } else {
            handler.generate_ast(json_output).await?;
            handler.mutations.clone()
        };

        if mutants.is_empty() {
            if !mutation_config.show_progress && !json_output {
                sh_println!("  No mutants generated for {}", path.display())?;
            }
            continue;
        }

        // Sort mutations by span for optimal adaptive testing
        mutants.sort_by(|a, b| {
            a.span.lo().0.cmp(&b.span.lo().0).then_with(|| b.span.hi().0.cmp(&a.span.hi().0))
        });

        // Create progress display if enabled (not in JSON mode)
        let progress = if mutation_config.show_progress && !json_output {
            let p =
                MutationProgress::with_timeout(mutants.len(), num_workers, config.mutation.timeout);
            // Show relative path from project root
            let display_path =
                path.strip_prefix(&config.root).unwrap_or(&path).display().to_string();
            p.set_current_file(&display_path);
            Some(p)
        } else if !json_output {
            sh_println!("  Generated {} mutants, testing in parallel...", mutants.len())?;
            None
        } else {
            None
        };

        // Run mutations in parallel using isolated workspaces
        let results = run_mutations_parallel_with_progress(
            mutants.clone(),
            path.clone(),
            handler.src.clone(),
            config.clone(),
            evm_opts.clone(),
            num_workers,
            progress.clone(),
            json_output,
            mutation_config.filter_args.clone(),
            mutation_config.isolate,
        )?;

        // Collect results for caching
        let mut results_vec = Vec::with_capacity(results.len());
        for result in results {
            results_vec.push((result.mutant.clone(), result.result.clone()));
            match result.result {
                MutationResult::Dead => handler.add_dead_mutant(result.mutant),
                MutationResult::Alive => {
                    handler.mark_span_survived(result.mutant.span);
                    handler.add_survived_mutant(result.mutant);
                }
                MutationResult::Invalid => handler.add_invalid_mutant(result.mutant),
                MutationResult::Skipped => handler.add_skipped_mutant(result.mutant),
                MutationResult::TimedOut => handler.add_timed_out_mutant(result.mutant),
            }
        }

        // Detect cancellation early so we can decide whether the result set is
        // complete before persisting it. Without this guard a Ctrl+C mid-run
        // would write a *partial* results vector to the cache and the next run
        // would treat that subset as the full answer for this file.
        let file_cancelled = progress.as_ref().is_some_and(|p| p.is_cancelled());
        let complete_run = !file_cancelled && results_vec.len() == mutants.len();

        // Persist results for caching only when the run for this file is
        // complete. Partial caches are silent correctness bugs:
        //   - cancelled runs would be reloaded as authoritative
        //   - non-cancelled-but-short result vectors indicate a bug, not a hit
        // The mutants list itself is fine to persist (it's deterministic from
        // the AST + operator set) and so are survived spans (best-effort hint).
        //
        // Sort the persisted result vector by mutant span so the on-disk
        // cache is independent of rayon worker completion order; otherwise
        // the cache file changes content-hash run-to-run even when the
        // outcomes are identical, defeating diffing and reproducibility.
        results_vec.sort_by(|(a, _), (b, _)| {
            a.span.lo().0.cmp(&b.span.lo().0).then_with(|| a.span.hi().0.cmp(&b.span.hi().0))
        });
        if !mutants.is_empty() && !build_id.is_empty() {
            let _ = handler.persist_cached_mutants(&build_id, &mutants);
            if complete_run {
                let _ = handler.persist_cached_results(&build_id, &results_vec);
            }
            let _ = handler.persist_survived_spans(&build_id);
        }

        mutation_summary.merge(handler.get_report());

        // If cancelled, break out of the loop
        if file_cancelled {
            cancelled = true;
            break;
        }
    }

    // Report results
    let duration = start_time.elapsed();
    let duration_secs = duration.as_secs_f64();

    // Only show human-readable report if not in JSON mode
    if !json_output {
        MutationReporter::new().report(&mutation_summary, duration);
    }

    Ok(MutationRunResult { summary: mutation_summary, cancelled, duration_secs })
}

/// Build a single digest of inputs that affect mutant pass/fail outcomes but
/// are not already covered by the source / build hash. Cache entries should
/// be invalidated when any of these change.
///
/// Inputs folded in:
/// - test filter (`--match-test`, `--no-match-test`, `--match-contract`, `--no-match-contract`,
///   `--match-path`, `--no-match-path`)
/// - `--isolate`
/// - the entire `EvmOpts` serialized as JSON. This is intentionally broad: `EvmOpts` carries fork
///   URL/block, networks, env (chain_id, block, timestamp, basefee), gas-limit toggles, sender,
///   initial balance, and other knobs that can each individually flip a mutant from `Alive` to
///   `Dead`. Hashing the whole serialized blob trades occasional conservative invalidations for not
///   having to keep a hand-picked field list in lock-step with `EvmOpts`.
fn runtime_context_digest(mutation_config: &MutationRunConfig, evm_opts: &EvmOpts) -> u64 {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    let mut h = DefaultHasher::new();

    // Test filter
    let f = &mutation_config.filter_args;
    f.test_pattern.as_ref().map(|r| r.as_str()).hash(&mut h);
    f.test_pattern_inverse.as_ref().map(|r| r.as_str()).hash(&mut h);
    f.contract_pattern.as_ref().map(|r| r.as_str()).hash(&mut h);
    f.contract_pattern_inverse.as_ref().map(|r| r.as_str()).hash(&mut h);
    f.path_pattern.as_ref().map(|g| g.as_str()).hash(&mut h);
    f.path_pattern_inverse.as_ref().map(|g| g.as_str()).hash(&mut h);

    // Execution model
    mutation_config.isolate.hash(&mut h);

    // Fold in the full serialized `EvmOpts`. A serde failure here is harmless:
    // we just omit it from the hash — callers compare digests by equality so
    // mismatches still invalidate, and a fixed failure mode is preferable to
    // panicking inside the orchestrator.
    if let Ok(blob) = serde_json::to_vec(evm_opts) {
        blob.hash(&mut h);
    }

    h.finish()
}

/// Resolve which paths to mutate based on configuration.
///
/// Resolution order:
/// 1. Pick the *base* set of candidate files:
///    - `--mutate-path <GLOB>` → all source files matching the glob, OR
///    - explicit `--mutate PATH...` → those validated files, OR
///    - default → every Solidity file under `config.src`.
/// 2. If `--mutate-contract <REGEX>` is set, intersect the base set with files that contain at
///    least one contract whose name matches the regex. The per-file contract filter still
///    re-applies inside the handler.
fn resolve_mutate_paths(
    config: &Config,
    output: &ProjectCompileOutput<MultiCompiler>,
    mutation_config: &MutationRunConfig,
) -> Result<Vec<PathBuf>> {
    // 1. Base path set.
    let base: Vec<PathBuf> = if let Some(pattern) = &mutation_config.mutate_path_pattern {
        source_files_iter(&config.src, MultiCompilerLanguage::FILE_EXTENSIONS)
            .filter(|entry| entry.is_sol() && !entry.is_sol_test() && pattern.is_match(entry))
            .collect()
    } else if !mutation_config.mutate_paths.is_empty() {
        let root_canon =
            config.root.canonicalize().wrap_err("failed to canonicalize project root")?;
        let mut validated = Vec::with_capacity(mutation_config.mutate_paths.len());
        for path in &mutation_config.mutate_paths {
            let resolved = if path.is_relative() { config.root.join(path) } else { path.clone() };
            if !resolved.exists() {
                eyre::bail!("mutate path does not exist: {}", resolved.display());
            }
            if !resolved.is_file() {
                eyre::bail!("mutate path is not a file: {}", resolved.display());
            }
            let canon = resolved
                .canonicalize()
                .wrap_err_with(|| format!("failed to canonicalize: {}", resolved.display()))?;
            if !canon.starts_with(&root_canon) {
                eyre::bail!("mutate path is outside the project root: {}", resolved.display());
            }
            if !canon.is_sol() {
                eyre::bail!("mutate path is not a Solidity file: {}", resolved.display());
            }
            if canon.is_sol_test() {
                eyre::bail!(
                    "mutate path is a test file, not a source file: {}",
                    resolved.display()
                );
            }
            validated.push(canon);
        }
        validated
    } else {
        source_files_iter(&config.src, MultiCompilerLanguage::FILE_EXTENSIONS)
            .filter(|entry| entry.is_sol() && !entry.is_sol_test())
            .collect()
    };

    // 2. Intersect with `--mutate-contract` if set, so explicit `--mutate <paths>` combined with
    //    `--mutate-contract <regex>` does the principled thing (the listed files, restricted to
    //    those containing a matching contract) instead of silently expanding to every source file.
    let paths = if let Some(contract_pattern) = &mutation_config.mutate_contract_pattern {
        base.into_iter()
            .filter(|entry| {
                output
                    .artifact_ids()
                    .filter(|(id, _)| id.source == *entry)
                    .any(|(id, _)| contract_pattern.is_match(&id.name))
            })
            .collect()
    } else {
        base
    };

    Ok(paths)
}

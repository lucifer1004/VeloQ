//! Tiny dev tool: prints the projected wire format for every veloq
//! response type. Useful when iterating on the projector or when
//! sanity-checking what `--help` would show once PR II lands.
//!
//!   cargo run -p veloq-nsys-query --example print_wire_format -- metrics

use veloq_core::wire_format::wire_format_for;
use veloq_nsys_query::correlate::CorrelateResponse;
use veloq_nsys_query::gaps::GapsResponse;
use veloq_nsys_query::hardware::HardwareResponse;
use veloq_nsys_query::inspect::InspectResponse;
use veloq_nsys_query::metrics::MetricsResponse;
use veloq_nsys_query::ncu_command::NcuCommandResponse;
use veloq_nsys_query::search::SearchResponse;
use veloq_nsys_query::slices::SlicesResponse;
use veloq_nsys_query::stats::StatsResponse;
use veloq_nsys_query::summary::Summary;
use veloq_nsys_query::timeline::TimelineResponse;

fn dump<T: schemars::JsonSchema>(label: &str) {
    let wf = wire_format_for::<T>();
    println!("=== {label} ===");
    println!("{}", wf.render());
    println!();
}

fn main() {
    let target = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    match target.as_str() {
        "summary" => dump::<Summary>("summary"),
        "stats" => dump::<StatsResponse>("stats"),
        "search" => dump::<SearchResponse>("search"),
        "inspect" => dump::<InspectResponse>("inspect"),
        "correlate" => dump::<CorrelateResponse>("correlate"),
        "ncu-command" => dump::<NcuCommandResponse>("ncu-command"),
        "gaps" => dump::<GapsResponse>("gaps"),
        "timeline" => dump::<TimelineResponse>("timeline"),
        "slices" => dump::<SlicesResponse>("slices"),
        "hardware" => dump::<HardwareResponse>("hardware"),
        "metrics" => dump::<MetricsResponse>("metrics"),
        _ => {
            dump::<Summary>("summary");
            dump::<StatsResponse>("stats");
            dump::<SearchResponse>("search");
            dump::<InspectResponse>("inspect");
            dump::<CorrelateResponse>("correlate");
            dump::<NcuCommandResponse>("ncu-command");
            dump::<GapsResponse>("gaps");
            dump::<TimelineResponse>("timeline");
            dump::<SlicesResponse>("slices");
            dump::<HardwareResponse>("hardware");
            dump::<MetricsResponse>("metrics");
        }
    }
}

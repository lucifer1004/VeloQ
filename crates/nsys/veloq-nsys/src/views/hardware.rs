use veloq_core::tabular::{TabularView, cell_opt};
use veloq_nsys_query::hardware::HardwareResponse;

/// Hardware view — one row per GPU across every host, plus
/// per-host metadata as `meta` lines. Multi-GPU traces produce
/// one row per device; single-host single-GPU traces still get a
/// row so the table is never empty when a GPU exists. NIC and
/// system/CPU info land on `meta` to keep the row schema stable
/// (a NIC list isn't comparable in shape to the GPU table).
pub fn hardware_view(data: &HardwareResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "host",
        "gpu_id",
        "gpu_name",
        "chip",
        "compute",
        "sms",
        "vram_bytes",
        "bus",
    ]);
    for host in &data.rows {
        let host_label = host
            .system
            .as_ref()
            .and_then(|s| s.hostname.clone())
            .unwrap_or_else(|| format!("host#{:04x}", host.hw_host_id));
        if host.gpus.is_empty() {
            // Surface host-only rows so a CPU-only profile still
            // appears in the table view rather than vanishing.
            v.push_row(vec![
                host_label.clone(),
                String::new(),
                "<no GPUs>".into(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
        for gpu in &host.gpus {
            let compute = match (gpu.compute_major, gpu.compute_minor) {
                (Some(a), Some(b)) => format!("{a}.{b}"),
                _ => String::new(),
            };
            v.push_row(vec![
                host_label.clone(),
                gpu.id.to_string(),
                gpu.name.clone(),
                gpu.chip_name.clone().unwrap_or_default(),
                compute,
                cell_opt(gpu.sm_count),
                cell_opt(gpu.total_memory),
                gpu.bus_location.clone().unwrap_or_default(),
            ]);
        }

        // Per-host meta lines — system/cpu/driver/nics summarised as
        // free-text so the agent (and humans) get every signal
        // without inflating the row schema.
        if let Some(sys) = &host.system {
            if let Some(ref h) = sys.hostname {
                v.push_meta(format!("{host_label}.hostname"), h.clone());
            }
            if let Some(ref os) = sys.os_description {
                v.push_meta(format!("{host_label}.os"), os.clone());
            }
            if let Some(ref k) = sys.kernel_version {
                v.push_meta(format!("{host_label}.kernel"), k.clone());
            }
        }
        if let Some(cpu) = &host.cpu {
            v.push_meta(
                format!("{host_label}.cpu"),
                match cpu.core_count {
                    Some(n) => format!("{} ({} cores)", cpu.model, n),
                    None => cpu.model.clone(),
                },
            );
        }
        if let Some(drv) = &host.drivers {
            // Parsed CUDA version is the agent-actionable one
            // (`13.0` not `13000`); fall back to raw when the
            // string isn't an integer-encoded version.
            if let Some(parsed) = drv.cuda_version_parsed() {
                v.push_meta(format!("{host_label}.cuda"), parsed);
            } else if let Some(ref raw) = drv.cuda_driver_version {
                v.push_meta(format!("{host_label}.cuda_raw"), raw.clone());
            }
            if let Some(ref nv) = drv.nv_driver_version {
                v.push_meta(format!("{host_label}.nv_driver"), nv.clone());
            }
        }
        for nic in &host.nics {
            v.push_meta(
                format!("{host_label}.nic{}", nic.id),
                format!(
                    "{} vendor={} device={}",
                    nic.name,
                    cell_opt(nic.vendor_id),
                    cell_opt(nic.device_id)
                ),
            );
        }
    }
    v.push_meta("host_count", data.rows.len().to_string());
    v
}

use veloq_vis::VizAxis;

pub(super) fn axis_i32(axes: &[VizAxis], name: &str) -> Option<i32> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<i32>().ok())
}
pub(super) fn axis_i64(axes: &[VizAxis], name: &str) -> Option<i64> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<i64>().ok())
}

pub(super) fn axis_usize(axes: &[VizAxis], name: &str) -> Option<usize> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<usize>().ok())
}
pub(super) fn stream_track_key(process: i64, device: i32, stream: i64) -> String {
    format!("cuda-stream|pid:{process}|dev:{device}|stream:{stream}")
}

pub(super) fn device_group_track_key(process: i64, device: i32) -> String {
    format!("gpu-device|pid:{process}|dev:{device}")
}

pub(super) fn gpu_summary_track_key(process: i64, device: i32) -> String {
    format!("gpu-summary|pid:{process}|dev:{device}")
}

pub(super) fn cuda_api_track_key(process: i64) -> String {
    format!("cuda-api|pid:{process}")
}

pub(super) fn nvtx_track_key(depth: usize, process: i64, device: Option<i32>) -> String {
    match device {
        Some(device) => format!("nvtx|depth:{depth}|pid:{process}|dev:{device}"),
        None => format!("nvtx|depth:{depth}|pid:{process}"),
    }
}

pub(super) fn axis(name: &str, value: impl ToString) -> VizAxis {
    VizAxis {
        name: name.to_string(),
        value: value.to_string(),
    }
}

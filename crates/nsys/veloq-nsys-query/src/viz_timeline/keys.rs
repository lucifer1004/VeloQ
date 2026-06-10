use veloq_vis::VizAxis;

pub(super) fn axis_i32(axes: &[VizAxis], name: &str) -> Option<i32> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<i32>().ok())
}

pub(super) fn axis_usize(axes: &[VizAxis], name: &str) -> Option<usize> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<usize>().ok())
}
pub(super) fn stream_track_key(device: i32, stream: i64) -> String {
    format!("cuda-stream|dev:{device}|stream:{stream}")
}

pub(super) fn device_group_track_key(device: i32) -> String {
    format!("gpu-device|dev:{device}")
}

pub(super) fn gpu_summary_track_key(device: i32) -> String {
    format!("gpu-summary|dev:{device}")
}

pub(super) fn nvtx_track_key(depth: usize, device: Option<i32>) -> String {
    match device {
        Some(device) => format!("nvtx|depth:{depth}|dev:{device}"),
        None => format!("nvtx|depth:{depth}"),
    }
}

pub(super) fn axis(name: &str, value: impl ToString) -> VizAxis {
    VizAxis {
        name: name.to_string(),
        value: value.to_string(),
    }
}

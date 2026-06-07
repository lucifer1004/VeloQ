use veloq_core::tabular::TabularView;

pub(crate) fn push_time_window_meta(v: &mut TabularView, time_window_ns: Option<(i64, i64)>) {
    if let Some((start, end)) = time_window_ns {
        v.push_meta("time_window_ns", format!("{start}-{end}"));
    }
}

pub(crate) fn push_nvtx_scope_meta(v: &mut TabularView, nvtx_scope: Option<&str>) {
    if let Some(scope) = nvtx_scope {
        v.push_meta("nvtx_scope", scope.to_string());
    }
}

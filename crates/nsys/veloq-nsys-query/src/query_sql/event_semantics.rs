use crate::EventKind;
use crate::column_map::{self, ColumnMap};

/// Per-kind SQL facts shared across NSys query verbs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EventSemantics {
    kind: EventKind,
}

impl EventSemantics {
    pub(crate) fn new(kind: EventKind) -> Self {
        Self { kind }
    }

    pub(crate) fn kind(self) -> EventKind {
        self.kind
    }

    pub(crate) fn table(self) -> &'static str {
        self.kind.table()
    }

    pub(crate) fn label(self) -> &'static str {
        self.kind.as_str()
    }

    pub(crate) fn display_name_expr(self) -> &'static str {
        crate::kind_sql::display_name_expr(self.kind)
    }

    pub(crate) fn short_name_expr(self) -> &'static str {
        crate::kind_sql::short_name_expr(self.kind)
    }

    pub(crate) fn name_joins(self) -> &'static str {
        crate::kind_sql::name_joins(self.kind)
    }

    pub(crate) fn device_expr(self) -> &'static str {
        if self.kind.is_location_bearing() {
            crate::kind_sql::GPU_DEVICE_ID_EXPR
        } else {
            "CAST(NULL AS INTEGER)"
        }
    }

    pub(crate) fn context_expr(self) -> &'static str {
        if self.kind.is_location_bearing() {
            crate::kind_sql::GPU_CONTEXT_ID_EXPR
        } else {
            "CAST(NULL AS BIGINT)"
        }
    }

    pub(crate) fn stream_expr(self) -> &'static str {
        if self.kind.is_location_bearing() {
            crate::kind_sql::GPU_STREAM_ID_EXPR
        } else {
            "CAST(NULL AS BIGINT)"
        }
    }

    pub(crate) fn stats_bytes_expr(self) -> &'static str {
        match self.kind {
            EventKind::Memcpy | EventKind::Memset => "CAST(COALESCE(t.bytes, 0) AS BIGINT)",
            _ => "CAST(NULL AS BIGINT)",
        }
    }

    pub(crate) fn size_bytes_expr(self) -> Option<&'static str> {
        match self.kind {
            EventKind::Memcpy | EventKind::Memset => Some("CAST(t.bytes AS BIGINT)"),
            _ => None,
        }
    }

    /// `graphId` projection for the kind's table. Kernel is
    /// schema-aware: Nsight 2025.3 `--cuda-graph-trace=node` exports
    /// may omit the column entirely, in which case we project NULL
    /// (no graph attribution to report). `EventKind::Graph` reads
    /// `CUPTI_ACTIVITY_KIND_GRAPH_TRACE.graphId`, which is part of
    /// that table's core schema and stays unconditional.
    pub(crate) fn graph_id_expr(self, columns: &ColumnMap) -> String {
        match self.kind {
            EventKind::Kernel => format!(
                "CAST({} AS BIGINT)",
                column_map::maybe_col(columns, self.kind.table(), "graphId")
            ),
            EventKind::Graph => "CAST(t.graphId AS BIGINT)".to_string(),
            _ => "CAST(NULL AS BIGINT)".to_string(),
        }
    }

    /// `graphNodeId` projection for the kind's table. Schema-aware
    /// for the same reason as [`Self::graph_id_expr`]: node-mode traces may
    /// lack the column on kernel/memcpy/memset.
    pub(crate) fn graph_node_id_expr(self, columns: &ColumnMap) -> String {
        match self.kind {
            EventKind::Kernel | EventKind::Memcpy | EventKind::Memset => format!(
                "CAST({} AS BIGINT)",
                column_map::maybe_col(columns, self.kind.table(), "graphNodeId")
            ),
            _ => "CAST(NULL AS BIGINT)".to_string(),
        }
    }

    pub(crate) fn event_type_expr(self) -> &'static str {
        match self.kind {
            EventKind::Nvtx => "CAST(t.eventType AS BIGINT)",
            _ => "CAST(NULL AS BIGINT)",
        }
    }

    pub(crate) fn attributed_view(self) -> Option<&'static str> {
        match self.kind {
            EventKind::Kernel => Some(crate::nvtx_attribution::KERNEL_VIEW),
            EventKind::Memcpy => Some(crate::nvtx_attribution::MEMCPY_VIEW),
            EventKind::Memset => Some(crate::nvtx_attribution::MEMSET_VIEW),
            EventKind::Sync => Some(crate::nvtx_attribution::SYNC_VIEW),
            EventKind::Runtime => Some(crate::nvtx_attribution::RUNTIME_VIEW),
            _ => None,
        }
    }

    pub(crate) fn attribution_filter(self, alias: &str) -> Option<String> {
        self.attributed_view()
            .map(|view| crate::nvtx_attribution::filter_clause(view, alias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantics_tracks_event_kind_identity() {
        for kind in EventKind::ALL {
            let sem = EventSemantics::new(*kind);
            assert_eq!(sem.table(), kind.table());
            assert_eq!(sem.label(), kind.as_str());
        }
    }

    #[test]
    fn location_axes_are_null_for_host_only_kinds() {
        let runtime = EventSemantics::new(EventKind::Runtime);
        assert_eq!(runtime.device_expr(), "CAST(NULL AS INTEGER)");
        assert_eq!(runtime.context_expr(), "CAST(NULL AS BIGINT)");
        assert_eq!(runtime.stream_expr(), "CAST(NULL AS BIGINT)");

        let kernel = EventSemantics::new(EventKind::Kernel);
        assert_eq!(kernel.device_expr(), crate::kind_sql::GPU_DEVICE_ID_EXPR);
        assert_eq!(kernel.context_expr(), crate::kind_sql::GPU_CONTEXT_ID_EXPR);
        assert_eq!(kernel.stream_expr(), crate::kind_sql::GPU_STREAM_ID_EXPR);
    }

    #[test]
    fn attribution_views_match_supported_kinds() {
        assert_eq!(
            EventSemantics::new(EventKind::Kernel).attributed_view(),
            Some(crate::nvtx_attribution::KERNEL_VIEW)
        );
        assert_eq!(
            EventSemantics::new(EventKind::Runtime).attributed_view(),
            Some(crate::nvtx_attribution::RUNTIME_VIEW)
        );
        assert_eq!(
            EventSemantics::new(EventKind::Graph).attributed_view(),
            None
        );
        assert_eq!(EventSemantics::new(EventKind::Nvtx).attributed_view(), None);
    }

    #[test]
    fn graph_and_byte_expressions_are_kind_specific() {
        let mut columns = ColumnMap::new();
        columns.insert(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            ["graphId".to_string(), "graphNodeId".to_string()]
                .into_iter()
                .collect(),
        );
        columns.insert(
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            ["graphNodeId".to_string()].into_iter().collect(),
        );
        assert_eq!(
            EventSemantics::new(EventKind::Memcpy).stats_bytes_expr(),
            "CAST(COALESCE(t.bytes, 0) AS BIGINT)"
        );
        assert_eq!(
            EventSemantics::new(EventKind::Kernel).stats_bytes_expr(),
            "CAST(NULL AS BIGINT)"
        );
        // GRAPH_TRACE's graphId is core schema — no probing.
        assert_eq!(
            EventSemantics::new(EventKind::Graph).graph_id_expr(&ColumnMap::new()),
            "CAST(t.graphId AS BIGINT)"
        );
        assert_eq!(
            EventSemantics::new(EventKind::Kernel).graph_id_expr(&columns),
            "CAST(t.\"graphId\" AS BIGINT)"
        );
        assert_eq!(
            EventSemantics::new(EventKind::Memcpy).graph_node_id_expr(&columns),
            "CAST(t.\"graphNodeId\" AS BIGINT)"
        );
    }

    #[test]
    fn graph_expressions_project_null_when_columns_absent() {
        let columns = ColumnMap::new();
        assert_eq!(
            EventSemantics::new(EventKind::Kernel).graph_id_expr(&columns),
            "CAST(NULL AS BIGINT)"
        );
        assert_eq!(
            EventSemantics::new(EventKind::Kernel).graph_node_id_expr(&columns),
            "CAST(NULL AS BIGINT)"
        );
        assert_eq!(
            EventSemantics::new(EventKind::Memset).graph_node_id_expr(&columns),
            "CAST(NULL AS BIGINT)"
        );
    }
}

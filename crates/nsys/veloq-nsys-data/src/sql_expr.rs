//! DuckDB SQL snippets for NSys parquet integer boundary handling.
//!
//! NSys parquetdir exports preserve unsigned physical types for some
//! opaque identifiers (`globalTid`, `globalVid`, `vmId`,
//! `correlationId`, ...). DuckDB refuses a plain `CAST(x AS BIGINT)`
//! when a `UBIGINT` value has the high bit set, but veloq's existing
//! wire and sidecar contracts store `globalTid`-style bitfields as
//! `i64` with the same two's-complement bit pattern. Keep that
//! conversion explicit at the SQL boundary.

/// Interpret an unsigned-or-signed 64-bit SQL expression as an `i64`
/// with the same bit pattern.
///
/// The expression is intended for static internal column references,
/// never user input. NULL stays NULL so callers can continue reading
/// `Option<i64>`.
pub fn u64_bits_to_i64(expr: &str) -> String {
    format!(
        "CASE \
         WHEN {expr} IS NULL THEN NULL \
         WHEN CAST({expr} AS HUGEINT) > 9223372036854775807 \
         THEN CAST(CAST({expr} AS HUGEINT) - 18446744073709551616 AS BIGINT) \
         ELSE CAST({expr} AS BIGINT) \
         END"
    )
}

/// Project a SQL expression as an unsigned decimal string.
///
/// Use this for opaque identifiers that veloq stores as `u64` on the
/// Rust side (`globalVid`, `vmId`, NIC GUIDs). Reading them through
/// `String` avoids driver-level signed narrowing.
pub fn u64_decimal_string(expr: &str) -> String {
    format!("CAST({expr} AS VARCHAR)")
}

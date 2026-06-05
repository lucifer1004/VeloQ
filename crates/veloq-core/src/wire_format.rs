//! Project a `#[derive(JsonSchema)]` Rust type into a compact
//! TS-shorthand wire-format description, suitable for embedding in
//! clap `--help` long_about text and auto-generated reference docs.
//!
//! Design — **SSOT for response shape**. The serde Serialize impl on
//! a response struct (e.g. `StatsResponse`) defines the actual JSON
//! veloq emits. With [`schemars::JsonSchema`] derived on the same
//! type, we get a strict JSON Schema for free; this module walks that
//! schema and renders a human-friendly compact form so the answer to
//! "what does `veloq stats` return on the wire?" comes from the
//! struct itself, not a hand-maintained doc file.
//!
//! ## Output shape
//!
//! [`WireFormat::render`] produces text like:
//!
//! ```text
//! { count: int, total_matched: int, rows: StatsRow[] }
//!
//! StatsRow:
//!   { type: kernel|memcpy|memset|sync|graph,
//!     name: string, short_name: string?, device_id: int?,
//!     count: int, total_ns: int, avg_ns: int, … }
//! ```
//!
//! Conventions:
//! - `T?` — `Option<T>` or a field flagged `skip_serializing_if`.
//! - `T[]` — `Vec<T>` (and any other array-shaped serialisation).
//! - `int` — every Rust integer width; the wire format doesn't
//!   distinguish `i32` from `i64`. `float` for `f32`/`f64`. `bool`,
//!   `string`, `null` as expected.
//! - Nested object types are emitted as named references, with one
//!   definition per type, in declaration order.
//! - `#[serde(tag = "type")]` tagged unions become a union of
//!   `{ type: "kernel"|"memcpy"|… }` plus a per-variant breakout in the
//!   definitions section.
//!
//! ## What the projector does NOT do
//!
//! It does not render `#[doc]` comments inline (the text already lives
//! in the schema's `description` field) and does not emit JSON Schema
//! directly (`schemars::schema_for!` gives the strict form). veloq's
//! response structs share a tight idiom (plain structs of primitives,
//! `Option`, `Vec`, nested structs, and the occasional tagged-union
//! enum); the projector covers that idiom and bails verbosely on
//! anything exotic so a future contributor can decide how to render it.

use schemars::{JsonSchema, schema_for};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// One projected wire-format description. `root` is the compact
/// rendering of the top-level type; `definitions` carries every
/// nested struct/enum the root references, in declaration order so
/// the rendered output reads top-down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFormat {
    pub root: String,
    pub definitions: Vec<(String, String)>,
}

impl WireFormat {
    /// Render as a single multi-line text block, suitable for
    /// embedding in clap `long_about` or a markdown code fence. The
    /// root sits at the top; named definitions follow, separated
    /// from the root by a blank line and each prefixed by `<Name>:`.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.root);
        for (name, body) in &self.definitions {
            out.push_str("\n\n");
            out.push_str(name);
            out.push_str(":\n  ");
            out.push_str(&body.replace('\n', "\n  "));
        }
        out
    }
}

/// Project `T`'s schema into a `WireFormat`. The schema is built via
/// `schemars::schema_for!(T)`; `T` must `#[derive(JsonSchema)]`.
pub fn wire_format_for<T: JsonSchema>() -> WireFormat {
    let schema = schema_for!(T);
    // Schema in schemars 1.x is `serde_json::Value`-shaped.
    let root_value: Value = schema.into();
    project_root(&root_value)
}

/// Project an already-built schema value. Useful for tests + when
/// schemars's macro form isn't convenient (e.g. dynamic types). The
/// root document must conform to schemars's `$defs`-style emit.
pub fn project_root(root_value: &Value) -> WireFormat {
    let defs: BTreeMap<String, Value> = root_value
        .get("$defs")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let mut projector = Projector {
        defs: &defs,
        emitted: BTreeMap::new(),
        emission_order: Vec::new(),
        visiting: BTreeSet::new(),
    };
    let root = projector.project(root_value);
    let mut definitions = Vec::new();
    for name in &projector.emission_order {
        if let Some(body) = projector.emitted.remove(name) {
            definitions.push((name.clone(), body));
        }
    }
    WireFormat { root, definitions }
}

struct Projector<'a> {
    defs: &'a BTreeMap<String, Value>,
    /// Type name → rendered body. Populated lazily as types are
    /// referenced from the root.
    emitted: BTreeMap<String, String>,
    /// First-reference order, so `render()` writes types in the
    /// order they appear top-down (more intuitive than alphabetical
    /// when the root references one type that references another).
    emission_order: Vec<String>,
    /// Cycle guard. If a definition references itself transitively
    /// (none of veloq's response structs do today, but the projector
    /// should still terminate), we emit a `Self` placeholder.
    visiting: BTreeSet<String>,
}

impl Projector<'_> {
    /// Top-level project: turn a schema fragment into a compact
    /// string. Recurses into nested objects / arrays / unions as
    /// needed, queuing definitions for emission.
    fn project(&mut self, schema: &Value) -> String {
        // `$ref` to a named definition: emit the reference, ensure
        // the target gets rendered into `emitted`.
        if let Some(refstr) = schema.get("$ref").and_then(|v| v.as_str()) {
            return self.project_ref(refstr);
        }

        // Union / tagged-union via `oneOf` or `anyOf`. schemars emits
        // tagged enums as `oneOf` with each variant carrying a
        // `properties.type.const` discriminator.
        if let Some(variants) = schema.get("oneOf").and_then(|v| v.as_array()) {
            return self.project_union(variants);
        }
        if let Some(variants) = schema.get("anyOf").and_then(|v| v.as_array()) {
            return self.project_union(variants);
        }

        // `enum` keyword (string enums without an explicit tag): list
        // the literals.
        if let Some(values) = schema.get("enum").and_then(|v| v.as_array()) {
            return project_enum_literals(values);
        }
        if let Some(c) = schema.get("const") {
            return format_literal(c);
        }

        // Primitive / object / array via `type` keyword. schemars
        // sometimes emits an array of types (e.g. for `Option<T>`:
        // `type: ["string", "null"]`); handle that as a nullable.
        if let Some(t) = schema.get("type") {
            return self.project_typed(schema, t);
        }

        // Empty schema / unknown — agents should rarely see this on
        // a serde-Serialize type but render a placeholder so the
        // output never looks broken.
        "any".to_string()
    }

    fn project_typed(&mut self, schema: &Value, t: &Value) -> String {
        match t {
            Value::String(s) => self.project_typed_str(schema, s),
            Value::Array(arr) => {
                // ["T", "null"] → nullable T (Option / skip_serializing_if).
                let non_null: Vec<&str> = arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .filter(|s| *s != "null")
                    .collect();
                let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
                match (non_null.as_slice(), has_null) {
                    ([single], true) => {
                        // Force a synthetic single-type schema so we
                        // reuse the same primitive projection path.
                        // Build via the object API instead of `Value::IndexMut`
                        // so we don't trip the no-indexing lint on what is
                        // really a property mutation.
                        let mut synth = schema.clone();
                        if let Some(obj) = synth.as_object_mut() {
                            obj.insert("type".to_string(), Value::String((*single).into()));
                        }
                        let inner = self.project(&synth);
                        format!("{inner}?")
                    }
                    _ => {
                        // Unusual: multi-non-null union. Emit literal.
                        non_null.join("|")
                    }
                }
            }
            _ => "any".to_string(),
        }
    }

    fn project_typed_str(&mut self, schema: &Value, t: &str) -> String {
        match t {
            "object" => self.project_object(schema),
            "array" => {
                // schemars renders fixed-shape tuples (`(i64, i64)`,
                // `[i64; 3]`) as arrays with `prefixItems`: a per-slot
                // schema list. Render those as `[T1, T2, …]` so the
                // shape is visible. Vec<T>-shaped arrays fall back to
                // the homogeneous `items` projection (`T[]`).
                if let Some(prefix) = schema.get("prefixItems").and_then(|v| v.as_array()) {
                    let slots: Vec<String> = prefix.iter().map(|s| self.project(s)).collect();
                    return format!("[{}]", slots.join(", "));
                }
                let item = schema
                    .get("items")
                    .map(|v| self.project(v))
                    .unwrap_or_else(|| "any".to_string());
                format!("{item}[]")
            }
            "string" => "string".to_string(),
            "integer" => "int".to_string(),
            "number" => "float".to_string(),
            "boolean" => "bool".to_string(),
            "null" => "null".to_string(),
            _ => "any".to_string(),
        }
    }

    fn project_object(&mut self, schema: &Value) -> String {
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let required: BTreeSet<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        if props.is_empty() {
            // Could be a "passthrough" object with `additionalProperties`
            // or empty schema — render as `{}` for transparency.
            return "{}".to_string();
        }

        let mut parts: Vec<String> = Vec::with_capacity(props.len());
        for (name, sub) in &props {
            let inner = self.project(sub);
            // Two sources of optionality:
            //   - field not in `required` (e.g. `skip_serializing_if`)
            //   - inner type is nullable (`Option<T>` → `T?`)
            // Both true is the common case for `#[serde(skip_serializing_if)]
            // pub field: Option<T>` and we want a single `?`, not
            // `field?: T?`. Pick the outer marker when present and
            // strip a redundant inner one.
            let is_optional_field = !required.contains(name.as_str());
            let (rendered_inner, inner_was_nullable) = match inner.strip_suffix('?') {
                Some(stripped) => (stripped.to_string(), true),
                None => (inner, false),
            };
            let suffix_on_value = if !is_optional_field && inner_was_nullable {
                "?" // only inner was nullable — keep that marker on the value
            } else {
                ""
            };
            let suffix_on_name = if is_optional_field { "?" } else { "" };
            parts.push(format!(
                "{name}{suffix_on_name}: {rendered_inner}{suffix_on_value}"
            ));
        }
        format!("{{ {} }}", parts.join(", "))
    }

    /// `$ref: "#/$defs/TypeName"` → render the target into `emitted`
    /// (lazily, exactly once) and return its short name.
    fn project_ref(&mut self, refstr: &str) -> String {
        let name = refstr.rsplit('/').next().unwrap_or(refstr).to_string();
        if self.emitted.contains_key(&name) || self.visiting.contains(&name) {
            return name;
        }
        let target = match self.defs.get(&name) {
            Some(s) => s.clone(),
            None => return name, // unresolved ref — best effort
        };
        self.visiting.insert(name.clone());
        // Pre-register name in emission_order so referenced-from-multiple
        // types still appear in their first-discovered order.
        self.emission_order.push(name.clone());
        let body = self.project(&target);
        self.emitted.insert(name.clone(), body);
        self.visiting.remove(&name);
        name
    }

    /// `oneOf` / `anyOf` projection. veloq's serde-tagged unions
    /// (e.g. `EventDetails`) generate variants that each carry a
    /// `properties.type.const` literal — fold those into a single
    /// `{ type: A|B|C, … }` row. For untagged unions, just join the
    /// projected variants with `|`.
    fn project_union(&mut self, variants: &[Value]) -> String {
        // Try to extract a common tag/const discriminator.
        let mut tag_field: Option<String> = None;
        let mut tag_values: Vec<String> = Vec::new();
        let mut tag_extractable = true;
        for v in variants {
            let Some(props) = v.get("properties").and_then(|p| p.as_object()) else {
                tag_extractable = false;
                break;
            };
            // Find the single property whose value is a `const`.
            let mut found_tag = None;
            for (pname, pschema) in props {
                if let Some(c) = pschema.get("const").and_then(|c| c.as_str()) {
                    if found_tag.is_some() {
                        tag_extractable = false;
                        break;
                    }
                    found_tag = Some((pname.clone(), c.to_string()));
                }
            }
            match found_tag {
                Some((field, value)) => {
                    if let Some(existing) = &tag_field
                        && existing != &field
                    {
                        tag_extractable = false;
                        break;
                    }
                    tag_field = Some(field);
                    tag_values.push(value);
                }
                None => {
                    tag_extractable = false;
                    break;
                }
            }
        }

        if tag_extractable && let Some(tag) = tag_field {
            // Render per-tag-value → variant-body mapping so agents
            // can look up "if type='kernel' then the data shape is
            // KernelDetails" without a second roundtrip. Each
            // variant's body comes from `project(v)`:
            //   * `$ref` variants (e.g. `Kernel(KernelDetails)`) project
            //     to just the ref name (`KernelDetails`); the actual
            //     fields land as a top-level named definition via the
            //     normal $ref walk.
            //   * inline-struct variants (e.g. `NotFound { row_id }`)
            //     project to a `{ … }` block.
            // Avoids the old "VariantN: KernelDetails" wrapper noise
            // — those wrappers were pure indirection because the ref
            // name *is* the per-variant struct name.
            let mut mapping_lines: Vec<String> = Vec::with_capacity(variants.len());
            for (i, v) in variants.iter().enumerate() {
                let body = self.project(v);
                let tag_value = tag_values
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("variant{i}"));
                mapping_lines.push(format!("    \"{tag_value}\" → {body}"));
            }
            let values_joined = tag_values
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join("|");
            return format!(
                "{{ {tag}: {values_joined}, …per-variant fields:\n{}\n  }}",
                mapping_lines.join("\n")
            );
        }

        // Untagged: project each, joined by `|`.
        let parts: Vec<String> = variants.iter().map(|v| self.project(v)).collect();
        parts.join("|")
    }
}

fn project_enum_literals(values: &[Value]) -> String {
    let strs: Vec<String> = values
        .iter()
        .map(|v| match v {
            Value::String(s) => format!("\"{s}\""),
            other => other.to_string(),
        })
        .collect();
    strs.join("|")
}

fn format_literal(c: &Value) -> String {
    match c {
        Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(JsonSchema)]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    struct Flat {
        count: i64,
        ratio: f64,
        name: String,
        ok: bool,
    }

    #[test]
    fn flat_primitives() {
        let wf = wire_format_for::<Flat>();
        assert!(wf.root.contains("count: int"), "got `{}`", wf.root);
        assert!(wf.root.contains("ratio: float"));
        assert!(wf.root.contains("name: string"));
        assert!(wf.root.contains("ok: bool"));
        assert!(wf.definitions.is_empty(), "flat type needs no defs");
    }

    #[derive(JsonSchema)]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    struct WithOption {
        always: i64,
        sometimes: Option<String>,
    }

    #[test]
    fn option_becomes_question_mark() {
        let wf = wire_format_for::<WithOption>();
        assert!(wf.root.contains("always: int"));
        assert!(wf.root.contains("sometimes?: string"), "got `{}`", wf.root);
    }

    #[derive(JsonSchema)]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    struct WithVec {
        rows: Vec<i64>,
        labels: Vec<String>,
    }

    #[test]
    fn vec_becomes_brackets() {
        let wf = wire_format_for::<WithVec>();
        assert!(wf.root.contains("rows: int[]"), "got `{}`", wf.root);
        assert!(wf.root.contains("labels: string[]"));
    }

    #[derive(JsonSchema)]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    struct Inner {
        a: i64,
        b: String,
    }
    #[derive(JsonSchema)]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    struct Outer {
        single: Inner,
        many: Vec<Inner>,
    }

    #[test]
    fn nested_struct_emitted_as_definition() -> anyhow::Result<()> {
        let wf = wire_format_for::<Outer>();
        assert!(wf.root.contains("single: Inner"), "got `{}`", wf.root);
        assert!(wf.root.contains("many: Inner[]"));
        let inner_def = wf
            .definitions
            .iter()
            .find(|(n, _)| n == "Inner")
            .map(|(_, body)| body.clone())
            .ok_or_else(|| anyhow::anyhow!("expected Inner definition"))?;
        assert!(inner_def.contains("a: int"), "got `{inner_def}`");
        assert!(inner_def.contains("b: string"));
        Ok(())
    }

    #[test]
    fn render_assembles_root_and_definitions() {
        let wf = wire_format_for::<Outer>();
        let r = wf.render();
        assert!(r.starts_with('{'), "root first; got:\n{r}");
        assert!(r.contains("\n\nInner:\n  "), "definition prefix; got:\n{r}");
    }

    #[derive(serde::Serialize, JsonSchema)]
    #[serde(tag = "type", rename_all = "lowercase")]
    #[expect(
        dead_code,
        reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
    )]
    enum Tagged {
        Kernel { name: String },
        Memcpy { bytes: i64 },
    }

    #[test]
    fn tagged_union_collapses_to_type_field() -> anyhow::Result<()> {
        // For a `#[serde(tag = "type")]` enum, schemars emits each
        // variant with `properties.type = { const: "kernel" }`. The
        // projector should detect the shared discriminator and fold
        // into `{ type: "kernel"|"memcpy", … }` form.
        #[derive(JsonSchema)]
        #[expect(
            dead_code,
            reason = "JsonSchema-only fixture: fields are read by `schemars`'s derive at compile time, not by Rust code"
        )]
        struct Holder {
            event: Tagged,
        }
        let wf = wire_format_for::<Holder>();
        // The root contains a ref to Tagged.
        assert!(wf.root.contains("event: Tagged"), "got `{}`", wf.root);
        let tagged_def = wf
            .definitions
            .iter()
            .find(|(n, _)| n == "Tagged")
            .map(|(_, b)| b.clone())
            .ok_or_else(|| anyhow::anyhow!("expected Tagged definition"))?;
        assert!(
            tagged_def.contains("type:") && tagged_def.contains("\"kernel\""),
            "got `{tagged_def}`"
        );
        assert!(tagged_def.contains("\"memcpy\""));
        Ok(())
    }
}

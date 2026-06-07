//! Structural assertion: every PyTorch response's primary `rows[]`
//! element carries a `key` field. This mirrors the NSys/NCU smoke tests
//! and guards the canonical list contract for new PyTorch verbs/rows.

use veloq_pytorch_query::{
    CollectivesResponse, CorrelateResponse, InspectResponse, PrepResponse, SearchResponse,
    SlicesResponse, StatsResponse, SummaryResponse, TimelineResponse,
};

fn check_rows_have_key<T: schemars::JsonSchema>() -> anyhow::Result<()> {
    let type_name = std::any::type_name::<T>();
    let root = serde_json::Value::from(schemars::schema_for!(T));
    let defs = root
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let resolve = |v: &serde_json::Value| -> serde_json::Value {
        let Some(refstr) = v.get("$ref").and_then(serde_json::Value::as_str) else {
            return v.clone();
        };
        let name = refstr.rsplit('/').next().unwrap_or(refstr);
        defs.get(name).cloned().unwrap_or_else(|| v.clone())
    };

    let rows = root
        .get("properties")
        .and_then(|properties| properties.get("rows"))
        .ok_or_else(|| anyhow::anyhow!("{type_name}: schema missing rows property"))?;
    let items = rows
        .get("items")
        .ok_or_else(|| anyhow::anyhow!("{type_name}: rows.items missing"))?;
    let item = resolve(items);

    let variants = item
        .get("oneOf")
        .or_else(|| item.get("anyOf"))
        .and_then(serde_json::Value::as_array);
    if let Some(variants) = variants {
        for (i, variant) in variants.iter().enumerate() {
            let resolved = resolve(variant);
            if resolved
                .get("properties")
                .and_then(|properties| properties.get("key"))
                .is_none()
            {
                anyhow::bail!("{type_name}: rows[] variant #{i} lacks a `key` field");
            }
        }
        return Ok(());
    }

    if item
        .get("properties")
        .and_then(|properties| properties.get("key"))
        .is_none()
    {
        anyhow::bail!("{type_name}: rows[] item lacks a `key` field");
    }
    Ok(())
}

#[test]
fn every_primary_rows_item_carries_key() -> anyhow::Result<()> {
    check_rows_have_key::<SummaryResponse>()?;
    check_rows_have_key::<SearchResponse>()?;
    check_rows_have_key::<InspectResponse>()?;
    check_rows_have_key::<StatsResponse>()?;
    check_rows_have_key::<CorrelateResponse>()?;
    check_rows_have_key::<TimelineResponse>()?;
    check_rows_have_key::<SlicesResponse>()?;
    check_rows_have_key::<CollectivesResponse>()?;
    check_rows_have_key::<PrepResponse>()?;
    Ok(())
}

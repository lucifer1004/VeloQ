use veloq_vis::{VizLabelPolicy, VizRenderPolicy};

pub(super) fn request_fingerprint(
    start_ns: i64,
    end_ns: i64,
    tracks: &[String],
    highlights: &[String],
    render_policy: &VizRenderPolicy,
    label_policy: &VizLabelPolicy,
) -> String {
    let mut hash = Fnv1a64::new();
    hash.push(&start_ns.to_string());
    hash.push(&end_ns.to_string());
    for track in tracks {
        hash.push(track);
    }
    for highlight in highlights {
        hash.push(highlight);
    }
    hash.push(&render_policy.width_px.to_string());
    hash.push(&render_policy.max_tracks.to_string());
    hash.push(&render_policy.max_items.to_string());
    hash.push(&render_policy.min_interval_px.to_string());
    hash.push(&render_policy.aggregation.to_string());
    hash.push(&label_policy.mode.to_string());
    hash.push(&label_policy.min_label_px.to_string());
    hash.push(&label_policy.max_chars.to_string());
    format!("{:016x}", hash.finish())
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325,
        }
    }

    fn push(&mut self, value: &str) {
        for byte in value.as_bytes().iter().copied().chain([0]) {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}

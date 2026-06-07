use veloq_pytorch_data::{PytorchSidecar, sidecar_path_for_artifact};

pub(crate) fn path(artifact_dir: &str, sidecar: PytorchSidecar) -> String {
    sidecar_path_for_artifact(artifact_dir, sidecar)
        .display()
        .to_string()
}

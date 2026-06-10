use std::path::{Component, Path};

use crate::VisualizationError;
use crate::model::WrittenSvgArtifact;

pub fn write_svg_artifact(
    artifact_root: &Path,
    relative_dir: &Path,
    file_name: &str,
    svg: &str,
) -> Result<WrittenSvgArtifact, VisualizationError> {
    validate_relative_path(relative_dir)?;
    validate_svg_file_name(file_name)?;

    let dir = artifact_root.join(relative_dir);
    std::fs::create_dir_all(&dir).map_err(|source| VisualizationError::CreateArtifactDir {
        path: dir.display().to_string(),
        source,
    })?;

    let path = dir.join(file_name);
    let tmp_path = path.with_file_name(format!("{file_name}.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, svg).map_err(|source| VisualizationError::WriteArtifact {
        path: tmp_path.display().to_string(),
        source,
    })?;
    std::fs::rename(&tmp_path, &path).map_err(|source| {
        let _ = std::fs::remove_file(&tmp_path);
        VisualizationError::PublishArtifact {
            path: path.display().to_string(),
            source,
        }
    })?;

    let relative_path = relative_dir
        .join(file_name)
        .to_string_lossy()
        .replace('\\', "/");
    Ok(WrittenSvgArtifact {
        path,
        relative_path,
        format: "svg",
    })
}

fn validate_relative_path(path: &Path) -> Result<(), VisualizationError> {
    if path.is_absolute() {
        return Err(VisualizationError::UnsafeRelativePath);
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => return Err(VisualizationError::UnsafeRelativePath),
        }
    }
    Ok(())
}

fn validate_svg_file_name(file_name: &str) -> Result<(), VisualizationError> {
    let path = Path::new(file_name);
    let mut components = path.components();
    let Some(Component::Normal(_)) = components.next() else {
        return Err(VisualizationError::UnsafeSvgFileName);
    };
    if components.next().is_some() || path.extension().and_then(|e| e.to_str()) != Some("svg") {
        return Err(VisualizationError::UnsafeSvgFileName);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_svg_artifact_returns_portable_relative_path() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let written = write_svg_artifact(
            dir.path(),
            Path::new("figures/nsys"),
            "timeline.svg",
            "<svg/>",
        )?;
        assert_eq!(written.relative_path, "figures/nsys/timeline.svg");
        assert_eq!(written.format, "svg");
        assert!(written.path.exists());
        Ok(())
    }

    #[test]
    fn write_svg_artifact_rejects_parent_relative_dir() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let err = match write_svg_artifact(dir.path(), Path::new("../x"), "timeline.svg", "<svg/>")
        {
            Ok(_) => anyhow::bail!("unsafe path should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, VisualizationError::UnsafeRelativePath));
        Ok(())
    }
}

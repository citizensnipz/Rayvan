use std::path::Path;

pub fn path_exists(path: &Path) -> bool {
    path.exists()
}

pub fn is_directory(path: &Path) -> bool {
    path.is_dir()
}

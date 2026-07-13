#[derive(Debug, Clone, Copy)]
pub struct ProcessDescriptor {
    pub pid: u32,
}

pub fn current_process() -> ProcessDescriptor {
    ProcessDescriptor {
        pid: std::process::id(),
    }
}

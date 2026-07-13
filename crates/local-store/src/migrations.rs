pub const CURRENT_SCHEMA_VERSION: u32 = 1;

pub struct MigrationSet {
    pub version: u32,
    pub description: &'static str,
}

pub const INITIAL_MIGRATION: MigrationSet = MigrationSet {
    version: CURRENT_SCHEMA_VERSION,
    description: "Initial Rayvan local metadata schema",
};

use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::env;
use std::process;

struct ModelInspector {
    path: String,
    session: Session,
}

impl ModelInspector {
    fn new(path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .commit_from_file(path)?;

        Ok(ModelInspector {
            path: path.to_string(),
            session,
        })
    }

    fn print_details(&self) {
        println!("========================================");
        println!("Inspecting Model: {}", self.path);
        println!("========================================");

        println!("-> INPUTS:");
        for input in self.session.inputs() {
            println!("   Name: {}", input.name());
            // Dump the entire Outlet struct to the terminal
            println!("   Struct Dump: {:#?}", input); 
            println!("   ---");
        }

        println!("-> OUTPUTS:");
        for output in self.session.outputs() {
            println!("   Name: {}", output.name());
            // Dump the entire Outlet struct to the terminal
            println!("   Struct Dump: {:#?}", output);
            println!("   ---");
        }
        println!("========================================");
    }
}

fn main() -> Result<()> {
    // FIXED: Removed the `?` because commit() returns a bool, not a Result
    ort::init()
        .with_name("kornia-inspector")
        .commit();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -- <path_to_model.onnx>");
        process::exit(1);
    }

    let model_path = &args[1];

    match ModelInspector::new(model_path) {
        Ok(inspector) => inspector.print_details(),
        Err(e) => {
            eprintln!("FATAL ERROR: Failed to load model.");
            eprintln!("Reason: {:?}", e);
            process::exit(1);
        }
    }

    Ok(())
}
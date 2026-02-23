use anyhow::Result;
use ndarray::Array4; // The NumPy equivalent for Rust
use ort::{inputs, session::{builder::GraphOptimizationLevel, Session}, value::Value};
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
            println!("   Struct: {:#?}", input); 
            println!("   ---");
        }

        println!("-> OUTPUTS:");
        for output in self.session.outputs() {
            println!("   Name: {}", output.name());
            println!("   Struct: {:#?}", output);
            println!("   ---");
        }
        println!("========================================");
    }

    // NEW: The Inference Engine
    fn run_dummy_inference(&mut self) -> Result<()> {
        println!("========================================");
        println!("Running Inference with Dummy Data...");
        println!("========================================");

        // 1. Allocate memory: Create a 1x1x28x28 tensor filled with zeros
        let dummy_image = Array4::<f32>::zeros((1, 1, 28, 28));

        // 2a. Explicitly wrap the ndarray in an ONNX Value (Zero-copy)
        let input_tensor = Value::from_array(dummy_image)?;

        // 2b. Execute Graph: Pass the wrapped tensor
        let outputs = self.session.run(inputs!["Input3" => input_tensor])?;

        // 3. Extract Memory: Destructure the tuple into shape and data
        let (shape, data) = outputs["Plus214_Output_0"].try_extract_tensor::<f32>()?;

        // 4. Print the raw tensor data
        println!("Output Probabilities (Shape: {:?}):", shape);
        println!("{:?}", data);
        println!("========================================");
        Ok(())
    }
}

fn main() -> Result<()> {
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
        Ok(mut inspector) => {
            // First inspect the structure
            inspector.print_details();
            // Then run the dummy inference
            if let Err(e) = inspector.run_dummy_inference() {
                eprintln!("Inference failed: {:?}", e);
            }
        },
        Err(e) => {
            eprintln!("FATAL ERROR: Failed to load model.");
            eprintln!("Reason: {:?}", e);
            process::exit(1);
        }
    }

    Ok(())
}
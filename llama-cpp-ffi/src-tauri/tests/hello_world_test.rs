use llama_cpp_ffi_lib::core::{gemma_4_model_paths, Engine};

#[test]
fn gemma_4_model_loads_and_answers_text_prompt() -> Result<(), Box<dyn std::error::Error>> {
    let paths = gemma_4_model_paths()?;
    let mut engine = Engine::load(&paths.model, &paths.mmproj)?;

    let response = engine.prompt_text("Return a short hello greeting.")?;

    assert!(
        !response.trim().is_empty(),
        "expected non-empty response from text prompt"
    );
    Ok(())
}

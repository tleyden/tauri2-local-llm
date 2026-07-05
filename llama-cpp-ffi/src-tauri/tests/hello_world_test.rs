use llama_cpp_ffi_lib::core::{gemma_4_model_paths, Engine};

#[test]
fn gemma_4_model_loads_and_answers_text_prompt() -> Result<(), Box<dyn std::error::Error>> {
    let paths = gemma_4_model_paths()?;
    let mut engine = Engine::load(&paths.model, &paths.mmproj)?;

    let response = engine.prompt_text_with_stats("Return a short hello greeting.")?;

    assert!(
        !response.text.trim().is_empty(),
        "expected non-empty response from text prompt"
    );
    assert!(
        response.input_tokens > 0,
        "expected text prompt to report input tokens"
    );
    assert!(
        response.decode_tokens_per_second() > 0.0,
        "expected text prompt decode tokens/sec to be positive"
    );
    println!(
        "text benchmark: input_tokens={} generated_tokens={} prefill_seconds={:.2} decode_seconds={:.2} decode_tokens_per_second={:.2} total_seconds={:.2} total_tokens_per_second={:.2}",
        response.input_tokens,
        response.generated_tokens,
        response.prefill_elapsed.as_secs_f64(),
        response.decode_elapsed.as_secs_f64(),
        response.decode_tokens_per_second(),
        response.elapsed.as_secs_f64(),
        response.tokens_per_second()
    );
    Ok(())
}

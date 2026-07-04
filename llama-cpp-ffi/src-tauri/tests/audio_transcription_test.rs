use std::path::Path;

use llama_cpp_ffi_lib::core::{gemma_4_model_paths, Engine};

#[test]
fn gemma_4_native_audio_transcribes_hello_world() -> Result<(), Box<dyn std::error::Error>> {
    let paths = gemma_4_model_paths()?;
    let mut engine = Engine::load(&paths.model, &paths.mmproj)?;

    assert!(
        engine.support_audio(),
        "expected mtmd context to report audio support"
    );
    assert_eq!(
        engine.audio_sample_rate(),
        Some(16_000),
        "expected mtmd context audio sample rate to be 16000 Hz"
    );

    let audio_path = Path::new("../testdata/test_tts.wav");
    let response = engine.prompt_audio(audio_path, "Transcribe this audio exactly.")?;

    assert!(
        response.to_lowercase().contains("hello world"),
        "expected transcript to contain 'hello world', got: {response:?}"
    );
    Ok(())
}

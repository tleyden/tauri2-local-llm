use std::path::Path;

use llama_cpp_ffi_lib::core::{gemma_4_model_paths, Engine};

#[test]
fn gemma_4_native_audio_returns_expected_process_management_answer(
) -> Result<(), Box<dyn std::error::Error>> {
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

    assert_major_phrases(
        &response,
        &[
            "classic scenario",
            "major issue",
            "modern software development",
            "resource leakage",
            "improper process management",
            "desktop application",
            "operating system",
            "associated processes",
            "threads",
            "background tasks",
            "poorly designed",
            "zombie processes",
            "background workers",
            "receive the signal",
            "shut down",
            "breakdown of why this happens",
        ],
    );
    Ok(())
}

fn assert_major_phrases(response: &str, expected_phrases: &[&str]) {
    let normalized_response = normalize_text(response);

    for phrase in expected_phrases {
        let normalized_phrase = normalize_text(phrase);
        assert!(
            normalized_response.contains(&normalized_phrase),
            "expected response to contain phrase {phrase:?}, got: {response:?}"
        );
    }
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

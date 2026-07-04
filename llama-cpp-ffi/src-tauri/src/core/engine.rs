use std::ffi::CString;
use std::io;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MODEL_DIR_NAME: &str = "gemma-4-12b-it-Q5_K_S";
const MAX_GENERATED_TOKENS: usize = 96;
const CONTEXT_SIZE: u32 = 8192;
const BATCH_SIZE: u32 = 1024;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub mmproj: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PromptResponse {
    pub text: String,
    pub input_tokens: usize,
    pub input_positions: i32,
    pub generated_tokens: usize,
    pub elapsed: Duration,
}

impl PromptResponse {
    pub fn tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.elapsed.as_secs_f64()
    }
}

#[derive(Debug)]
pub struct Engine {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    mtmd: MtmdContext,
}

impl Engine {
    pub fn load(model_path: impl AsRef<Path>, mmproj_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let mmproj_path = mmproj_path.as_ref();

        ensure_file(model_path, "model")?;
        ensure_file(mmproj_path, "mmproj")?;

        let backend = backend()?;
        let mut model_params = LlamaModelParams::default();
        if backend.supports_gpu_offload() {
            model_params = model_params.with_n_gpu_layers(999);
        }

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;
        let mtmd_params = MtmdContextParams {
            use_gpu: backend.supports_gpu_offload(),
            print_timings: false,
            n_threads: default_thread_count(),
            media_marker: CString::new(mtmd_default_marker())?,
            image_min_tokens: -1,
            image_max_tokens: -1,
        };
        let mtmd = MtmdContext::init_from_file(path_to_str(mmproj_path)?, &model, &mtmd_params)?;

        Ok(Self {
            backend,
            model,
            mtmd,
        })
    }

    pub fn support_audio(&self) -> bool {
        self.mtmd.support_audio()
    }

    pub fn audio_sample_rate(&self) -> Option<u32> {
        self.mtmd.get_audio_sample_rate()
    }

    pub fn prompt_text(&mut self, prompt: &str) -> Result<String> {
        let prompt = self.chat_prompt(prompt)?;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;
        let mut ctx = self.new_context()?;
        let mut batch = LlamaBatch::get_one(&tokens)?;
        ctx.decode(&mut batch)?;

        Ok(generate_response(&self.model, &mut ctx, tokens.len(), MAX_GENERATED_TOKENS)?.text)
    }

    pub fn prompt_audio(&mut self, wav_path: impl AsRef<Path>, prompt: &str) -> Result<String> {
        Ok(self.prompt_audio_with_stats(wav_path, prompt)?.text)
    }

    pub fn prompt_audio_with_stats(
        &mut self,
        wav_path: impl AsRef<Path>,
        prompt: &str,
    ) -> Result<PromptResponse> {
        let wav_path = wav_path.as_ref();
        ensure_file(wav_path, "audio")?;
        let started_at = Instant::now();

        let audio = MtmdBitmap::from_file(&self.mtmd, path_to_str(wav_path)?, false)?;
        let content = format!("{} {}", mtmd_default_marker(), prompt);
        let prompt = self.chat_prompt(&content)?;
        let chunks = self.mtmd.tokenize(
            MtmdInputText {
                text: prompt,
                add_special: true,
                parse_special: true,
            },
            &[&audio],
        )?;
        let input_tokens = chunks.total_tokens();
        let input_positions = chunks.total_positions();

        let mut ctx = self.new_context()?;
        let n_past = chunks.eval_chunks(
            &self.mtmd,
            &ctx,
            0,
            0,
            i32::try_from(BATCH_SIZE).expect("batch size fits in i32"),
            true,
        )?;

        let mut response = generate_response(
            &self.model,
            &mut ctx,
            usize::try_from(n_past).unwrap_or(0),
            MAX_GENERATED_TOKENS,
        )?;
        response.input_tokens = input_tokens;
        response.input_positions = input_positions;
        response.elapsed = started_at.elapsed();
        Ok(response)
    }

    fn chat_prompt(&self, user_content: &str) -> Result<String> {
        Ok(format!(
            "<|turn>user\n{user_content}<turn|>\n<|turn>model\n"
        ))
    }

    fn new_context(&self) -> Result<llama_cpp_2::context::LlamaContext<'_>> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            .with_n_batch(BATCH_SIZE)
            .with_n_ubatch(BATCH_SIZE)
            .with_n_threads(default_thread_count())
            .with_n_threads_batch(default_thread_count());

        Ok(self.model.new_context(self.backend, params)?)
    }
}

pub fn gemma_4_model_paths() -> Result<ModelPaths> {
    let dir = project_root()?.join(".models").join(MODEL_DIR_NAME);
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("expected model directory at {}", dir.display()),
        )
        .into());
    }

    let ggufs = gguf_files(&dir)?;
    let model = ggufs
        .iter()
        .find(|path| !file_name_lower(path).contains("mmproj"))
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no non-mmproj GGUF model file found in {}", dir.display()),
            )
        })?;
    let mmproj = ggufs
        .iter()
        .find(|path| file_name_lower(path).contains("mmproj"))
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no mmproj GGUF file found in {}", dir.display()),
            )
        })?;

    Ok(ModelPaths { model, mmproj })
}

fn generate_response(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    mut n_past: usize,
    max_tokens: usize,
) -> Result<PromptResponse> {
    let mut sampler = LlamaSampler::greedy();
    let mut output = Vec::new();
    let mut generated_tokens = 0;

    for _ in 0..max_tokens {
        let token = sampler.sample(ctx, -1);
        if model.is_eog_token(token) {
            break;
        }

        sampler.accept(token);
        generated_tokens += 1;
        output.extend(model.token_to_piece_bytes(token, 32, false, None)?);

        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, i32::try_from(n_past)?, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_past += 1;
    }

    Ok(PromptResponse {
        text: String::from_utf8_lossy(&output).into_owned(),
        input_tokens: 0,
        input_positions: 0,
        generated_tokens,
        elapsed: Duration::ZERO,
    })
}

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }

    let backend = LlamaBackend::init()?;
    let _ = BACKEND.set(backend);
    Ok(BACKEND
        .get()
        .expect("backend was just initialized and stored"))
}

fn project_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "src-tauri manifest directory has no parent project root",
            )
            .into()
        })
}

fn gguf_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_gguf_files(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_gguf_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, files)?;
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            files.push(path);
        }
    }
    Ok(())
}

fn file_name_lower(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_lowercase()
}

fn ensure_file(path: &Path, label: &str) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("expected {label} file at {}", path.display()),
    )
    .into())
}

fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path is not valid UTF-8: {}", path.display()),
        )
        .into()
    })
}

fn default_thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(i32::MAX as usize) as i32
}

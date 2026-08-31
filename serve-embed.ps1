# Embedding server. Leave this running while you index.
#
# Every path and filename here can be overridden by an environment variable,
# so the same checkout runs on more than one machine without being edited.
# The defaults match config.py, which reads the same variable names.
#
#   $env:MODELS_DIR   = "D:\models"            # default: %USERPROFILE%\models
#   $env:EMBED_GGUF   = "bge-m3-Q8_0.gguf"
#   $env:LLAMA_EXTRA  = "--device CUDA0"       # backend-specific extras

$Models = if ($env:MODELS_DIR) { $env:MODELS_DIR } else { Join-Path $env:USERPROFILE "models" }
$Name   = if ($env:EMBED_GGUF) { $env:EMBED_GGUF } else { "bge-m3-Q8_0.gguf" }
$Server = Join-Path $Models "llama-server\llama-server.exe"
$Gguf   = Join-Path $Models $Name
$Extra  = if ($env:LLAMA_EXTRA) { $env:LLAMA_EXTRA -split ' ' } else { @() }

if (-not (Test-Path $Server)) { Write-Error "No llama-server.exe at $Server"; exit 1 }
if (-not (Test-Path $Gguf))   { Write-Error "No model at $Gguf"; exit 1 }

# --embedding      exposes /v1/embeddings; without it the endpoint 404s
# --pooling cls    BGE models pool on the CLS token. Get this wrong and you
#                  still get vectors -- just consistently worse ones, with no
#                  error to tell you. This is the flag most worth checking.
# -ub 4096         micro-batch must be >= the longest input; our chunks cap at
#                  ~2000 chars (~700 tokens), so this has comfortable margin
# -ngl 99          offload every layer to the GPU. Harmless on a CPU-only
#                  build -- it just has nowhere to put them.
& $Server `
    -m $Gguf `
    --host 127.0.0.1 --port 8081 `
    --embedding `
    --pooling cls `
    -c 8192 `
    -ub 4096 `
    -ngl 99 `
    @Extra

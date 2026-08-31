# Chat server. Only needed for stage 3 (generation), not for indexing.
#
# Overridable the same way as serve-embed.ps1:
#
#   $env:MODELS_DIR   = "D:\models"
#   $env:CHAT_GGUF    = "Qwen3-8B-Q4_K_M.gguf"
#   $env:LLAMA_EXTRA  = "--device CUDA1"       # backend-specific extras

$Models = if ($env:MODELS_DIR) { $env:MODELS_DIR } else { Join-Path $env:USERPROFILE "models" }
$Name   = if ($env:CHAT_GGUF)  { $env:CHAT_GGUF }  else { "Qwen3-8B-Q4_K_M.gguf" }
$Server = Join-Path $Models "llama-server\llama-server.exe"
$Gguf   = Join-Path $Models $Name
$Extra  = if ($env:LLAMA_EXTRA) { $env:LLAMA_EXTRA -split ' ' } else { @() }

if (-not (Test-Path $Server)) { Write-Error "No llama-server.exe at $Server"; exit 1 }
if (-not (Test-Path $Gguf))   { Write-Error "No model at $Gguf"; exit 1 }

# --jinja    use the model's own chat template from the GGUF metadata. Without
#            it llama-server falls back to a generic template and Qwen3's
#            thinking switch is silently ignored -- index.py --doctor detects
#            exactly this and tells you.
# -c 8192    context. RAG prompts are long: 6-8 chunks plus the question plus
#            the answer. ask.py reads the real value from the server, so
#            raising it here is picked up automatically.
# -ngl 99    offload every layer to the GPU.
& $Server `
    -m $Gguf `
    --host 127.0.0.1 --port 8080 `
    --jinja `
    -c 8192 `
    -ngl 99 `
    @Extra

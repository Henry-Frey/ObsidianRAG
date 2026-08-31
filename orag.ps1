# Terminal front-end. Dot-source it to get an `orag` command anywhere:
#
#     . C:\Users\Who\ObsidianRAG\orag.ps1
#     orag ask "what did we decide about the rollout"
#
# To have it every session, add that dot-source line to your profile:
#
#     notepad $PROFILE        (create the file if it does not exist)
#
# Machine-specific settings belong in environment variables, not in this file,
# so the same checkout works on the laptop and the desktop:
#
#     $env:OBSIDIAN_VAULT = "D:\Notes"
#     $env:MODELS_DIR     = "D:\models"
#     $env:EMBED_GGUF     = "bge-m3-Q8_0.gguf"
#     $env:CHAT_GGUF      = "Qwen3-8B-Q4_K_M.gguf"
#     $env:LLAMA_EXTRA    = "--device CUDA0"

$script:OragRoot = $PSScriptRoot

function orag {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)][string]$Command,
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
    )

    if (-not $Command) { $Command = "help" }
    $root = $script:OragRoot
    if (-not $Rest) { $Rest = @() }

    switch ($Command) {
        "ask"     { & python (Join-Path $root "ask.py")    @Rest }
        "search"  { & python (Join-Path $root "search.py") @Rest }
        "index"   { & python (Join-Path $root "index.py")  @Rest }
        "doctor"  { & python (Join-Path $root "index.py")  --doctor @Rest }
        "stats"   { & python (Join-Path $root "index.py")  --stats  @Rest }
        "links"   { & python (Join-Path $root "index.py")  --links  @Rest }
        "dates"   { & python (Join-Path $root "index.py")  --dates  @Rest }
        "show"    { & python (Join-Path $root "index.py")  --show   @Rest }

        # Each server blocks its terminal, so these open their own window.
        "serve" {
            $which = if ($Rest.Count) { $Rest[0] } else { "" }
            switch ($which) {
                "embed" { Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $root "serve-embed.ps1") }
                "chat"  { Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $root "serve-chat.ps1") }
                "both"  {
                    Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $root "serve-embed.ps1")
                    Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $root "serve-chat.ps1")
                }
                default { Write-Host "orag serve embed | chat | both" }
            }
        }

        "env" {
            Write-Host "root        : $root"
            Write-Host "vault       : $(if ($env:OBSIDIAN_VAULT) { $env:OBSIDIAN_VAULT } else { '(unset -- config.py default)' })"
            Write-Host "models      : $(if ($env:MODELS_DIR) { $env:MODELS_DIR } else { Join-Path $env:USERPROFILE 'models' })"
            Write-Host "embed gguf  : $(if ($env:EMBED_GGUF) { $env:EMBED_GGUF } else { 'bge-m3-Q8_0.gguf' })"
            Write-Host "chat gguf   : $(if ($env:CHAT_GGUF) { $env:CHAT_GGUF } else { 'Qwen3-8B-Q4_K_M.gguf' })"
            Write-Host "llama extra : $env:LLAMA_EXTRA"
        }

        default {
            Write-Host @"
orag <command> [args]

  ask     "question"      answer from your notes, with citations
  search  "query"         show the retrieved chunks only
  index                   build or update the index (--rebuild, --no-embed, --dry-run)
  doctor                  check models and both llama-servers
  stats | links | dates   index reports
  show    "note.md"       print how one note was chunked
  serve   embed|chat|both start a llama-server in its own window
  env                     show which paths and models are in effect

Anything after the command is passed straight through, so
`orag ask -k 4 --since 01.01.26 "..."` works.
"@
        }
    }
}

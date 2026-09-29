# Third-party notices

## STS2 MCP

This project uses and modifies [STS2 MCP](https://github.com/Gennadiyev/STS2MCP), an open-source mod and local API for Slay the Spire 2. The distributed `STS2_MCP.dll` is a locally modified build; its source files at the repository root are kept as local development references and are not included in this repository.

The upstream repository's license file states:

> Copyright 2026 Yikun Ji (Kunologist)

STS2 MCP is licensed under the MIT License. The complete upstream notice is reproduced below. The local changes are not official upstream changes.

### Changes in the distributed MCP build

- Combat readiness checks consult the game's live hand-input guard and action-disabled state, while retaining the player-phase check as a fallback for remote players.
- The state payload includes additional combat hand/phase readiness fields, the complete run deck, and card-upgrade preview details used by strategic decisions.

### MIT License for STS2 MCP

Copyright 2026 Yikun Ji (Kunologist)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Qwen3-4B and the strategic LoRA adapter

The base model for `artifacts/models/qwen3_4b_lora/` is [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), published by Qwen under the Apache License 2.0. The base model weights are not redistributed here; only the project-trained LoRA adapter and tokenizer files are included. The official model repository provides the [license text](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE).

The adapter is a project-trained artifact for this game agent. It is provided as a LoRA adapter and requires the separately obtained Qwen3-4B base model for inference. Review the upstream license and model terms before redistributing or using the base model.

## PunchOffInstantFix

`mods/PunchOffInstantFix/` contains a project-specific helper mod for training reliability. It is maintained by this project and is separate from STS2 MCP.

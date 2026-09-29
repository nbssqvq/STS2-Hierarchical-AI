# Third-party notices

## STS2 MCP

This project uses and adapts [STS2 MCP](https://github.com/Gennadiyev/STS2MCP), an open-source mod and local API for Slay the Spire 2. The repository includes its C# source (including locally modified files) and a locally built release DLL.

The upstream repository's license file states:

> Copyright 2026 Yikun Ji (Kunologist)

The upstream project is licensed under the MIT License. The full notice is reproduced below. Changes in the local integration are identified here and are not official upstream changes.

### Local modifications

- `McpMod.Helpers.cs`: combat readiness checks consult the game's live hand input guard and action-disabled state, while retaining the player-phase check as a fallback for remote players.
- `McpMod.StateBuilder.cs`: the state payload includes additional combat hand/phase readiness fields, the complete run deck, and card-upgrade preview details used by strategic decisions.

## PunchOffInstantFix

`mods/PunchOffInstantFix/` contains a project-specific helper mod for training reliability. It is maintained in this project and is separate from STS2 MCP.

## MIT License for STS2 MCP

Copyright 2026 Yikun Ji (Kunologist)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

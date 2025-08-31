# GitHub AI Integration for ChaosEater

This guide shows how to set up and use GitHub AI models with ChaosEater.

## Setup

### 1. Environment Variables

Set your GitHub token as an environment variable:

**Bash/Linux/macOS:**
```bash
export GITHUB_TOKEN="your-github-token-goes-here"
```

**PowerShell:**
```powershell
$Env:GITHUB_TOKEN="your-github-token-goes-here"
```

**Windows Command Prompt:**
```cmd
set GITHUB_TOKEN=your-github-token-goes-here
```

### 2. Available GitHub AI Models

The following GitHub AI models are now supported:
- `github/gpt-4o` - Latest GPT-4o model via GitHub AI
- `github/gpt-4o-mini` - Smaller, faster GPT-4o model via GitHub AI

### 3. Using in ChaosEater Demo

1. Run the ChaosEater demo:
   ```bash
   streamlit run ChaosEater_demo.py
   ```

2. In the sidebar, select one of the GitHub models:
   - `github/gpt-4o`
   - `github/gpt-4o-mini`

3. Enter your GitHub token in the "GitHub Token" field

4. Optionally modify the "GitHub Base URL" (default: `https://models.github.ai/inference`)

5. Proceed with your chaos engineering experiments as usual

### 4. Pricing

GitHub AI models are currently **free** to use, so no billing costs are shown in the usage display.

### 5. Code Example

```python
from chaos_eater.utils.llms import load_llm
import os

# Set your GitHub token
os.environ["GITHUB_TOKEN"] = "your-github-token"

# Load GitHub AI model
llm = load_llm(
    model_name="github/gpt-4o-mini",
    temperature=0.0,
    seed=42,
    github_base_url="https://models.github.ai/inference"  # Optional: custom base URL
)

# Use with ChaosEater
from chaos_eater.chaos_eater import ChaosEater
from chaos_eater.ce_tools.ce_tool import CEToolType, CETool

chaoseater = ChaosEater(
    llm=llm,
    ce_tool=CETool.init(CEToolType.chaosmesh),
    work_dir="sandbox",
    namespace="chaos-eater"
)
```

## Notes

- GitHub AI uses the same OpenAI-compatible API as regular OpenAI models
- Token usage is tracked the same way as OpenAI models
- All existing ChaosEater functionality works seamlessly with GitHub AI models
- Models are currently free but this may change in the future

## Troubleshooting

1. **Invalid token error**: Make sure your GitHub token has the necessary permissions for AI model access
2. **Connection issues**: Verify the base URL is correct and accessible
3. **Model not found**: Ensure the model name is correctly formatted as `github/model-name`
4. **Pydantic validation errors**: The GitHub AI integration includes robust validation for JSON responses to handle any format differences between providers

### Common Issues with GitHub AI Integration

**ValidationError: str type expected**
This error occurs when the LLM response format doesn't match expectations. The updated code now handles:
- `None` responses from interrupted streams
- JSON objects that need string extraction
- Empty or malformed responses
- Different response formats between GitHub AI and OpenAI

The system will automatically provide fallback error messages if the LLM response cannot be processed properly.

## Testing

Run the test script to verify your setup:
```bash
python test_github_ai.py
```
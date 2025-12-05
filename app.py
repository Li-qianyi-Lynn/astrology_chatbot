# Astrology Chatbot - HuggingFace Space Deployment
# This is the streamlined version of the Colab notebook

import torch
import json
import os
import gradio as gr
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback to old punkt if punkt_tab fails
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

print("=" * 70)
print("🌟 Astrology Chatbot - Loading Model")
print("=" * 70)

# Configuration
class Config:
    MODEL_NAME = 'microsoft/phi-2'
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    MAX_LENGTH = 512
    FINAL_MODEL_DIR = './final_phi2_astro_model_v2'

config = Config()

# Load model with 4-bit quantization
print("Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

try:
    # Try to load fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        config.FINAL_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.FINAL_MODEL_DIR, trust_remote_code=True)
    print("✓ Fine-tuned model loaded successfully!")
except Exception as e:
    print(f"Fine-tuned model not found, loading base model: {e}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    print("✓ Base model loaded successfully!")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print(f"✓ Device: {model.device}")
if torch.cuda.is_available():
    print(f"✓ GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Zodiac options
ZODIAC_OPTIONS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                  'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
TOPIC_OPTIONS = ['Love', 'Career', 'Friendship', 'Growth', 'Identity']

def fix_encoding_issue(text):
    """Remove encoding artifacts"""
    text = text.replace('Ã¯Â¿Â½', '')
    return text

def chat_with_astro(message, zodiac, topic, history, desired_max_tokens, temperature):
    """Generate astrological response"""
    if not message.strip():
        return history

    prompt = f"""### Instruction:
You are an astrology advisor. The user is a {zodiac}. Provide personalized advice based on their zodiac traits.

### User ({zodiac}, asking about {topic}):
{message}

### Assistant:
"""

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    generation_max_tokens = desired_max_tokens + 50

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_response = full_text.split('### Assistant:')[-1].strip()
    
    if '### ' in raw_response:
        raw_response = raw_response.split('### ')[0].strip()

    raw_response = fix_encoding_issue(raw_response)

    # Post-process to ensure complete sentences
    sentences = sent_tokenize(raw_response)
    final_response_parts = []
    current_token_length = 0

    for sentence in sentences:
        sentence_token_count = len(tokenizer.encode(sentence))

        if current_token_length + sentence_token_count <= desired_max_tokens or not final_response_parts:
            final_response_parts.append(sentence)
            current_token_length += sentence_token_count
        else:
            break

    response = " ".join(final_response_parts).strip()
    if not response and raw_response:
        response = raw_response

    # Format for Gradio Chatbot (new format: dicts with 'role' and 'content')
    user_msg = f"[{zodiac}] {message}"
    # Convert history to new format if needed
    if history and len(history) > 0 and isinstance(history[0], list):
        # Convert old format [["user", "assistant"]] to new format
        new_history = []
        for msg_pair in history:
            if isinstance(msg_pair, list) and len(msg_pair) == 2:
                new_history.append({"role": "user", "content": msg_pair[0]})
                new_history.append({"role": "assistant", "content": msg_pair[1]})
        history = new_history
    
    return history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response}
    ]

def submit_and_clear(message, zodiac, topic, history, max_tok, temp):
    """Handle submit and clear input"""
    new_history = chat_with_astro(message, zodiac, topic, history, max_tok, temp)
    return new_history, ""

# Create Gradio interface
with gr.Blocks(title="Astrology Chatbot") as demo:
    gr.Markdown("# 🌟 Astrology Chatbot\n### Powered by Fine-tuned Phi-2 with Zodiac Personalization")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450, label="Chat")
            
            with gr.Row():
                zodiac_select = gr.Dropdown(
                    choices=ZODIAC_OPTIONS, 
                    value='Aries', 
                    label='Your Zodiac Sign'
                )
                topic_select = gr.Dropdown(
                    choices=TOPIC_OPTIONS, 
                    value='Love', 
                    label='Topic'
                )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask your astrological question...", 
                    label="Your Message", 
                    scale=4
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat 🗑️")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            max_tokens = gr.Slider(
                50, 300, 
                value=150, 
                step=10, 
                label="Response Length (tokens)"
            )
            temperature = gr.Slider(
                0.1, 1.5, 
                value=0.7, 
                step=0.1, 
                label="Creativity"
            )

    # Connect events
    send_btn.click(
        submit_and_clear, 
        [msg, zodiac_select, topic_select, chatbot, max_tokens, temperature], 
        [chatbot, msg]
    )
    msg.submit(
        submit_and_clear, 
        [msg, zodiac_select, topic_select, chatbot, max_tokens, temperature], 
        [chatbot, msg]
    )
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    is_spaces = os.environ.get("SPACE_ID") is not None
    demo.queue().launch(share=not is_spaces)